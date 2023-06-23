#!/usr/bin/env python3
from pathlib import Path
import os
import re
import textwrap
import numpy as np
import astropy.units as u
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from library import lightcurves, plot, jktebop, utility


# ---------------------------------------------------------------------
# Handle setting up and interpreting the ingest target configuration
# ---------------------------------------------------------------------
args = utility.set_up_argument_parser().parse_args()
if args.new_file:
    utility.write_ingest_config(
        args.new_file,
        utility.new_ingest_config("New Sys", **{
            "polies": { "term": "sf", "degree": 1, "gap_threshold": 0.5 },
            "fitting_params": { "dummy_token": "dummy value" }
        }))
    quit()
else:
    config = utility.read_ingest_config(args.file)

detrend_clip = 0.5
ml_phase_bins = 1024
sys_name = config.sys_name or config.target

# Set up output locations and file prefixing (& make sure chars are safe subset)
prefix = re.sub(r'[^\w\d-]', '_', config.prefix or sys_name.lower())
output_dir = Path(config.output_dir or f"./staging/{prefix}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\nWill write files prefixed '{prefix}' to directory {output_dir}")


# ---------------------------------------------------------------------
# Use MAST to DL any timeseries/light-curves for the system/sector(s)
# ---------------------------------------------------------------------
lcs = LightCurveCollection([])
results = lk.search_lightcurve(target=config.target, sector=config.sectors,
                               mission="TESS", author="SPOC",
                               exptime=config.exptime)
if results:
    lcs = results.download_all(download_dir=f"{output_dir}", cache=True, 
                               flux_column=config.flux_column, 
                               quality_bitmask=config.quality_bitmask)

# This will load the acquired data directly from the cache so we're not 
# dependent on search/download above. Useful for testing/dev or if MAST down.
#fits_files = sorted(output_dir.rglob("tess*_lc.fits"))
#lcs = LightCurveCollection([
#    lk.read(f"{f}", 
#            flux_column=config.flux_column, 
#            quality_bitmask=config.quality_bitmask) 
#    for f in fits_files
#])

print(f"\nFound {len(lcs)} light-curve(s) for {sys_name} sectors {lcs.sector}.")
if len(lcs):
    # TODO: Arrange a proper location from which to pick up the model.
    # Suppress annoying TF info messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    model = load_model("./cnn_model.h5")


# ---------------------------------------------------------------------
# Process each of the system's light-curves in turn
# ---------------------------------------------------------------------
for lc in lcs:  
    sector = lc.meta['SECTOR']
    file_stem = f"{prefix}_s{sector:0>4}"
    int_time = (lc.meta["INT_TIME"] + lc.meta["READTIME"]) * u.min

    narrative = f"Processing {len(lc)} row(s) of {config.flux_column} data "\
        f"(meeting the quality bitmask of {config.quality_bitmask}) "\
        f"for {sys_name} ({lc.meta['OBJECT']}) sector {sector} "\
        f"(camera {lc.meta['CAMERA']}/CCD {lc.meta['CCD']}). This covers the "\
        f"period of {lc.meta['DATE-OBS']} to {lc.meta['DATE-END']} "\
        f"with an integration time of {int_time.to(u.s)}."
    print()
    print("\n".join(textwrap.wrap(narrative, 75)))


    # ---------------------------------------------------------------------
    # Apply any additional data masks (NaN/negative fluxes, date range)
    # ---------------------------------------------------------------------
    # We'll need to explicitly mask NaNs & negative fluxes as they may still be 
    # present (only hardest seems to exclude them) and they'll break detrending.
    # Similarly, mask user defined ranges of suspect quality/abberations.
    filter_mask = np.isnan(lc.flux)
    filter_mask |= lc.flux < 0
    print(f"NaN/negative flux masks affect {sum(filter_mask.unmasked)} row(s).")

    if config.quality_masks and len(config.quality_masks):
        print(f"Applying time range quality masks")
        for qm in config.quality_masks:
            filter_mask |= lightcurves.mask_from_time_range(lc, qm)

    lc = lc[~filter_mask]
    print(f"Additional filters mask {sum(filter_mask.unmasked)} "\
            f"row(s) leaving {len(lc)}.")


    # ---------------------------------------------------------------------
    # Optional binning of the light-curve
    # ---------------------------------------------------------------------
    if config.bin_time and config.bin_time > 0:
        bin_time = config.bin_time * u.s
        if int_time.to(u.s) >= bin_time:
            print(f"Light-curve already in bins >= {bin_time}")
        else:
            print(f"Binning light-curve to bins of {bin_time} duration.")
            lc = lc.bin(time_bin_size=bin_time, aggregate_func=np.nanmean)
            lc = lc[~np.isnan(lc.flux)] # Binning may have re-introduced NaNs
            print(f"After binning light-curve has {len(lc)} rows.")


    # ---------------------------------------------------------------------
    # Convert to relative mags with fitted polynomial detrending
    # ---------------------------------------------------------------------
    print(f"Detrending & 'zeroing' magnitudes by subtracting polynomial.")
    lightcurves.append_magnitude_columns(lc, "delta_mag", "delta_mag_err")
    lc["delta_mag"] -= lightcurves.fit_polynomial(lc.time, 
                                                  lc["delta_mag"], 
                                                  degree=2, 
                                                  res_sigma_clip=detrend_clip, 
                                                  reset_const_coeff=False)


    # ---------------------------------------------------------------------
    # Find the primary epoch and orbital period
    # ---------------------------------------------------------------------
    (primary_epoch, primary_epoch_ix) = lightcurves.find_primary_epoch(lc)
    print(f"The primary epoch for sector {sector} is at JD {primary_epoch.jd}")
    if config.period:
        period = config.period * u.d
        print(f"An orbital period of {period} was specified by the user.")
    else:
        period = lightcurves.find_period(lc, primary_epoch)
        print(f"No period specified. Found {period} based on eclipse timings.")


    # ---------------------------------------------------------------------
    # Optionally plot the light-curve incl primary eclipse for diagnostics
    # ---------------------------------------------------------------------
    if config.plot_lc:
        ax = plot.plot_light_curve_on_axes(
            lc, title=f"Light-curve of {sys_name} sector {sector}")
        primary_mag = lc["delta_mag"][primary_epoch_ix]
        ax.scatter([primary_epoch.value], [primary_mag.value], zorder=-10,
                   marker="x", s=64., lw=.5, c="k", label="primary eclipse")
        plt.savefig(output_dir / (file_stem + "_lightcurve.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Phase fold the LC & interpolate on the fold to use for our estimates
    # ---------------------------------------------------------------------
    print(f"Phase folding the LC to get a sample curve for param estimation.")
    fold_lc = lightcurves.phase_fold_lc(lc, primary_epoch, period, 0.75)
    (phases, mags) = lightcurves.get_reduced_folded_lc(fold_lc, ml_phase_bins)


    # ---------------------------------------------------------------------
    # Optionally plot the folded LC overlaid with the interpolated one for diags
    # ---------------------------------------------------------------------
    if config.plot_fold:
        ax = plot.plot_folded_light_curve_on_axes(
            fold_lc, title=f"Folded light-curve of {sys_name} sector {sector}")
        ax.scatter(phases, mags, c="k", marker="+", 
                   s=8, alpha=.5, linewidth=.5, zorder=10)
        plt.savefig(output_dir / (file_stem + "_folded.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Use the ML model to estimate system parameters
    # ---------------------------------------------------------------------   
    # Now we can invoke the ML model to interpret the folded data & estimate
    # the parameters for JKTEBOP. Need the mag data for LCs in 
    # shape[#LCs, 1024, 1] giving predictions with shape[#LCs, #features]
    print(f"Estimating system parameters.")
    predictions = model.predict(np.array([np.transpose([mags])]), verbose=0)
    (rA_plus_rB, k, bA, inc, ecosw, esinw, J, L3) = predictions[0, :]

    # The directly predicted inc needs scaling up
    inc *= 100
    inc_calc = utility.calculate_inclination(bA, rA_plus_rB, k, ecosw, esinw)
    print(f"Inclination {inc:.6f} (prediction), {inc_calc:.6f} (calculation).")


    # ---------------------------------------------------------------------
    # Build polies before trimming so they're not affected by gaps from trimming
    # ---------------------------------------------------------------------    
    poly_instructions = jktebop.build_polies_for_lc(lc, config.polies)


    # ---------------------------------------------------------------------
    # Apply any user requests to trim the light-curves (for data reduction)
    # ---------------------------------------------------------------------
    if config.trim_masks and len(config.trim_masks) > 0:
        print(f"Applying requested trim masks to final light-curve")
        trim_mask = [False] * len(lc)
        for tm in config.trim_masks:
            trim_mask |= lightcurves.mask_from_time_range(lc, tm)
        lc = lc[~trim_mask]
        print(f"Trimming masks {sum(trim_mask)} row(s) leaving {len(lc)}.")

        if config.plot_lc:
            ax = plot.plot_light_curve_on_axes(
                lc, title=f"Trimmed light-curve of {sys_name} sector {sector}")
            plt.savefig(output_dir / (file_stem + "_trimmed.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Generate JKTEBOP .dat and .in file for task3.
    # ---------------------------------------------------------------------
    overrides = config.fitting_params or {}
    params = {
        "rA_plus_rB": rA_plus_rB,
        "k": k,
        "inc": inc,
        "qphot": 0.,
        "esinw": esinw,
        "ecosw": ecosw,
        "J": J,
        "L3": L3,
        "L3_fit": 1,
        "LD_A": "quad",
        "LD_B": "quad",
        "LD_A1": 0.25,
        "LD_B1": 0.25,
        "LD_A1_fit": 1,
        "LD_B1_fit": 1,
        "LD_A2": 0.22,
        "LD_B2": 0.22,
        "LD_A2_fit": 0,
        "LD_B2_fit": 0,
        "reflA": 0.,
        "reflB": 0.,
        "period": period.to(u.d).value,
        "primary_epoch": primary_epoch.jd - 2.4e6,

        **overrides
    }

    jktebop.write_task3_in_file(output_dir / (file_stem + ".in"), 
                                poly_instructions, **params)
    jktebop.write_data_to_dat_file(lc, output_dir / (file_stem + ".dat"))
    print(f"JKTEBOP dat & in files were written to {output_dir.resolve()}")