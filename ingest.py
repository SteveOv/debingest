#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import textwrap
import json
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
ap = utility.set_up_argument_parser()
args = ap.parse_args()
if args.new_file is not None:
    new_file = Path(args.new_file)
    utility.save_new_ingest_json(new_file, args)
    quit()
elif args.file:
    print(f"Configuring pipeline from {args.file} & any command line overrides")
    # Read the JSON file and use it as the basis of our target config but with
    # overrides from the command line to apply missing default values or 
    # otherwise override where non-default values have been given.
    with open(args.file, "r") as f:
        file_args = json.load(f)
        overrides = {k: v for k, v in vars(args).items() 
                     if k not in file_args or ap.get_default(k) != v}
        args = argparse.Namespace(**{**file_args, **overrides})
else:
    print(f"Configuring pipeline based on command line arguments")

utility.echo_ingest_parameters(args)
detrend_clip = 0.5
ml_phase_bins = 1024

sys_name = args.sys_name if args.sys_name else args.target
sys_file_label = "".join(c for c in sys_name if c not in r':*?"\/<>|').\
    lower(). \
    replace(' ', '_')
staging_dir = Path(f"./staging/{sys_file_label}")
staging_dir.mkdir(parents=True, exist_ok=True)
print(f"\nOutput will be written to {staging_dir}")

# ---------------------------------------------------------------------
# Use MAST to DL any timeseries/light-curves for the system/sector(s)
# ---------------------------------------------------------------------
lcs = LightCurveCollection([])
results = lk.search_lightcurve(target=args.target, sector=args.sectors,
                               mission=args.mission, author=args.author,
                               exptime=args.exptime)
if results:
    lcs = results.download_all(download_dir=f"{staging_dir}", cache=True, 
                               flux_column=args.flux_column, 
                               quality_bitmask=args.quality_bitmask)

# This will load the acquired data directly from the cache so we're not 
# dependent on search/download above. Useful for testing/dev or if MAST down.
#fits_files = sorted(staging_dir.rglob("tess*_lc.fits"))
#lcs = LightCurveCollection([
#    lk.read(f"{f}", 
#            flux_column=args.flux_column, 
#            quality_bitmask=args.quality_bitmask) 
#    for f in fits_files
#])

print(f"\nFound {len(lcs)} light-curves for {sys_name} sectors {lcs.sector}.")
if len(lcs):
    # TODO: Arrange a proper location from which to pick up the model.
    # Suppress annoying TF info messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    model = load_model("./cnn_model.h5")


# ---------------------------------------------------------------------
# Process each of the system's light-curves in turn
# ---------------------------------------------------------------------
for lc in lcs:  
    sector = f"{lc.meta['SECTOR']:0>4}"
    tic = f"{lc.meta['OBJECT']}"
    int_time = (lc.meta["INT_TIME"] + lc.meta["READTIME"]) * u.min
    file_stem = f"{sys_file_label}_s{sector}"

    narrative = f"Processing {len(lc)} row(s) of {args.flux_column} data "\
        f"(meeting the quality bitmask of {args.quality_bitmask}) "\
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

    if args.quality_masks and len(args.quality_masks):
        print(f"Applying time range quality masks")
        for qm in args.quality_masks:
            filter_mask |= lightcurves.mask_from_time_range(lc, qm)

    lc = lc[~filter_mask]
    print(f"Additional filters mask {sum(filter_mask.unmasked)} "\
            f"row(s) leaving {len(lc)}.")


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
    if args.period is None:
        period = lightcurves.find_period(lc, primary_epoch)
        print(f"No period specified. Found {period} based on eclipse timings.")
    else:
        period = args.period * u.d
        print(f"An orbital period of {period} was specified by the user.")


    # ---------------------------------------------------------------------
    # Optionally plot the light-curve incl primary eclipse for diagnostics
    # ---------------------------------------------------------------------
    if args.plot_lc:
        ax = plot.plot_light_curve_on_axes(
            lc, title=f"{sys_name} sector {sector} light-curve")
        primary_mag = lc["delta_mag"][primary_epoch_ix]
        ax.scatter([primary_epoch.value], [primary_mag.value], zorder=-10,
                   marker="x", s=64., lw=.5, c="k", label="primary eclipse")
        plt.savefig(staging_dir / (file_stem + "_lightcurve.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Phase fold the LC & interpolate on the fold to use for our estimates
    # ---------------------------------------------------------------------
    print(f"Phase folding the LC to get a sample curve for param estimation.")
    fold_lc = lightcurves.phase_fold_lc(lc, primary_epoch, period, 0.75)
    (phases, mags) = lightcurves.get_reduced_folded_lc(fold_lc, ml_phase_bins)


    # ---------------------------------------------------------------------
    # Optionally plot the folded LC overlaid with the interpolated one for diags
    # ---------------------------------------------------------------------
    if args.plot_fold:
        ax = plot.plot_folded_light_curve_on_axes(fold_lc, column = "delta_mag",
                    title = f"Folded light-curve of {sys_name} sector {sector}")
        ax.scatter(phases, mags, c="k", marker="+", 
                   s=8, alpha=.5, linewidth=.5, zorder=10)
        plt.savefig(staging_dir / (file_stem + "_folded.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Use the ML model to estimate system parameters
    # ---------------------------------------------------------------------   
    # Now we can invoke the ML model to interpret the folded data & estimate
    # the parameters for JKTEBOP. Need the mag data in shape[1, 1024, 1]
    # Predictions for a single model will have shape[1, 7]
    print(f"Estimating system parameters.")
    predictions = model.predict(np.array([np.transpose([mags])]), verbose=0)
    (rA_plus_rB, k, bA, bB, ecosw, esinw, J) = predictions[0, :]
    inc = utility.calculate_inclination(bA, rA_plus_rB, k, ecosw, esinw)


    # ---------------------------------------------------------------------
    # Build polies before trimming so they're not affected by gaps from trimming
    # ---------------------------------------------------------------------    
    poly_instructions = jktebop.build_polies_for_lc(lc, args.polies)


    # ---------------------------------------------------------------------
    # Apply any user requests to trim the light-curves (for data reduction)
    # ---------------------------------------------------------------------
    if args.trim_masks is not None and len(args.trim_masks) > 0:
        print(f"Applying requested trim masks to final light-curve")
        trim_mask = [False] * len(lc)
        for tm in args.trim_masks:
            trim_mask |= lightcurves.mask_from_time_range(lc, tm)
        lc = lc[~trim_mask]
        print(f"Trimming masks {sum(trim_mask)} row(s) leaving {len(lc)}.")

        if args.plot_lc:
            ax = plot.plot_light_curve_on_axes(
                    lc, title=f"Trimmed {sys_name} sector {sector} light-curve")
            plt.savefig(staging_dir / (file_stem + "_trimmed.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Generate JKTEBOP .dat and .in file for task3.
    # ---------------------------------------------------------------------
    overrides = args.fitting_params if args.fitting_params else {}
    params = {
        "rA_plus_rB": rA_plus_rB,
        "k": k,
        "inc": inc,
        "qphot": 0.,
        "esinw": esinw,
        "ecosw": ecosw,
        "J": J,
        "L3": 0.,
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

    jktebop.write_task3_in_file(staging_dir / (file_stem + ".in"), 
                                poly_instructions, **params)
    jktebop.write_data_to_dat_file(lc, staging_dir / (file_stem + ".dat"))
    print(f"JKTEBOP dat & in files were written to {staging_dir.resolve()}")