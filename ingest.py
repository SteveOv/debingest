#!/usr/bin/env python3
"""Entry point for the ingest process."""
from pathlib import Path
import re
import textwrap
import numpy as np
import astropy.units as u
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt

from library import lightcurves, plot, jktebop, utility, estimator


# ---------------------------------------------------------------------
# Handle setting up and interpreting the ingest target configuration
# ---------------------------------------------------------------------
args = utility.set_up_argument_parser().parse_args()
if args.new_file:
    utility.write_ingest_config(
        args.file,
        utility.new_ingest_config("New Sys", **{
            "polies": { "term": "sf", "degree": 1, "gap_threshold": 0.5 },
            "fitting_params": { "dummy_token": "dummy value" }
        }))
    quit()
else:
    config = utility.read_ingest_config(args.file)

DETREND_CLIP = 0.5
ML_PHASE_BINS = 1024
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

print(f"\nFound {len(lcs)} light-curves for {sys_name} sectors {lcs.sector}.")
states = []   # This is where we populate our ongoing pipeline state per sector


# ---------------------------------------------------------------------
# Load LC, apply quality masks, optional binning, detrend and derive mags
# ---------------------------------------------------------------------
print("Initial ingest, masking, binning and derivation of delta-mags")
for lc in lcs:
    file_stem = f"{prefix}_s{lc.meta['SECTOR']:0>4}"
    int_time = (lc.meta["INT_TIME"] + lc.meta["READTIME"]) * u.min

    narrative = f"Read {len(lc)} row(s) of {config.flux_column} data "\
        f"(meeting the quality bitmask of {config.quality_bitmask}) "\
        f"for {sys_name} ({lc.meta['OBJECT']}) sector {lc.meta['SECTOR']} "\
        f"(camera {lc.meta['CAMERA']}/CCD {lc.meta['CCD']}). This covers the "\
        f"period of {lc.meta['DATE-OBS']} to {lc.meta['DATE-END']} "\
        f"with an integration time of {int_time.to(u.s)}."
    print()
    print("\n".join(textwrap.wrap(narrative, 75)))

    # We'll need to explicitly mask NaNs & negative fluxes as they may still be
    # present (only hardest seems to exclude them) and they'll break detrending.
    filter_mask = np.isnan(lc.flux)
    filter_mask |= lc.flux < 0
    print(f"NaN/negative flux masks affect {sum(filter_mask.unmasked)} row(s).")

    # Similarly, mask user defined ranges of suspect quality/abberations.
    if config.quality_masks and len(config.quality_masks):
        print("Applying time range quality masks")
        for qm in config.quality_masks:
            filter_mask |= lightcurves.mask_from_time_range(lc, qm)

    lc = lc[~filter_mask]
    print(f"Additional filters mask {sum(filter_mask.unmasked)} "\
            f"row(s) leaving {len(lc)}.")

    # Optional binning of the light-curve
    if config.bin_time and config.bin_time > 0:
        bin_time = config.bin_time * u.s
        if int_time.to(u.s) >= bin_time:
            print(f"Light-curve already in bins >= {bin_time}")
        else:
            print(f"Binning light-curve to bins of {bin_time} duration.")
            lc = lc.bin(time_bin_size=bin_time, aggregate_func=np.nanmean)
            lc = lc[~np.isnan(lc.flux)] # Binning may have re-introduced NaNs
            print(f"After binning light-curve has {len(lc)} rows.")

    # Convert to relative mags with fitted polynomial detrending
    print("Detrending & 'zeroing' magnitudes by subtracting polynomial.")
    lightcurves.append_magnitude_columns(lc, "delta_mag", "delta_mag_err")
    lc["delta_mag"] -= lightcurves.fit_polynomial(lc.time,
                                                  lc["delta_mag"],
                                                  degree=2,
                                                  res_sigma_clip=DETREND_CLIP,
                                                  reset_const_coeff=False)

    # Set up the pipeline state for this sector going forward
    states.append(utility.new_sector_state(
        sys_name,
        lc.meta['SECTOR'],
        file_stem,
        lc["time", "flux", "flux_err", "delta_mag", "delta_mag_err"]))

    # Clear these down as they should not be used beyond this block
    del lc
del lcs


# ---------------------------------------------------------------------
# Find the ephemerides (& optionally plot the LCs)
# ---------------------------------------------------------------------
print("\nFinding orbital ephemerides")
for ss in states:
    (ss.primary_epoch, pe_ix) = lightcurves.find_primary_epoch(ss.lc)
    print(f"The {ss.name} sector {ss.sector} P.E. is JD {ss.primary_epoch.jd}")
    if config.period:
        ss.period = config.period * u.d
        print(f"An orbital period of {ss.period} was specified by the user.")
    else:
        ss.period = lightcurves.find_period(ss.lc, ss.primary_epoch)
        print(f"No period specified. Found {ss.period} from eclipse timings.")

    # Optionally plot the light-curve incl primary eclipse for diagnostics
    if config.plot_lc:
        ax = plot.plot_light_curve_on_axes(ss.lc,
                    title=f"Light-curve of {ss.name} sector {ss.sector}")
        pe_mag = ss.lc["delta_mag"][pe_ix]
        ax.scatter([ss.primary_epoch.value], [pe_mag.value], zorder=-10,
                   marker="x", s=64., lw=.5, c="k", label="primary eclipse")
        plt.savefig(output_dir / f"{ss.file_stem}_lightcurve.png", dpi=300)


# ---------------------------------------------------------------------
# Phase fold the LC & interpolate on the fold to use for our estimates
# ---------------------------------------------------------------------
print("\nPhase folding LCs in preparation for parameter estimation")
for ss in states:
    print(f"Folding {ss.name} sector {ss.sector} light-curve.")
    flc = lightcurves.phase_fold_lc(ss.lc, ss.primary_epoch, ss.period, 0.75)
    phase, ss.fold_mags = lightcurves.get_reduced_folded_lc(flc, ML_PHASE_BINS)

    # Optionally plot the folded LC overlaid with the interpolated one for diags
    if config.plot_fold:
        ax = plot.plot_folded_light_curve_on_axes(flc,
            title=f"Folded light-curve of {ss.name} sector {ss.sector}")
        ax.scatter(phase, ss.fold_mags, c="k", marker="+",
                   s=8, alpha=.5, linewidth=.5, zorder=10)
        plt.savefig(output_dir / f"{ss.file_stem}_folded.png", dpi=300)


# ---------------------------------------------------------------------
# Use the ML model to estimate system parameters
# ---------------------------------------------------------------------
print("\nEstimating system parameters")
e = estimator.Estimator()
df = e.predict(np.array([ss.fold_mags[:, np.newaxis] for ss in states]))

calc_names = utility.append_calculated_params(df)
col_names = [n for n in df.columns if not n.endswith("_sigma")]
for (ix, row), ss in zip(df.iterrows(), states):
    # Print out each individual set of predictions
    print(f"\nPredictions for light-curve of {ss.name} sector {ss.sector}")
    utility.echo_predictions(
        col_names,
        ["*" if n in calc_names else "" for n in col_names],
        [row[n] for n in col_names],
        [row[f"{n}_sigma"] for n in col_names],
        "Mean", "StdDev")

    # Now build the prediction dictionary
    ss.predictions = {name: row[name] for name in col_names}


# ---------------------------------------------------------------------
# Prepare and output dat/in files for JKTEBOP
# ---------------------------------------------------------------------
print(f"\nWriting JKTEBOP fitting files to {output_dir.resolve()}")
for ss in states:

    # Build polies before trimming so they're not affected by gaps from trimming
    poly_instructions = jktebop.build_polies_for_lc(ss.lc, config.polies)

    # Apply any user requests to trim the light-curves (for data reduction)
    if config.trim_masks and len(config.trim_masks) > 0:
        print("Applying requested trim masks to final light-curve")
        trim_mask = [False] * len(ss.lc)
        for tm in config.trim_masks:
            trim_mask |= lightcurves.mask_from_time_range(ss.lc, tm)
        ss.lc = ss.lc[~trim_mask]
        print(f"Trimming masks {sum(trim_mask)} row(s) leaving {len(ss.lc)}.")

        if config.plot_lc:
            ax = plot.plot_light_curve_on_axes(ss.lc,
                 title=f"Trimmed light-curve of {ss.name} sector {ss.sector}")
            plt.savefig(output_dir / f"{ss.file_stem}_trimmed.png", dpi=300)

    # The final setting of fitting params: defaults <- predictions <- overrides
    overrides = config.fitting_params or {}
    params = {
        "qphot": 0.,
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
        "period": ss.period.to(u.d).value,
        "primary_epoch": ss.primary_epoch.jd - 2.4e6,
        **ss.predictions,
        **overrides
    }

    # Generate JKTEBOP .dat and .in file for task3.
    jktebop.write_task3_in_file(output_dir / f"{ss.file_stem}.in",
                                poly_instructions, **params)
    jktebop.write_data_to_dat_file(ss.lc, output_dir / f"{ss.file_stem}.dat")
