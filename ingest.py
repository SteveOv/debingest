#!/usr/bin/env python3
from pathlib import Path
import argparse
import textwrap
import json
import numpy as np
from scipy.interpolate.interpolate import interp1d
import astropy.units as u
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import functions

# -------------------------------------------------------------------
# Command line will contain a list of systems to ingest and options
# -------------------------------------------------------------------
description = "An ingest pipeline for TESS dEB light-curves. \
It searches for available light-curves for the requested identifier \
via the MAST portal and appropriate fits files are downloaded. \
For each fits file, the flux data is converted to relative magnitudes, \
a detrending polynomial is subtracted and some quality filters applied. \
Subsequently, the primary epoch and orbital period are calculated \
and a phase folded light-curve is passed to a Machine-learning model for \
system parameter estimation. \
Finally, the light-curve is prepared for fitting by JKTEBOP with the \
relative magnitudes being written to a text dat file and the primary \
epoch, period and estimated parameters used to create the in file which \
contains the JKTEBOP processing parameters and instructions."

ap = argparse.ArgumentParser(description=description)

# Must have 1 of these two. User must specify the target (& all other args) at
# the command line or specify a json file (& the args are read from file with
# overrides from the command line where given in both [specific code for this]).
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--target", type=str, dest="target",
                help="Search identifier for the system to ingest.")
group.add_argument("-f", "--file", type=Path, dest="file",
                help="Search identifier for the system to ingest.")

# These are not part of the group above
ap.add_argument("-s", "--sector", type=int, 
                dest="sectors", action="append", 
                help="A sector to search for, or omit to search on all sectors.\
                    Multiple sector arguments supported.")
#ap.add_argument("-m", "--mission", type=str, dest="mission", default="TESS",
#                help="The source mission: currently only TESS supported")
#ap.add_argument("-a", "--author", type=str, dest="author", default="SPOC",
#                help="The author of the data: currently only SPOC supported")
ap.add_argument("-fl", "--flux", type=str, 
                dest="flux_column", default="sap_flux",
                help="The flux column to use: sap_flux or pdcsap_flux. \
                    The default is sap_flux.")
ap.add_argument("-q", "--quality", type=str, dest="quality_bitmask",
                help="Quality bitmask to filter out low quality data (may be a \
                    numerical bitmask or text: none, default, hard, hardest). \
                    The default value is default.")
ap.add_argument("-e", "--exptime", type=str, dest="exptime",
                help="Exposure time/cadence: set to long, short, fast or an \
                    exact time in seconds. If omitted Will retrieve all.")
ap.add_argument("-p", "--period", type=np.double, dest="period",
                help="The period, in days, of the system. If not specified the \
                    period will be calculated on light-curve eclipse spacing.")
ap.add_argument("-c", "--clip", type=np.double, 
                nargs=2, dest="clips", action="append",
                help="A clip time range (from, to) to remove from light-curves\
                    prior to processing. Multiple clip arguments supported.")
ap.add_argument("-pl", "--plot-lc", dest="plot_lc",
                action="store_true", required=False,
                help="Write a plot of each sector's light-curve to a png file")
ap.add_argument("-pf", "--plot-fold", dest="plot_fold",
                action="store_true", required=False,
                help="Write a plot of each sector folded data to a png file")
ap.set_defaults(target=None, file=None, 
                sectors=[], mission="TESS", author="SPOC", exptime=None,
                flux_column="sap_flux", quality_bitmask="default", 
                period=None, clips=[], plot_lc=False, plot_fold=False)
args = ap.parse_args()


# ---------------------------------------------------------------------
# Handle setting up the pipeline processing configuration
# ---------------------------------------------------------------------
if args.target is not None:
    print(f"Configuring pipeline based on command line arguments")
else:
    print(f"Configuring pipeline from {args.file} & any command line overrides")
    # Read the JSON file and use it as the basis of our pipeline arguments but
    # with overrides from the command line arguments to apply missing default
    # values or otherwise override where non-default values have been given.
    with open(args.file, "r") as f:
        file_args = json.load(f)
        overrides = {k: v for k, v in vars(args).items() 
                     if k not in file_args or ap.get_default(k) != v}
        args = argparse.Namespace(**{**file_args, **overrides})

detrend_clip = 0.5
model_phase_bins = 1024

sys_name = args.target
sys_label = "".join(c for c in sys_name if c not in r':*?"\/<>|').\
    lower(). \
    replace(' ', '_')
staging_dir = Path(f"./staging/{sys_label}")
staging_dir.mkdir(parents=True, exist_ok=True)


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
    model = load_model("./cnn_model.h5")


# ---------------------------------------------------------------------
# Process each of the system's light-curves in turn
# ---------------------------------------------------------------------
for lc in lcs:  
    sector = f"{lc.meta['SECTOR']:0>4}"
    tic = f"{lc.meta['OBJECT']}"
    int_time = (lc.meta["INT_TIME"] + lc.meta["READTIME"]) * u.min
    file_stem = f"{sys_label}_s{sector}"

    narrative = f"Processing {len(lc)} row(s) {args.flux_column} data "\
        f"(meeting the quality bitmask of {args.quality_bitmask}) "\
        f"for {sys_name} sector {sector} (camera {lc.meta['CAMERA']} / "\
        f"CCD {lc.meta['CCD']}). This covers the period of "\
        f"{lc.meta['DATE-OBS']} to {lc.meta['DATE-END']} "\
        f"with an integration time of {int_time.to(u.s)}."
    print()
    print("\n".join(textwrap.wrap(narrative, 70)))


    # ---------------------------------------------------------------------
    # Apply any additional data filters (NaN/negative fluxes, date range)
    # ---------------------------------------------------------------------
    # We'll still need to filter out NaN & negative fluxes as they may still be 
    # present (only hardest seems to exclude them) and they'll break detrending.
    filter_mask = np.isnan(lc.flux)
    filter_mask |= lc.flux < 0
    print(f"NaN/negative flux clip masks {sum(filter_mask.unmasked)} row(s).")

    if args.clips and len(args.clips):
        for clip in args.clips:
            filter_mask |= functions.clip_mask_from_time_range(lc, clip)

    lc = lc[~filter_mask]
    print(f"Additional filters mask {sum(filter_mask.unmasked)} "\
            f"row(s) leaving {len(lc)}.")


    # ---------------------------------------------------------------------
    # Convert to relative mags with fitted polynomial detrending
    # ---------------------------------------------------------------------
    print(f"Detrending & 'zeroing' magnitudes by subtracting polynomial.")
    functions.append_magnitude_columns(lc, "delta_mag", "delta_mag_err")
    lc["delta_mag"] -= functions.fit_polynomial(lc.time, 
                                                lc["delta_mag"], 
                                                degree=2, 
                                                res_sigma_clip=detrend_clip, 
                                                reset_const_coeff=False)


    # ---------------------------------------------------------------------
    # Find the primary epoch and orbital period
    # ---------------------------------------------------------------------
    (primary_epoch, primary_epoch_ix) = functions.find_primary_epoch(lc)
    print(f"The primary epoch for sector {sector} is at JD {primary_epoch.jd}")
    if args.period is None:
        period = functions.find_period(lc, primary_epoch)
        print(f"No period specified. Found {period} based on eclipse timings.")
    else:
        period = args.period * u.d
        print(f"An orbital period of {period} was specified by the user.")


    # ---------------------------------------------------------------------
    # Optionally plot the light-curve incl primary eclipse for diagnostics
    # ---------------------------------------------------------------------
    if args.plot_lc:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        lc[[primary_epoch_ix]].scatter(column="delta_mag", ax=ax, 
                                       marker="x", s=64., linewidth=.5,
                                       color="k", label="primary eclipse")
        lc.scatter(column="delta_mag", ax=ax, s=2., label=None)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in",
                       bottom=True, top=True, left=True, right=True)
        ax.get_legend().remove()
        ax.set(title=f"{sys_name} sector {sector} light-curve",
               ylabel="Relative magnitude [mag]")
        plt.savefig(staging_dir / (file_stem + "_lightcurve.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Phase fold the LC & interpolate on the fold to use for our estimates
    # ---------------------------------------------------------------------
    print(f"Phase folding the LC to get a sample curve for param estimation.")
    fold_lc = functions.phase_fold_lc(lc, primary_epoch, period, 0.75)

    # Now we sample/interpolate the folded LC at 1024 points.
    # So far, linear interpolation is producing lower variance
    min_phase = fold_lc.phase.min()
    interp = interp1d(fold_lc.phase, fold_lc["delta_mag"], kind="linear")
    phases = np.linspace(min_phase, min_phase+1., model_phase_bins+1)[:-1]
    mags = interp(phases)


    # ---------------------------------------------------------------------
    # Optionally plot the folded LC overlaid with the interpolated one for diags
    # ---------------------------------------------------------------------
    if args.plot_fold:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        fold_lc.scatter(column="delta_mag", ax=ax, s=2., alpha=0.25, label=None)
        ax.scatter(phases, mags, color="k", marker="+", s=8., linewidth=0.5)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", 
                       bottom=True, top=True, left=True, right=True)
        ax.set(title=f"Folded light-curve of {sys_name} sector {sector}",
               ylabel="Relative magnitude [mag]")
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

    # Calculate the orbital inclination from the impact parameter.
    # In training the mae of bA is usually lower, so we'll use that.
    # inc = arccos( bA * rA * [1+esinw]/[1-e^2] )
    rA = np.divide(rA_plus_rB, np.add(1, k))
    omega = np.arctan(np.divide(ecosw, esinw))
    e = np.divide(ecosw, np.cos(omega))
    cosi = np.multiply(np.multiply(rA, bA), 
                       np.divide(np.add(1, esinw), 
                                 np.subtract(1, np.power(e, 2))))
    inc = np.rad2deg(np.arccos(cosi))


    # ---------------------------------------------------------------------
    # Generate JKTEBOP .dat and .in file for task3.
    # ---------------------------------------------------------------------
    params = {
        "rA_plus_rB": rA_plus_rB,
        "k": k,
        "inc": inc,
        "qphot": 0.,
        "esinw": esinw,
        "ecosw": ecosw,
        "J": J,
        "L3": 0.,
        # TODO: Do we train LD params?
        "LD_A": "pow2",
        "LD_B": "pow2",
        "LD_A1": 0.65,
        "LD_B1": 0.65,
        "LD_A2": 0.47,
        "LD_B2": 0.47,
        "reflA": 0.,
        "reflB": 0.,
        "period": period.to(u.d).value,
        "primary_epoch": primary_epoch.jd - 2.4e6,
    }

    append_text = []
    if args.polies:
        for poly in args.polies:
            poly_from = functions.to_time(np.min(poly["time_range"]), lc)
            poly_to = functions.to_time(np.max(poly["time_range"]), lc)
            if lc.time.min() < poly_from < lc.time.max() \
                    or lc.time.min() < poly_to < lc.time.max():
                poly["time_range"] = (poly_from, poly_to)
                append_text += ["\n" + functions.build_poly_instr(**poly)]

    functions.write_task3_in_file(staging_dir / (file_stem + ".in"), 
                                  append_text, **params)
    functions.write_data_to_dat_file(lc, staging_dir / (file_stem + ".dat"))
    print(f"JKTEBOP dat & in files were written to {staging_dir.resolve()}")