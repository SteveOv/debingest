#!/usr/bin/env python3
from pathlib import Path
import argparse
import textwrap
import numpy as np
from scipy.interpolate.interpolate import interp1d
from scipy.signal import find_peaks, find_peaks_cwt
import astropy.units as u
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import utils

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
ap.add_argument("-t", "--target", type=str, dest="target", required=True,
                help="Search identifier for the system to ingest.")
ap.add_argument("-s", "--sectors", type=int, nargs="*", dest="sectors",
                help="Sectors to search for or omit to search on all sectors.")
#ap.add_argument("-m", "--mission", type=str, dest="mission", default="TESS",
#                help="The source mission: currently only TESS supported")
#ap.add_argument("-a", "--author", type=str, dest="author", default="SPOC",
#                help="The author of the data: currently only SPOC supported")
ap.add_argument("-f", "--flux", type=str, dest="flux_column",default="sap_flux",
                help="The flux column to use: sap_flux or pdcsap_flux. \
                    The default is sap_flux.")
ap.add_argument("-q", "--quality", type=str, dest="quality_bitmask",
                help="Quality bitmask to filter out low quality data (may be a \
                    numerical bitmask or text: none, default, hard, hardest). \
                    The default value is default.")
ap.add_argument("-p", "--period", type=np.double, dest="period",
                help="The period, in days, of the system. If not specified the \
                    period will be calculated on light-curve eclipse spacing.")
ap.add_argument("-pl", "--plot-lc", dest="plot_lc",
                action="store_true", required=False,
                help="Write a plot of each sector's light-curve to a png file")
ap.add_argument("-pf", "--plot-fold", dest="plot_fold",
                action="store_true", required=False,
                help="Write a plot of each sector folded data to a png file")
ap.set_defaults(target=None, sectors=[], mission="TESS", author="SPOC", 
                flux_column="sap_flux", quality_bitmask="default", 
                period=None, plot_lc=False, plot_fold=False)
args = ap.parse_args()

detrend_sigma_clip = 0.5
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
                               mission=args.mission, author=args.author)
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

print(f"\nFound {len(lcs)} light-curves for {sys_name} sectors {args.sectors}.")
if len(lcs):
    # TODO: Arrange a proper location from which to pick up the model.
    model = load_model("./cnn_model.h5")

# ---------------------------------------------------------------------
# Process each of the system's light-curves in turn
# ---------------------------------------------------------------------
for lc in lcs: 
    sector = f"{lc.meta['SECTOR']:0>4}"
    tic = f"{lc.meta['OBJECT']}"
    frame_time = lc.meta["FRAMETIM"] * u.min
    file_stem = f"{sys_label}_s{sector}"

    narrative = f"Processing {len(lc)} row(s) {args.flux_column} data "\
        f"(meeting the quality bitmask of {args.quality_bitmask}) "\
        f"for {sys_name} sector {sector}. This covers the period of "\
        f"{lc.meta['DATE-OBS']} to {lc.meta['DATE-END']} "\
        f"in bins of {frame_time}."
    print()
    print("\n".join(textwrap.wrap(narrative, 70)))


    # ---------------------------------------------------------------------
    # Apply any additional data filters (NaN, sigma clip, date range)
    # ---------------------------------------------------------------------
    # We'll still need to filter out NaNs as they may still be present
    # (only hardest seems to exclude them) and they'll break detrending.
    filter_mask = np.isnan(lc.flux)
    print(f"NaN clip finds {sum(filter_mask.unmasked)} data points.")

    lc = lc[~filter_mask]
    print(f"Additional filters mask {sum(filter_mask.unmasked)} "\
            f"row(s) leaving {len(lc)}.")


    # ---------------------------------------------------------------------
    # Convert to relative mags with fitted polynomial detrending
    # ---------------------------------------------------------------------
    print(f"Detrending & 'zeroing' magnitudes by subtracting polynomial.")
    lc["delta_mag"] = u.Quantity(-2.5 * np.log10(lc.flux.value) * u.mag)
    lc["delta_mag_err"] = u.Quantity(
        2.5 
        * 0.5
        * np.abs(
            np.subtract(
                np.log10(np.add(lc.flux.value, lc.flux_err.value)),
                np.log10(np.subtract(lc.flux.value, lc.flux_err.value))
            )
        )
        * u.mag)

    # Quadratic detrending and y-shifting relative to zero 
    lc["delta_mag"] -= utils.fit_polynomial(
        lc.time, 
        lc["delta_mag"], 
        degree=2, 
        res_sigma_clip=detrend_sigma_clip, 
        reset_const_coeff=False)


    # ---------------------------------------------------------------------
    # Find the orbital period & primary_epoch and phase fold the light-curve
    # ---------------------------------------------------------------------
    std_mag = np.std(lc["delta_mag"].data)
    (peak_ixs, peak_props) = find_peaks(lc["delta_mag"], 
                                        prominence=(std_mag, ), 
                                        width=5)
    strongest_peak_ix = peak_ixs[peak_props["prominences"].argmax()]
    primary_epoch = lc.time[strongest_peak_ix]
    print(f"The primary epoch (strongest eclipse) is at JD {primary_epoch.jd}")

    if args.period is None:
        # If no period is specified, derive it from a periodogram restricted 
        # to a frequency range based on the known peak/eclipse spacing. 
        # LK docs recommend normalize("ppm") and oversample_factor=100.
        print(f"No period specified. Choosing one based on eclipse timings.")
        eclipse_diffs = np.diff(lc.time[peak_ixs])
        max_fr = np.reciprocal(np.min(eclipse_diffs))
        min_fr = np.reciprocal(np.multiply(np.max(eclipse_diffs), 2))
        pg = lc.normalize(unit="ppm").to_periodogram("ls", 
                                                     maximum_frequency=max_fr,
                                                     minimum_frequency=min_fr,
                                                     oversample_factor=100)

        # The period should be a harmonic of the periodogram's max-power peak
        # Set this candidates up with the most likely last so we fall through.
        periods = [np.multiply(pg.period_at_max_power, f) for f in [1., 2.]]
    else:
        periods = [args.period * u.d]        
        print(f"An orbital period of {periods[0]} has been specified.")

    for period in periods:
        # Test periods by looking for 2 peaks on a folded LC (having rotated it
        # so phase 0 is positioned at 0.25). We'll need the folded LC later.
        # If the test fails we'll fall through on the last period option.
        fold_lc = lc.fold(period, epoch_time=primary_epoch, 
                          normalize_phase=True, wrap_phase=u.Quantity(0.75))
        pc = len(find_peaks(fold_lc["delta_mag"], prominence=(std_mag, ), width=5)[0])
        print(f"\tTesting period {period}: found {pc} distinct peak(s)")
        if pc == 2:
            break


    # ---------------------------------------------------------------------
    # Optionally plot the light-curve incl primary eclipse for diagnostics
    # ---------------------------------------------------------------------
    if args.plot_lc:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        lc[[strongest_peak_ix]].scatter(column="delta_mag", ax=ax, 
                                        marker="x", s=64., linewidth=.5,
                                        color="k", label="primary eclipse")
        lc.scatter(column="delta_mag", ax=ax, s=2., label=None)
        ax.invert_yaxis()
        ax.get_legend().remove()
        ax.set(title=f"{sys_name} sector {sector} light-curve",
                ylabel="Relative magnitude [mag]")
        plt.savefig(staging_dir / (file_stem + "_lightcurve.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Interpolate the folded data to base our estimated params on
    # ---------------------------------------------------------------------
    # Now we sample/interpolate the folded LC at 1024 points.
    # So far, linear interpolation is producing lower variance
    min_phase = fold_lc.phase.min()
    interp = interp1d(fold_lc.phase, fold_lc["delta_mag"], kind="linear")
    phases = np.linspace(min_phase, min_phase+1., model_phase_bins+1)[:-1]
    mags = interp(phases)

    if args.plot_fold:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        fold_lc.scatter(column="delta_mag", ax=ax, s=2., 
                        alpha=0.25, label=None)
        ax.scatter(phases, mags, color="k", marker="+", s=8., linewidth=0.5)
        ax.invert_yaxis()
        ax.set(title=f"Folded light-curve of {sys_name} sector {sector}",
                ylabel="Relative magnitude [mag]")
        plt.savefig(staging_dir / (file_stem + "_folded.png"), dpi=300)


    # ---------------------------------------------------------------------
    # Use the ML model to estimate system parameters
    # ---------------------------------------------------------------------
    # Now we can invoke the ML model to interpret the folded data & estimate
    # the parameters for JKTEBOP. Need the mag data in shape[1, 1024, 1]
    predictions = model.predict(np.array([np.transpose([mags])]), verbose=0)

    # Predictions for a single model will have shape[1, 7]
    (rA_plus_rB, k, bA, bB, ecosw, esinw, J) = predictions[0, :]

    # Need e, omega and rA to calculate orbital inc
    omega = np.arctan(np.divide(ecosw, esinw))
    e = np.divide(ecosw, np.cos(omega))
    rA = np.divide(rA_plus_rB, np.add(1, k))

    # Calculate the orbital inclination from the impact parameter.
    # In training the mae of bA is usually lower, so we'll use that.
    # inc = arccos( bA * rA * [1+esinw]/[1-e^2] )
    cosi = np.multiply(np.multiply(rA, bA), 
                        np.divide(np.add(1, esinw), 
                                    np.subtract(1, np.power(e, 2))))
    inc = np.rad2deg(np.arccos(cosi))


    # ---------------------------------------------------------------------
    # Generate JKTEBOP .dat and .in file for task3 and invoke.
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

    utils.write_task3_in_file(staging_dir / (file_stem + ".in"), **params)
    utils.write_data_to_dat_file(lc, staging_dir / (file_stem + ".dat"))
    print(f"JKTEBOP dat & in files were written to {staging_dir.resolve()}")