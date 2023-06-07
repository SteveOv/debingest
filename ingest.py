from pathlib import Path
import argparse
import textwrap
from string import Template
import numpy as np
from scipy.interpolate.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import utils

# -------------------------------------------------------------------
# Command line will contain a list of systems to ingest and options
# -------------------------------------------------------------------
description = "Ingest pipeline for TESS dEB light-curves. \
    Searches for available light-curves for the requested identifier \
    in the MAST portal. Appropriate fits files are downloaded."

ap = argparse.ArgumentParser(description=description)
ap.add_argument("-f", "--flux", type=str, dest="flux_column",
                help="The flux column to use: sap_flux or pdcsap_flux. \
                    The default is sap_flux.")
ap.add_argument("-q", "--quality", type=str, dest="quality_bitmask",
                help="Quality bitmask to filter out low quality data (may be a \
                    numerical bitmask or text: none, default, hard, hardest). \
                    The default value is default.")
ap.add_argument("-pl", "--plot-lc", dest="plot_lc",
                action="store_true", required=False,
                help="Write a plot of each sector's light-curve to a png file")
ap.add_argument("-pf", "--plot-fold", dest="plot_fold",
                action="store_true", required=False,
                help="Write a plot of each sector folded data to a png file")
ap.add_argument("-s", "--systems", type=str, dest="systems", 
                nargs="+", required=True,
                help="Search identifier for each system to ingest.")
ap.set_defaults(systems=[], flux_column="sap_flux", quality_bitmask="default",
                plot_lc=False, plot_fold=False)
args = ap.parse_args()

detrend_sigma_clip = 0.5
model_phase_bins = 1024

# TODO: Arrange a proper location from which to pick up the model.
model = load_model("./cnn_model.h5")

for system in args.systems:
    sys_label = system.replace(" ", "_").lower()
    staging_dir = Path(f"./staging/{sys_label}")
    staging_dir.mkdir(parents=True, exist_ok=True)


    # ---------------------------------------------------------------------
    # Use MAST to DL any TESS SPOC timeseries/light-curves for the system
    # ---------------------------------------------------------------------
    # Could iterate over the results but I'd rather separate the 
    # data aquisition from parsing. Once we have the assets we can
    # comment these lines to avoid the overhead of search/dl on every run
    results = lk.search_lightcurve(system, mission="TESS", author="SPOC")
    results.download_all(download_dir=f"{staging_dir}", cache=True)

    # Now we will load the acquired data directly from the cache
    # so we're not directly dependent on the previous download
    fits_files = sorted(staging_dir.rglob("tess*_lc.fits"))
    lcs = LightCurveCollection([
        lk.read(f"{f}", 
                flux_column=args.flux_column, 
                quality_bitmask=args.quality_bitmask) 
        for f in fits_files
    ])


    # ---------------------------------------------------------------------
    # Process each of the system's light-curves in turn
    # ---------------------------------------------------------------------
    print(f"\nFound {len(lcs)} light-curve files for {system}.")
    for lc in lcs: 
        sector = f"{lc.meta['SECTOR']:0>4}"
        tic = f"{lc.meta['OBJECT']}"
        int_time = lc.meta["INT_TIME"] * u.min
        file_stem = f"{sys_label}_s{sector}"

        narrative = f"Processing {len(lc)} row(s) {args.flux_column} data "\
            f"(meeting the quality bitmask of {args.quality_bitmask}) "\
            f"for {system} sector {sector}. This covers the period of "\
            f"{lc.meta['DATE-OBS']} to {lc.meta['DATE-END']} "\
            f"with an integration time of {int_time}."
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
        # Optionally plot the light-curve for diagnostics
        # ---------------------------------------------------------------------
        if args.plot_lc:
            fig = plt.figure(figsize=(8, 4), constrained_layout=True)
            ax = fig.add_subplot(111)
            lc.scatter(column="delta_mag", ax=ax, 
                       ylabel="Relative Magnitude [mag]")
            ax.invert_yaxis()
            ax.get_legend().remove()
            ax.set(title=f"{system} ({tic}) sector {sector}")
            plt.savefig(staging_dir / (file_stem + ".png"), dpi=300)


        # ---------------------------------------------------------------------
        # Find the orbital period & primary_epoch and phase fold the light-curve
        # ---------------------------------------------------------------------
        # TODO - find primary epoch - time of best primary eclipse
        #        something like lc.time[lc.flux.argmax()]
        if sector == "0004":
            primary_epoch = Time(1415.482922, format="btjd", scale="tdb") 
        else:
            primary_epoch = Time(2152.142892, format="btjd", scale="tdb")

        # Find the orbital period
        permax = np.subtract(lc.time.max(), lc.time.min())
        pg = lc.normalize(unit="ppm").to_periodogram("ls", 
                                                     maximum_frequency=1,
                                                     minimum_frequency=1/permax,
                                                     oversample_factor=100)
        for period_factor in [2]:
            # Find the period it should be a harmonic of the max-power peak
            # TODO: try multiples & count peaks? (I know 2 works for CW Eri)
            period = np.multiply(pg.period_at_max_power, period_factor)

            # We need a phase folded LC to sample for the estimation model.  
            # Rotate the phase so that phase 0 is at the 25% position.
            fold_lc = lc.fold(period, 
                              epoch_time=primary_epoch,
                              normalize_phase=True,
                              wrap_phase=u.Quantity(0.75))


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
            fold_lc.scatter(column="delta_mag", ax=ax,
                            ylabel="Relative Magnitude [mag]")
            ax.errorbar(phases, mags, fmt="k.", markersize=0.75)
            ax.invert_yaxis()
            ax.get_legend().remove()
            ax.set(title=f"Folded light-curve of {system} sector {sector}")
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