from pathlib import Path
import argparse
import textwrap
import numpy as np
import astropy.units as u
from astropy.time import Time
import lightkurve as lk
from lightkurve import LightCurveCollection
import matplotlib.pyplot as plt

import utils

# -------------------------------------------------------------------
# Command line will contain a list of systems to ingest and options
# -------------------------------------------------------------------
description = "Ingest pipeline for TESS dEB light-curves. \
    Searches for available light-curves for the requested identifier \
    in the MAST portal. Appropriate fits files are downloaded."

ap = argparse.ArgumentParser(description=description)
ap.add_argument("-f", "--flux", type=str, dest="flux_column",
                help="The flux column to use sap_flux or pdcsap_flux")
ap.add_argument("-pl", "--plot-lc", dest="plot_lc",
                action="store_true", required=False,
                help="Write a plot of each sector light-curve to a png file")
ap.add_argument("-s", "--systems", type=str, dest="systems", 
                nargs="+", required=True,
                help="Search identifier for each system to ingest.")
ap.set_defaults(systems=[], flux_column="sap_flux", plot_lc=False)
args = ap.parse_args()

quality_threshold = 0
detrend_sigma_clip = 0.5


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
    lcs = LightCurveCollection(
        [lk.read(f"{f}", flux_column=args.flux_column) for f in fits_files])


    # ---------------------------------------------------------------------
    # Process each of the system's light-curves in turn
    # ---------------------------------------------------------------------
    print(f"\nFound {len(lcs)} light-curve files for {system}.")
    for lc in lcs: 
        sector = f"{lc.meta['SECTOR']:0>4}"
        tic = f"{lc.meta['OBJECT']}"
        int_time = lc.meta["INT_TIME"] * u.min

        narrative = f"Processing sector {sector} covering the period of "\
            f"{lc.meta['DATE-OBS']} to {lc.meta['DATE-END']} with an "\
            f"integration time of {int_time} over {len(lc)} row(s)."
        print()
        print("\n".join(textwrap.wrap(narrative, 70)))


        # ---------------------------------------------------------------------
        # Apply any data filters (NaN, quality, sigma clip, date range)
        # ---------------------------------------------------------------------
        # TODO: Replace this with setting of quality_bitmask in lk.read() 
        #       with value from an arg. Logic below is equiv to "hardest". 
        #       We'll still need to lk.remove_nans() as they break detrending.
        filter_mask = np.isnan(lc.flux)
        print(f"NaN clip finds {sum(filter_mask.unmasked)} data points.")

        if quality_threshold is not None:
            quality_mask = lc["quality"] > quality_threshold
            print(f"Quality clip finds {sum(quality_mask)} data points.")
            filter_mask |= quality_mask
   
        lc = lc[~filter_mask]
        print(f"Quality filters mask {sum(filter_mask.unmasked)} "\
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
            fig = plt.figure(figsize=(6, 4), constrained_layout=True)
            ax = fig.add_subplot(111)
            lc.scatter(column="delta_mag", ax=ax, 
                       ylabel="Relative Magnitude [mag]")
            ax.invert_yaxis()
            ax.get_legend().remove()
            ax.set(title=f"{system} ({tic}) sector {sector}")
            plt.savefig(staging_dir / f"{sys_label}_s{sector}.png", dpi=300)


        # ---------------------------------------------------------------------
        # Find the orbital period and suitable primary_epoch
        # ---------------------------------------------------------------------
        # TODO - maybe find peaks to find the largest (primary) eclipse
        #        and a periodogram to find the period
        pg = lc.to_periodogram("bls")
        print(pg)

        # ---------------------------------------------------------------------
        # Make a phase-fold and binned copy of the lightcurve
        # ---------------------------------------------------------------------
        # TODO - the ML model is set up for 1024 bins


        # ---------------------------------------------------------------------
        # Use the ML model to estimate system parameters
        # ---------------------------------------------------------------------
        # TODO 
        

        # ---------------------------------------------------------------------
        # Generate JKTEBOP .dat and .in file for task3 and invoke.
        # ---------------------------------------------------------------------
        # TODO - in file requires token substitution from the ML estimates
        #        and the period/primary_epoch values from periodogram