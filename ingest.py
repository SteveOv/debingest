from pathlib import Path
import argparse

import numpy as np
import astropy.units as u
from astropy.time import Time
import lightkurve as lk
from lightkurve import LightCurveCollection

import utils

# -------------------------------------------------------------------
# Command line will contain a filespec to match on
# -------------------------------------------------------------------
description = "Ingest pipeline for TESS dEB light-curves. \
    Searches for available light-curves for the requested identifier \
    in the MAST portal. Appropriate fits files are downloaded."

ap = argparse.ArgumentParser(description=description)
ap.add_argument("-s", "--systems", dest="systems", nargs="+", required=True,
                help="Identifier for each system to ingest.")
ap.set_defaults(systems=None)
args = ap.parse_args()

flux_column = "sap_flux"
quality_threshold = 0
detrend_sigma_clip = 0.5

systems = args.systems
for system in systems:
    cache_dir = Path(f"./__mast_cache/{system.replace(' ', '_').lower()}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Search for TESS lightcurve data and cache the fits files.  
    # Could iterate over the results but I'd rather separate the data aquisition from parsing.
    # Once we have the assets we can comment these lines out to avoid the overhead of search & dl on every run
    results = lk.search_lightcurve(system, mission="TESS", author="SPOC")
    results.download_all(download_dir=f"{cache_dir}", cache=True)

    # Now we will load the acquired data directly from the cache - so we're not directly dependent on the previous download
    lcs = LightCurveCollection([lk.read(f"{fits_file}", flux_column=flux_column) for fits_file in sorted(cache_dir.rglob("tess*_lc.fits"))])

    print(f"Opening {len(lcs)} lightcurve files for {system}.\n")
    for lc in lcs: 
        sector = f"{lc.meta['SECTOR']:0>4}"
        tic = f"{lc.meta['OBJECT']}"
        bjdref = lc.meta['BJDREFI'] + lc.meta["BJDREFF"]
        int_time = lc.meta["INT_TIME"] * u.min
        from_time = Time(lc.meta['TSTART']+bjdref, format="jd", scale=lc.meta["TIMESYS"].lower())
        to_time = Time(lc.meta['TSTOP']+bjdref, format="jd", scale=lc.meta["TIMESYS"].lower())

        print(f"Selected Metadata:")
        print(f"\tTIC = {tic}")
        print(f"\tSECTOR / CAMERA / CCD = {sector} / {lc.meta['CAMERA']} / {lc.meta['CCD']}")
        print(f"\tObserved from {from_time} to {to_time} {to_time.format}/{to_time.scale} with integration time of {int_time}")
 

        # ---------------------------------------------------------------------
        # Apply any Quality filters (NaN, quality, sigma clip, date range)
        # ---------------------------------------------------------------------
        print(f"Initial row count = {len(lc)}")
        filter_mask = np.isnan(lc.flux)
        print(f"NaN clip selects {sum(filter_mask.unmasked)} data points")

        if quality_threshold is not None:
            quality_mask = lc["quality"] > quality_threshold
            print(f"Quality clip selects {sum(quality_mask)} data points")
            filter_mask |= quality_mask
   
        lc = lc[~filter_mask]
        print(f"Initial 'quality' filters applied to {sum(filter_mask.unmasked)} rows leaving {len(lc)} row(s).")


        # ---------------------------------------------------------------------
        # Convert to relative mags and simple quadratic detrending
        # ---------------------------------------------------------------------
        print(f"Detrending & 'zeroing' magnitudes by subtracting polynomial.")
        lc["delta_mag"] = u.Quantity(-2.5 * np.log10(lc.flux.value) * u.mag)
        lc["delta_mag_err"] = u.Quantity(2.5 
                                        * 0.5 
                                        * np.abs(np.log10(lc.flux.value + lc.flux_err.value) 
                                                - np.log10(lc.flux.value - lc.flux_err.value)) * u.mag)
        lc["delta_mag"] -= utils.fit_polynomial(lc.time, 
                                                lc["delta_mag"], 
                                                degree=2, 
                                                res_sigma_clip=detrend_sigma_clip, 
                                                reset_const_coeff=False)


        # ---------------------------------------------------------------------
        # Optionally plot the light-curve for diagnostics
        # ---------------------------------------------------------------------
        # TODO


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