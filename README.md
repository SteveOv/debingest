# debingest

## Detached Eclipsing Binary Light-curve ingest pipeline

This code base was developed in VSCode (on Ubuntu) within and an **Anaconda** 
environment named **debingest** configured to run _Python 3.8_.  
The configuration of the evironment is in `environment.yml`.

Set up the environment, having first cloned the GitHub repo, by kicking off a 
new Terminal and running the following (tested on Ubuntu 22.04);
```sh
$ conda env create -f environment.yml
```

## Usage
The entry point for this pipeline is `ingest.py`.  This should be run in the
context of the `debingest` environment, with example usage shown below;

```sh
$ conda activate debingest

$ python3 ingest.py --flux sap_flux --quality hard --plot-lc --plot-fold --system 'CW Eri' --period 2.72837
```
where
- `--flux`/`-f`: the flux data column to use: **sap_flux** or pdcsap_flux
- `--quality`/`-q`: the quality filter set: none, **default**, hard or hardest
- `--plot-lc`/`-pl`: instructs the pipeline to plot each lightcurve to a png
- `--plot-fold`/`-pf`: instructs the pipeline to plot each folded LC to a png
- `--system`/`-s`: the MAST identifiers for the system to process
- `--period`/`-p`: the optional orbital period to use - calculated if omitted

> If you first run `chmod +x ingest.py` (or equivalent) in the terminal 
> you remove the need to specify python3 whenever you run ingest.py.

## Processing
The pipeline will carry out the following tasks for the specified system:
- the MAST portal is queried on the system identifier for TESS/SPOC light-curves
- any located fits files are downloaded
- for each fits/sector:
  - the fits file is read and filtered based on the `--quality` argument
  - the data is further filtered removing rows where the `--flux` column is NaN
  - magnitudes are calculated from the `--flux` and corresponding error columns
    - a low order polynomial is subtracted to detrend the data
    - this also y-shifts the data so that the magnitudes are relative to zero
  - the primary epoch (most prominent eclipse) is found
  - if no `--period` specified, an estimate period will be found from eclipses
  - if `--plot-lc` the light-curve & primary epoch is plotted to a png
  - the magnitude data is phase-folded on the primary eclipse/period
    - a 1024 bin interpolated phase-folded light-curve is derived from this
    - this is passed to a Machine-Learning model for system parameter estimation
    - if `--plot-fold` both folded light-curves are plotted to a png
  - the filtered light-curve magnitude data is written to a JKTEBOP dat file
  - the estimated system parameters are used to write a JKTEBOP _task 3_ in file