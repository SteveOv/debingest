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

$ python3 ingest.py --flux sap_flux --quality hardest --plot-lc --plot-fold --systems 'CW Eri' 'V505 Per'
```
where
- `--flux` - the flux data column to use: **sap_flux** or pdcsap_flux
- `--quality` - the quality filter to apply: none, **default**, hard or hardest
- `--plot-lc` - instructs the pipeline to plot each lightcurve to a png
- `--plot-fold` - instructs the pipeline to plot each folded LC to a png
- `--systems` - a list of the MAST identifiers for the system(s) to process

> If you first run `chmod +x ingest.py` (or equivalent) in the terminal 
> you remove the need to specify python3 whenever you run ingest.py.

## Processing
The pipeline will carry out the following tasks for each specified system
in turn:
- the MAST portal is queried on the system identifier for TESS/SPOC light-curves
- any located fits files are downloaded
- for each fits/sector:
  - the fits file is read and filtered based on the `--quality` argument
  - the data is further filtered removing rows where the `--flux` column is NaN
  - magnitudes are calculated from the `--flux` and corresponding error columns
    - a low order polynomial is subtracted to detrend the data
    - this also y-shifts the data so that the magnitudes are relative to zero
  - the primary epoch (most prominent eclipse) and orbital period are found
  - if `--plot-lc` the light-curve & primary epoch is plotted to a png
  - the magnitude data is phase-folded on the primary eclipse/period
    - a 1024 bin interpolated phase-folded light-curve is derived from this
    - this is passed to a Machine-Learning model for system parameter estimation
    - if `--plot-fold` both folded light-curves are plotted to a png
  - the filtered light-curve magnitude data is written to a JKTEBOP dat file
  - the estimated system parameters are used to write a JKTENOP _task 3_ in file