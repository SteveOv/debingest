# debingest

## Detached Eclipsing Binary Light-curve ingest pipeline

This code base was developed in VSCode (on Ubuntu) within and an **Anaconda** 
environment named **debingest** configured to run _Python 3.8_. The 
configuration of the evironment is in `environment.yml`.

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

$ python3 ingest.py -t 'CW Eri' -s 4 -s 31 -fl sap_flux -q hard -p 2.72837 -c 58420.0 58423.0 -pl -pf 
```
where
- `-t`/`--target`: required MAST identifier for the target system to process
- `-s`/`--sector`: an optional sector to find - finds all if omitted
- `-fl`/`--flux`: the flux data column to use: **sap_flux** or pdcsap_flux
- `-q`/`--quality`: the quality filter set: none, **default**, hard or hardest
- `-p`/`--period`: the optional orbital period to use - calculated if omitted
- `-c`/`--clip`: optional time range to clip from any LCs - must have 2 values
- `-pl`/`--plot-lc`: instructs the pipeline to plot each lightcurve to a png
- `-pf`/`--plot-fold`: instructs the pipeline to plot each folded LC to a png

The `-s` or `--sector` argument may be given multiple times, once for each 
sector required.  If there are no `-s` arguments then all available sectors
are found for processing. 

The `-c` or `--clip` argument may be specified multiple times if you require 
clipping of multiple time ranges or sectors. You cannot specify which sectors 
a clip applies to but, as the sectors will have been observed at different 
times, only those sectors that overlap a given clip will be affected.

> If you first run `chmod +x ingest.py` (or equivalent) in the terminal 
> you remove the need to specify python3 whenever you run ingest.py.

**Alternatively** the pipeline parameters can be set up in a json file and 
passed to ingest.py with the following:

```sh
$ python3 ingest.py -f examples/cw_eri.json
```
where
- `-f`/`--file`: is the file to load the pipeline parameters from.

With the following being the contents of cw_eri.json equivalent to the above
command line arguments (with the same default values and behaviour). 

```json
{
  "target": "CW Eri",
  "sectors": [
    4,
    31
  ],
  "flux_column": "sap_flux",
  "quality_bitmask": "hard",
  "clips": [
    [58420.0, 58423.0]
  ],
  "period": 2.72837,
  "plot_lc": true,
  "plot_fold": true
}
```

The file/json approach has the benefit that the parameters are persistent 
making it less furstrating and safer to set up complex or large parameter sets
than on the command line. There is the added benefit that the parameter file 
can be stored in a source control or document repository alongside other assets.

There is explicit support for overriding parameter file values with the command
line. The example below shows the quality bitmask in the file above being
overriden with the value hardest.

```sh
$ python3 ingest.py -f examples/cw_eri.json -q hardest
```

The json file adds support for configuration that would be difficult with 
the command line arguments. Currently this is limited to setting up poly
fitting instructions in the JKTEBOP in files produced.  An example of the
additional json file settings is shown below.

```json
{

  "polies": [
    { "term": "sf", "degree": 1, "time_range": [58410.00, 58420.00] },
    { "term": "sf", "degree": 1, "time_range": [58424.00, 58437.00] },
    { "term": "sf", "degree": 1, "time_range": [59144.00, 59157.70] },
    { "term": "sf", "degree": 1, "time_range": [59158.00, 59170.50] }
  ]

}
```

> The time values for clip or poly ranges will be interpreted as BTJD (<40,000), 
> reduced JD (<2.4e6) or JD (>= 2.4e6).

## Processing
The pipeline will carry out the following tasks for the specified system:
- the MAST portal is queried on the target/sectors for TESS/SPOC light-curves
- any located fits files are downloaded
- for each fits/sector:
  - the fits file is read and filtered based on the `--quality` argument
  - the data is filtered removing rows where the `--flux` column is NaN
  - the `--clip` ranges are applied - any data within these are removed
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
    - this includes the appropriate poly instructions for the data's time range