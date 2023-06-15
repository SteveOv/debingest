# debingest

## Detached Eclipsing Binary Light-curve ingest pipeline

This code base was developed in VSCode (on Ubuntu 22.04) within and an 
**Anaconda** environment named **debingest**. This environment is configured 
to support _Python 3.8_ and the libraries on which the code is dependent.

To set up the **debingest** environment, having first cloned the GitHub repo, 
open a Terminal, navigate to this directory and run the following;
```sh
$ conda env create -f environment.yml
```

Whenever you run the ingest pipeline, it will require this environment be
active. You can activate it with the following command;
```sh
$ conda activate debingest
```

## Command line Use
The entry point for this pipeline is `ingest.py`.  This should be run in the
context of the `debingest` environment, with example usage shown below;

```sh
$ python3 ingest.py -t 'CW Eri' -s 4 -s 31 -fl sap_flux -q hard -p 2.72837 -c 58420.0 58423.0 -pl -pf 
```
where
- `-t`/`--target`: required MAST identifier for the target system to process
- `-s`/`--sector`: an optional sector to find - finds all if omitted
- `-fl`/`--flux`: the flux data column to use: **sap_flux** or pdcsap_flux
- `-e`/`--exptime`: optionally filter on exposure time: long, short or fast
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

## Ingest JSON file use
**Alternatively** a target's pipeline parameters can be set up in a json file 
and passed to ingest.py with the `-f` or `--file` argument, as follows:

```sh
$ python3 ingest.py -f examples/cw_eri.json
```
where
- `-f`/`--file`: is the file to load the pipeline parameters from.

The following is the content of cw_eri.json equivalent to the above command 
line arguments (with the same default values and behaviour). 

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

The use of a target ingest json file will allow more complex config to be 
expressed and for it to be persisted and stored as an asset.

There is support for overriding parameter file values with the previosly 
discussed command line arguments. The example below shows the quality bitmask 
in the cw_eri.json file being overriden with the value hardest.

```sh
$ python3 ingest.py -f examples/cw_eri.json -q hardest
```

The json file adds support for configuration that would be difficult to express
with command line arguments. The initial use was for configuring poly fitting 
instructions within the JKTEBOP .in files produced. An example is shown below:

```json
{

  "polies": [
    { "term": "sf", "degree": 1, "time_range": [58410.00, 58420.00] },
    { "term": "sf", "degree": 1, "time_range": [58424.00, 58437.00] },
    { "term": "sf", "degree": 1, "gap_threshold": 0.5 }
  ]

}
```

There are now two types of poly config:
- a _manual poly_ will have a `time_range` parameter
  - the range over which the poly is applied is specified in the `time_range`
  - will only be applied to a light-curve where the time ranges overlap
  - will generate a single poly instruction
- an _auto-poly_ will have a `gap_threshold` parameter 
  - can apply to any light-curve
  - ranges are derived by splitting a light-curve on time gaps > threshold
  - will generate one or more poly instructions
- the `term` and `degree` parameter are common to both

Polies are processed in order, with the supported config being zero or more 
_manual polies_ listed before a final, optional _auto-poly_ (as 
shown). For each of the target's light-curves, the manual polies will be tested
and applied where they overlap with the light-curve on the time axis. The 
_auto-poly_ will be triggered where no _manual polies_ were applied. This set 
up allows for selective overriding of a default _auto-poly_ with _manual polies_
applied to those light-curves where the "gap_threshold" of the default 
_auto-poly_ is problematic. The different types of poly are exclusive per 
light-curve, once one has been triggered subsequent polies of a different 
type will be ignored.

> The time values for clip or poly date ranges will be interpreted as 
> BTJD (<40,000), reduced JD (<2.4e6) or JD (>= 2.4e6).

The json file also allows you to explicitly set the fitting parameters in the
JKTEBOP .in files written. You do this by populating a "fitting_params" 
dictionary in the json file with keys matching the template tokens to be set. 
The example below sets the photometric mass ratio, 3rd light and limb darkening 
values with any other token/parameters retaining the values estimated during 
ingest:

```json
{

  "fitting_params": {
    "qphot": 0.836,
    "L3": 0.080,
    "LD_A": "pow2",
    "LD_B": "pow2",
    "LD_A1": 0.6437,
    "LD_B1": 0.6445,       
    "LD_A1_fit": 1,
    "LD_B1_fit": 1,
    "LD_A2": 0.4676,
    "LD_B2": 0.4967,
    "LD_A2_fit": 0,
    "LD_B2_fit": 0
  },

}
```

The ./examples/cw_eri.json file is a good example of the range of configuration
possible using a json file.  Also see the ./library/task3.in.template file to
see what tokens are set during ingest.

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
    - any overrides from the fitting_params are applied
    - this includes the appropriate poly instructions for the data's time range