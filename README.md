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
$ python3 ingest.py -t 'CW Eri' -s 4 -fl sap_flux -q hard -b 240 -p 2.73 -qm 58420.0 58423.0 -pl -pf 
```
where
- `-t`/`--target`: required MAST identifier for the target system to process
- `-sys`/`--sys-name`: optional system name for labels (defaults to --target)
- `-pr`/`--prefix`: optional prefix for output files (defaults to --sys-name)
- `-o`/`--output-dir`: optional output location (defaults to staging/--prefix)
- `-s`/`--sector`: an optional sector to find - finds all if omitted
- `-fl`/`--flux`: the flux data column to use: **sap_flux** or pdcsap_flux
- `-e`/`--exptime`: optionally filter on exposure time: long, short or fast
- `-q`/`--quality`: the quality filter set: none, **default**, hard or hardes
- `-qm`/`--quality-mask`: optional time range to mask from any LCs
  - must have two values - a start and end time (i.e.: -qm 51000.0 52020.0)
  - these are applied after download and before detrending & conversion to mags
- `-b`/`--bin-time`: optionally bin the data to bins of this duration (seconds)
- `-p`/`--period`: the optional orbital period to use - calculated if omittedes
- `-pl`/`--plot-lc`: instructs the pipeline to plot each lightcurve to a png
- `-pf`/`--plot-fold`: instructs the pipeline to plot each folded LC to a png
- `-tm`/`--trim-mask`: optional time range to trim from the final LCs
  - must have two values - a start and end time (i.e.: -tm 51005.0 51007.5)
  - trim masks are applied last thing before writting JKTEBOP in & dat files

The `-sys` or `--sys-name` argument will be set to the same value as the target
if no value given. It is used in plot labels and titles and for the file prefix
& output directory (unless overriden with values given for `--prefix` or 
`--output-dir`).

The `-s` or `--sector` argument may be given multiple times, once for each 
sector required.  If there are no `-s` arguments then all available sectors
are found for processing. 

The `-qm`/`--quality-mask` and `-tm`/`--trim-mask` arguments may be specified 
multiple times if masking of multiple time ranges or sectors is required. You 
cannot specify which sector a mask applies to but, as their observations will 
cover different times, only those that overlap a given mask will be affected.

> If you first run `chmod +x ingest.py` (or equivalent) in the terminal 
> you remove the need to specify python3 whenever you run ingest.py.

## Ingest target JSON file use
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
    4
  ],
  "flux_column": "sap_flux",
  "quality_bitmask": "hard",
  "quality_masks": [
    [58420.0, 58423.0]
  ],
  "bin_time": 240,
  "period": 2.73,
  "plot_lc": true,
  "plot_fold": true,
  "polies": [],
  "trim_masks": [],
  "fitting_params": {}
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

## Generating a new ingest target JSON file
A new target JSON file can be generated with the `-n` or `--new-file` argument.

```sh
$ python3 ingest.py -n work/new_sys.json
```
where
- `-n`/`--new-file`: is the file to (over) write out

The new file will be written out with useful default values for most settings.
Alternatively, you can directly specify many of the values written when the file
is created by using the command line arguments documented above. The main 
exception is the target (`-t`) value as the target, file (`-f`) and new-file 
(`-n`) switches are mutually exclusive. The target value used will be taken from
the sys-name (if given) or the file name (if not). It is anticipated that you 
will need to edit the new ingest file before using it for ingest processing.

## Processing
The pipeline will carry out the following tasks for the specified system:
- the MAST portal is queried on the target/sectors for TESS/SPOC light-curves
- any located fits files are downloaded
- for each fits/sector:
  - the fits file is read and filtered based on the `--quality` argument
  - the data is filtered removing rows where the `--flux` column is NaN or <0.0
  - the `--quality-mask` ranges are applied - any data within these are excluded
  - if `--bin-time` given, the LC is binned to bins of this duration (seconds)
  - magnitudes are calculated from the `--flux` and corresponding error columns
    - a low order polynomial is subtracted to detrend the data
    - this also y-shifts the data so that the magnitudes are relative to zero
  - the primary epoch (most prominent eclipse) is found
  - if no `--period` specified, an estimate period will be found from eclipses
  - if `--plot-lc` the light-curve & primary epoch is plotted to a png
  - the magnitude data is phase-folded on the primary eclipse/period
    - a 1024 point interpolated phase-folded light-curve is derived from this
    - this is passed to a Machine-Learning model for system parameter estimation
    - if `--plot-fold` both folded light-curves are plotted to a png
  - any configured `polies` instructions are generated 
  - the `--trim-mask` ranges are applied to exclude excess data from the LC
    - if `--plot-lc` a second "_trimmed" light-curve is plotted to a png
  - the filtered & masked LC magnitude data is written to a JKTEBOP dat file
  - the estimated system parameters are used to write a JKTEBOP _task 3_ in file
    - any overrides from the fitting_params are applied
    - any polies generated above are appended
