# debingest

## Detached Eclipsing Binary Light-curve ingest pipeline

This code base was developed in VSCode (on Ubuntu 22.04) within the context of
an [Anaconda 3](https://www.anaconda.com/) environment named **debingest**. 
This environment is configured to support _Python 3.8_ and the libraries on 
which the code is dependent.

To set up the **debingest** environment, having first cloned the GitHub repo, 
open a Terminal, navigate to this directory and run the following command;
```sh
$ conda env create -f environment.yml
```

You will need to activate this environment, whenever you wish to use the ingest
pipeline, with the following command;
```sh
$ conda activate debingest
```

The role of the pipeline is to prepare data for dEB analyis and fitting
with [JKTEBOP](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html).

## Command line Use
The entry point for this pipeline is `ingest.py` with example usage shown 
below. The first example shows running a target ingest based on the 
configuration given in an existing config json file.

```sh
$ python3 ingest.py -f examples/cw_eri.json
```

This second example shows how to generate a new default ingest configuration
json file. Once generated you will need to edit this file to set up the 
configuration specific to your target.
```sh
$ python3 ingest.py -n examples/new_sys.json
```

> If you first run `chmod +x ingest.py` (or equivalent) in the terminal 
> you remove the need to specify python3 whenever you run ingest.py.

## The ingest JSON file
A target's ingest configuration is held in a json file and passed to ingest.py 
with the `-f` or `--file` argument. An example is shown below with the 
parameters, broadly listed in the order they are used.

```json
{
    "target": "CW Eri",
    "sys_name": "CW Eridani",
    "prefix": "cw_eri",
    "output_dir": "drop/cw_eri",
    "sectors": [
        4,
        31
    ],
    "flux_column": "sap_flux",
    "exp_time": "short",
    "quality_bitmask": "hardest",
    "quality_masks": [
        [58420.00, 58423.00]
    ],
    "bin_time": null,
    "period": 2.72837,
    "plot_lc": true,
    "plot_fold": true,
    "polies": [
        { "term": "sf", "degree": 1, "gap_threshold": 0.5 }
    ],
    "trim_masks":[
    ],
    "fitting_params": {
        "qphot": 0.836,
        "L3": 0.080,
        "LD_A": "pow2",
        "LD_B": "pow2",
        "LD_A1": 0.64,
        "LD_B1": 0.64,       
        "LD_A1_fit": 1,
        "LD_B1_fit": 1,
        "LD_A2": 0.47,
        "LD_B2": 0.50,
        "LD_A2_fit": 0,
        "LD_B2_fit": 0
    }
}
```

Many of these configuration parameters are optional and may be removed or set
to `null` if the default behaviour is required.

The time values for `quality_mask`, `trim_mask`, or `polies` date ranges are 
interpreted as BTJD (if < 40 000), reduced JD (< 2.4e6) or JD (>= 2.4e6) all 
with the scale matching the corresponding light-curve.

## Processing and parameter usage
This section describes the ingest pipelines stages and how the parameters
shown above are used. Each major stage, 1 to 6, is applied to all of the 
target's matching sectors/light-curves before moving on to the next stage.

The `target` is a compulsory search identifier suitable for locating your target
in the MAST portal (object name or TIC are known to work). The `sys_name` is
an optional name for use in plots and diagnostics messages which will default
to the _target_ value if omitted.

The optional `prefix` and `output_dir` values are used to identify where output 
files are written and how they're named. The prefix is used as a prefix for all
files. If omitted, the prefix will be derived from the _sys_name_ and the output
dir will be 'staging/`prefix`/'.

### 1. Target search and asset download
The `target`, `sectors` and `exptime` are used when when querying MAST for 
available timeseries data assets.  Both are optional and if not given they are 
assumed to be equivalent to 'any'. Suitable values for _exptime_ are long, 
short, fast or a numeric value in seconds, with 'short' being appropriate for 
TESS's 120 s cadence light-curve data.  

The `flux_column` may be set to **sap_flux** (the default value) or pdcsap_flux 
to indicate the source of the flux data to be used.

> The ingest pipeline makes extensive use of the Lightkurve library. For more
> information on the _target_, _sectors_, _exptime_, *flux_column* and 
> *quality_bitmask* values see the Lightkurve search and download documentation 
> [here](http://docs.lightkurve.org/reference/api/lightkurve.search_lightcurve.html)

### 2. Pre-processing data
The optional `quality_bitmask` and `quality_masks` are used to mask out poor 
quality data from a light-curve prior to processing. The *quality_bitmask* may
be set to none, **default**, hard, hardest or a numeric bitmask to be applied 
against the light-curves' QUALITY flag. The *quality_masks* are time ranges 
(from, to) over which all data will be masked from subsequent processing.

> The *quality_masks* and *trim_masks* both take zero or more two-item 
> arrays, each giving the start and end of a time range. For example, the 
> following defines a pair of ranges from JD 2451005 to 2451007 and 2451020 to 
> 2451022 (inclusive): `[[51005.0, 51007.0], [51020.0, 51022.0]]`

The optional `bin_time` parameter may be set to a time value (in seconds) to 
which the light-curve data will be (re)binned. This will be ignored if not set 
or it is given a value which is less than or equal to the _exptime_ of the data 
as downloaded. 

Now that each light-curve is downloaded, opened, masked and optionally binned
the fluxes are detrended and used to derive relative magnitudes.

### 3. Finding an orbital ephemeris
We now have the light-curves' data in a useable state for processing. First the 
*primary_epoch* is located by selecting the 'most prominent' eclipse in the 
light-curve. With the `period` (in days), this defines the dEB's ephemeris. The
_period_ will be estimated using a periodogram of the light-curve if not given.  

The `plot_lc` flag controls whether each light-curve, with its *primary_epoch*
highlighted, is plotted to a png file.

### 4. Phase folding the light-curves
The ephemeris is used to phase fold the light-curve data, and 1024 point
single phase reduced light-curves are derived for subsequent inspection for 
system parameter estimation. The `plot_fold` flag controls whether a plot
of each folded light-curve, overlaid with the model, is plotted to a png file.

### 5. System parameter estimation
The reduced folded light-curves are passed to a Machine-Learning model, 
trained to characterize folded dEB light-curves, for parameter estimation.
This us gives estimates of the following fitting parameters:
- `rA_plus_rB` (sum of the relative radii)
- `k` (ratio of the relative radii)
- `bA` (primary impact parameter)
- `inc` (orbital inclination)
- `ecosw` and `esinw` (combined eccentrity and argument of periastron)
- `J` (surface brightness ratio)
- `L3` (amount of third light)

### 6. Creating JKTEBOP fitting files
#### 6-1. Generating JKTEBOP poly instructions
During fitting JKTEBOP may alter the light-curve data by fitting polynomials
to chosen terms, given as `poly` instructions. Generally we instruct it to fit
low order polynomials to the `sf` (scale factor) term to normalize the data. 
Each _poly_ instruction applies over a time range and as a rule of thumb each 
should be a contiguous region of light-curve.

The `polies` config parameter defines what, if any, poly instructions are 
generated. There are two types: `time_range` (manual poly) configs apply over 
a user defined time range and `gap_threshold` configs (auto poly) give a gap 
size (in days) which acts as a boundary between one or more automatically 
generated time ranges. In both cases a `term` and polynomial `degree` may be 
specified (they default to 'sf' and 1 if omitted).

```json
  "polies": [
    { "term": "sf", "degree": 1, "time_range": [58410.00, 58420.00] },
    { "term": "sf", "degree": 1, "time_range": [58424.00, 58434.00] },
    { "term": "sf", "degree": 1, "gap_threshold": 0.5 }
  ]
```

The resulting poly instructions, to be written to the JTEBOP '.in' file, may
look similar to those shown below. This shows the result of the two manual
polies above.  In the absence of these, the auto-poly would generate a similar
output but the date values would be automatically generated from detecting
gaps in the data.

```
poly  sf  58415.00  0.0 0.0 0.0 0.0 0.0 0.0  1 1 0 0 0 0  58410.00 58420.00
poly  sf  58429.00  0.0 0.0 0.0 0.0 0.0 0.0  1 1 0 0 0 0  58424.00 58434.00
```

Poly configs are processed in order, with the supported pattern being zero or 
more _manual polies_ followed by an optional _auto-poly_ (as shown above). For 
each of the target's light-curves, the manual polies will generate an 
instruction if there is an overlap with the light-curve on the time axis. The 
_auto-poly_ will be used only where no _manual polies_ were applied. The two
types of poly config are mutually exclusive, triggering of one type for a given
light-curve will cause those of the other type to be subsequently ignored. 

#### 6-2. Trimming light-curves
The optional `trim_masks` config parameter controls what data, if any, is now
trimmed from the light-curves. Unlike *quality_masks* which are used to mask 
poor quality data prior to processing, the *trim_masks* are applied towards
the end of the pipeline to reduce the data passed on to JKTEBOP for fitting.
If `plot_lc` is set the trimmed light-curves will be plotted to png files.

#### 6-3. Writing the JKTEBOP .in/.dat files
Finally the processed and reduced light-curve time and magnitude data is 
written to a JKTEBOP compatible '.dat' file. The parameters for fitting these 
data are built up from a set of default values and ephemeris, overlaid with the 
estimated fitting parameters from the ML model and again with any user 
specified overrides given in the `fitting_params` config. These, along with the
poly instructions previously generated, are written to a '.in' file which is
the input instruction and parameters for JKTEBOP to fit the light-curve (the 
template for this file is found in library/task3.in.template). The default 
fitting params can be seen at the foot of ingest.py.
