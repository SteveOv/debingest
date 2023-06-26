"""
General utility functions not covered elsewhere.
"""
from typing import Union
from pathlib import Path
from argparse import Namespace, ArgumentParser
from types import SimpleNamespace
import json
import numpy as np
from lightkurve import LightCurve 


def set_up_argument_parser() -> ArgumentParser:
    """
    Handles command line arguments.  Sets up help text and the objects
    required to parse the command line and returns the resulting ArgumentParser.
    """
    ap = ArgumentParser(description="An ingest pipeline for TESS dEB \
light-curves. It searches for available light-curves for the requested \
identifier via the MAST portal and appropriate fits files are downloaded. \
For each fits file, the flux data is converted to relative magnitudes, \
a detrending polynomial is subtracted and some quality filters applied. \
Subsequently, the primary epoch and orbital period are calculated \
and a phase folded light-curve is passed to a Machine-learning model for \
system parameter estimation. \
Finally, the light-curve is prepared for fitting by JKTEBOP with the \
relative magnitudes being written to a text dat file and the primary \
epoch, period and estimated parameters used to create the in file which \
contains the JKTEBOP processing parameters and instructions.")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=Path, dest="file",
                    help="JSON file holding a target's ingest configuration")
    group.add_argument("-n", "--new-file", type=str, dest="new_file",
                    help="name of the new JSON configuration file to generate")

    ap.set_defaults(file=None, new_file=None)
    
    # a bit naughty as _optionals is private, but this is a useful clarification
    ap._optionals.title += " (you must give one of -h, -f or -n)"
    return ap


def new_ingest_config(target: str, **kwargs) -> Namespace:
    """
    Will return an ingest config which will be populated with valid
    default parameters, with overrides from any supplied kwargs.

    :target: the Id of the target system
    :**kwargs: the values to apply over the defaults
    """
    return Namespace(**{
        "target": target,
        "sys_name": target,
        "prefix": None,
        "output_dir": None,
        "sectors": [],
        "flux_column": "sap_flux",
        "exptime": None,
        "quality_bitmask": "default",
        "quality_masks": [],
        "bin_time": None,
        "period": None,
        "plot_lc": False,
        "plot_fold": False,
        "polies": [],
        "trim_masks": [],
        "fitting_params": {},
        **kwargs
    })


def write_ingest_config(file_name: Path, 
                        config: Union[Namespace, SimpleNamespace, dict], 
                        echo: bool = True):
    """
    Will save a new ingest json file to the indicated file_name. The file will
    be populated with default values. Any existing file will be overwritten.

    :file_name: the file to save
    :config: the config to write to file
    :echo: whether to echo the contents of the config to the terminal
    """
    if isinstance(config, Namespace) or isinstance(config, SimpleNamespace):
        config = vars(config)

    if echo:
        echo_ingest_config(config)

    with open(file_name, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Ingest config written to {f.name}")
    return


def read_ingest_config(file_name: Path, 
                       echo: bool = True) -> Namespace:
    """
    Reads the ingest config from the passed file.

    :file_name: the file to read from
    :echo: whether to echo the contents of the config to the terminal
    """
    print(f"Reading ingest config from {file_name}")
    with open(file_name, "r") as f:
        file_config = json.load(f)

    # We explicitly request target so we get a KeyError if not present
    config = new_ingest_config(file_config.pop("target"), **file_config)

    if echo:
        echo_ingest_config(config)
    return config


def echo_ingest_config(config: Union[Namespace, SimpleNamespace, dict],
                       show_nones: bool = False):
    """
    Will echo any members of the config except those of value of None
    (unless :show_nones: is true).
    """
    if isinstance(config, Namespace) or isinstance(config, SimpleNamespace):
        config = vars(config)

    print("The ingest configuration is:")
    for k, v in config.items():
        if show_nones or v:
            print(f"\t{k} = {v}")
    return


def new_sector_state(sector: int,
                     file_stem: str,
                     lc: LightCurve,
                     **kwargs) -> Namespace:
    """
    Creates a new sector state instance. This will store the ongoing state of
    the sector as it is passes along the pipeline.

    :sector: the sector number
    :file_stem: the stem name of any files to be generated from this sector
    :lc: the light-curve data for this sector
    :**kwags: anything else to be stored for this sector
    """
    return Namespace(**{
        "sector": sector,
        "file_stem": file_stem,
        "lc": lc,
        "fold_mags": None,
        "prim_epoch": None,
        "prim_epoch_ix": None,
        "period": None,
        "predictions": None,
        **kwargs
    })


def calculate_inc(bA: np.double,
                  rA_plus_rB: np.double,
                  k: np.double,
                  ecosw: np.double,
                  esinw: np.double) -> np.double:
    """
    Calculate the orbital inclination from the impact parameter.
    In training the mae of bA is usually lower, so we'll use that.
    inc = arccos( bA * rA * [1+esinw]/[1-e^2] )
    """
    rA = np.divide(rA_plus_rB, np.add(1, k))
    omega = np.arctan(np.divide(ecosw, esinw))
    e = np.divide(ecosw, np.cos(omega))
    cosi = np.multiply(np.multiply(rA, bA), 
                       np.divide(np.add(1, esinw), 
                                 np.subtract(1, np.power(e, 2))))
    return np.rad2deg(np.arccos(cosi))