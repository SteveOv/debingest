"""
General utility functions not covered elsewhere.
"""
from typing import Union, List
from pathlib import Path
from argparse import Namespace, ArgumentParser
from types import SimpleNamespace
import json
import numpy as np
from lightkurve import LightCurve
from pandas import DataFrame
from library import uncertainty_math as um

# pylint: disable=invalid-name

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

    ap.add_argument(dest="file", type=Path, metavar="TARGET-JSON-FILE",
                    help="the file which holds a target's ingest configuration")
    ap.add_argument("-n", "--new-file", action='store_true', required=False,
                    help="rather than processing the indicated file, ingest " \
                        + "will (over)write it with a default configuration")

    ap.set_defaults(file=None, new_file=False)
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
        "mission": "TESS",
        "author": "SPOC",
        "exptime": None,
        "flux_column": "sap_flux",
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

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Ingest config written to {f.name}")


def read_ingest_config(file_name: Path,
                       echo: bool = True) -> Namespace:
    """
    Reads the ingest config from the passed file.

    :file_name: the file to read from
    :echo: whether to echo the contents of the config to the terminal
    """
    print(f"Reading ingest config from {file_name}")
    with open(file_name, "r", encoding="utf-8") as f:
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
    if isinstance(config, (Namespace, SimpleNamespace)):
        config = vars(config)

    print("The ingest configuration is:")
    for k, v in config.items():
        if show_nones or v:
            print(f"{k:>18s} : {v}")


def echo_predictions(names: List[str],
                     flags: List[str],
                     preds: List[float],
                     stats: List[float],
                     pred_head: str = "Value",
                     stat_head: str = "Stat"):
    """
    Will echo the passed predicted values and accompanying statistic array.

    :names: the list of the predicted values
    :flags: single character flags to put next to the name
    :preds: the list of predicted values
    :stats: an accompanying statistic array, i.e.: std dev to go with means
    :pred_head: the heading to give the predicted values column
    :stat_head: the heading to give the stat column
    """
    if not len(names) == len(flags) == len(preds) == len(stats):
        raise ValueError("names, flags, preds and stats must be the same size")
    print(f"{'Predictions':>18}   :  {pred_head:^10s} ( {stat_head:^10s})")
    for n, f, v, s in zip(names, flags, preds, stats):
        print(f"{n:>18s} {f:1s} : {v:10.6f}  ({s:10.6f} )")


def new_sector_state(name: str,
                     tic_id: int,
                     sector: int,
                     file_stem: str,
                     lc: LightCurve,
                     **kwargs) -> Namespace:
    """
    Creates a new sector state instance. This will store the ongoing state of
    the sector as it is passes along the pipeline.

    :name: the system name
    :tic_id: the TESS TIC identifier for this system
    :sector: the sector number
    :file_stem: the stem name of any files to be generated from this sector
    :lc: the light-curve data for this sector
    :**kwags: anything else to be stored for this sector
    """
    return Namespace(**{
        "name": name,
        "tic_id": tic_id,
        "sector": sector,
        "file_stem": file_stem,
        "lc": lc,
        "fold_mags": None,
        "primary_epoch": None,
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


def append_calculated_params(df: DataFrame):
    """
    Append calculated param columns to the prediction DataFrame.

    :df: the pandas DataFrame to modify
    :returns: list of the base column names appended
    """
    original_cols = df.columns.values

    df["rA"], df["rA_sigma"] = \
        um.divide(df.rA_plus_rB, df.rA_plus_rB_sigma,
                  *um.add(1, 0, df.k, df.k_sigma))

    df["omega"], df["omega_sigma"] = \
        um.arctan(*um.divide(df.ecosw, df.ecosw_sigma,
                             df.esinw, df.esinw_sigma))

    df["e"], df["e_sigma"] = \
        um.divide(df.ecosw, df.ecosw_sigma,
                  *um.cos(df.omega, df.omega_sigma))

    rAbA, rAbA_sigma = \
        um.multiply(df.rA, df.rA_sigma, df.bA, df.bA_sigma)
    one_plus_esinw, one_plus_esinw_sigma = \
        um.add(1, 0, df.esinw, df.esinw_sigma)
    one_minus_e_sqr, one_minus_e_sqr_sigma = \
        um.subtract(1, 0, *um.power(df.e, df.e_sigma, 2, 0))
    cos_i, cos_i_sigma = \
        um.multiply(rAbA, rAbA_sigma,
                    *um.divide(one_plus_esinw, one_plus_esinw_sigma,
                               one_minus_e_sqr, one_minus_e_sqr_sigma))
    df["inc_calc"], df["inc_calc_sigma"] = \
        um.rad2deg(*um.arccos(cos_i, cos_i_sigma))

    new_names = [n for n in df.columns.values if n not in original_cols]
    return [n for n in new_names if not n.endswith("_sigma")]
