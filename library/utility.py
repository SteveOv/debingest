"""
General utility functions not covered elsewhere.
"""
from pathlib import Path
from argparse import Namespace, ArgumentParser
import json
import numpy as np


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

    # Must have 1 of these two. User must specify the target (& all other args) 
    # at the command line or specify a json file so the args are read from file
    # (cmd line args still read as an override of file [specific code for this])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--target", type=str, dest="target",
                    help="search identifier for the target system to ingest")
    group.add_argument("-f", "--file", type=Path, dest="file",
                    help="JSON file holding a target's ingest configuration")
    group.add_argument("-n", "--new-file", type=str, dest="new_file",
                    help="name of the new JSON configuration file to generate")

    # These are not part of the group above
    ap.add_argument("-sys", "--sys-name", type=str, 
                    dest="sys_name", default=None,
                    help="the system name if different to target")
    ap.add_argument("-s", "--sector", type=int, 
                    dest="sectors", action="append", metavar="SECTOR",
                    help="specific sector to find (multiple -s args supported)")
    #ap.add_argument("-m", "--mission", type=str, dest="mission", default="TESS",
    #                help="the source mission: currently only supports TESS")
    #ap.add_argument("-a", "--author", type=str, dest="author", default="SPOC",
    #                help="the data's author: currently only supports SPOC")
    ap.add_argument("-fl", "--flux", type=str, 
                    dest="flux_column", default="sap_flux",
                    choices=["sap_flux", "pdcsap_flux"],
                    help="the flux column to use (defaults to sap_flux)")
    ap.add_argument("-e", "--exptime", type=str, dest="exptime",
                    help="exposure time/cadence with options of long, short, \
                        fast or an exact time in seconds (any if omitted)")
    ap.add_argument("-q", "--quality", type=str, dest="quality_bitmask",
                    help="quality bitmask to exclude poor quality data: may be \
                        a numerical bitmask or text {none, default, hard, \
                        hardest} with a default value of default")
    ap.add_argument("-qm", "--quality-mask", type=np.double, nargs=2, 
                    dest="quality_masks", action="append", metavar="TIME",
                    help="a time range (from, to) to mask out problematic data \
                        from light-curves prior to processing (multiple -qm \
                        args supported)")
    ap.add_argument("-p", "--period", type=np.double, dest="period",
                    help="the period of the system (in days) if you wish to \
                    override the ingest calculated period")
    ap.add_argument("-pl", "--plot-lc", dest="plot_lc",
                    action="store_true", required=False,
                    help="plot of each sector's light-curve to a png file")
    ap.add_argument("-pf", "--plot-fold", dest="plot_fold",
                    action="store_true", required=False,
                    help="plot of each sector folded data to a png file")
    ap.add_argument("-tm", "--trim-mask", type=np.double, 
                    nargs=2, dest="trim_masks", action="append", metavar="TIME",
                    help="a time range (from, to) to trim from the final \
                        light-curve to reduce the data processing on fitting \
                        (multiple -tm args supported)")

    ap.set_defaults(target=None, file=None, new_file=None, sys_name=None,
                    sectors=[], mission="TESS", author="SPOC", exptime=None,
                    flux_column="sap_flux", quality_bitmask="default", 
                    quality_masks=[], period=None, 
                    plot_lc=False, plot_fold=False, 
                    polies=[], trim_masks=[], fitting_params={})
    
    # a bit naughty as _optionals is private, but this is a useful clarification
    ap._optionals.title += " (you must give one of -h, -t, -f or -n)"
    return ap


def echo_ingest_parameters(args: Namespace):
    """
    Will echo any members of the args that don't have a value of None.
    """
    print("Ingest parameters being used:")
    for k, v in vars(args).items():
        if v:
            print(f"\t{k} = {v}")
    return


def save_new_ingest_json(file_name: Path, 
                         args: Namespace):
    """
    Will save a new ingest json file to the indicated file_name. The file will
    be populated with default values. Any existing file will be overwritten.

    !file_name! the file to save

    !arg! the args Namespace which will contain any requested values or defaults

    !target! the target name or will default to file_name.stem if omitted
    """
    new_args = vars(args)
    
    # Remove stuff that should not be seen in the JSON file
    for k in ["file", "new_file", "mission", "author"]:
        if k in new_args:
            new_args.pop(k)

    new_args["target"] = new_args.get("sys_name") or file_name.stem

    # Add some dummy values to demonstrate how various settings are written
    if new_args["quality_masks"] is None or len(new_args["quality_masks"]) == 0:
        new_args["quality_masks"] = [[45000.0, 45001.0]]

    # Set up a default auto-poly known to work well on TESS light-curves
    new_args["polies"] = [
        { "term": "sf", "degree": 1, "gap_threshold": 0.5 }
    ]

    if new_args["trim_masks"] is None or len(new_args["trim_masks"]) == 0:
        new_args["trim_masks"] = [[45001.0, 45002.0]]

    new_args["fitting_params"]["dummy_token"] = "value"
    
    echo_ingest_parameters(args)
    with open(file_name, "w") as f:
        json.dump(new_args, f, ensure_ascii=False, indent=2)
        print(f"New ingest target JSON file saved to '{f.name}'")
    return


def calculate_inclination(bA: np.double,
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