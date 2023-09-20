"""Functions for accessing external catelogue data"""
from typing import Tuple
from pathlib import Path
from inspect import getsourcefile
import warnings

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.io import ascii as ascii_io

def get_tess_ebs_ephemeris_for_tic(tic_id: int) -> Tuple[Time, u.Quantity] :
    """
    Read ephemeris information from the J/ApJS/258/16/tess-ebs catalogue data.

    :tic_id: the numeric TESS ID of the target object
    :returns: a tuple of (primary_epoch, period) or (None, None) if not found
    """
    data_dir = Path(getsourcefile(lambda:0)).parent / "data/J_ApJS_258_16/"

    with warnings.catch_warnings():
        # Suppress the warning about the unsuported "datime" unit in the ReadMe
        warnings.filterwarnings("ignore", message="'\"datime\"' did not parse")
        table = ascii_io \
                    .get_reader(ascii_io.Cds, readme=f"{data_dir}/ReadMe") \
                    .read(f"{data_dir}/tess-ebs.dat")

    rows = table[table["TIC"] == tic_id]
    if len(rows) > 0:
        # TODO: Have yet to find definitive info on what the format/scale is
        primary_epoch = Time(rows[0]["BJD0"], format="btjd", scale="utc")
        period = np.mean(rows["Per"].value) * u.d
    else:
        primary_epoch = period = None

    return primary_epoch, period
