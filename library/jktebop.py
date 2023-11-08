"""
Functions for interacting with or preparing data for JKTEBOP.
"""
from typing import List, Union, Callable, Tuple
from pathlib import Path
from string import Template
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.io import ascii as io_ascii
from lightkurve import LightCurve
from library import lightcurves

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

def write_data_to_dat_file(lc: LightCurve,
                           file_name: Path,
                           column_names: List[str] = None,
                           column_formats: List[Union[str, Callable]] = None,
                           overwrite: bool = True,
                           include_headers: bool = False,
                           comment_prefix: str = "# ",
                           delimiter: str = "\t"):
    """
    This will write the contents of the passed LightCurve columns to a
    JKTEBOP compatible delimited dat file.

    :lc: the source LightCurve

    :file_name: the name and location of the file to write

    :column_names: the columns (in lc) to write (time, delta_mag, delta_mag_err)

    :column_formats: the formats used to write the columns to file text

    :overwrite: whether to overwrite or append to the file

    :include_headers: whether to include a header row
    
    :comment_prefix: prefix to use for comments/header row

    :delimiter: the column delimiter to use 
    """

    # If these are not specified default to including all of them.
    if column_names is None:
        column_names = ["time", "delta_mag", "delta_mag_err"]
    if column_formats is None:
        column_formats = [lambda t: f"{t.jd-2.4e6:.6f}", "%.6f", "%.6f"]

    # Check and set up the formats.
    if len(column_names) != len(column_formats):
        raise ValueError("Different number of column_names to column_formats."
                         + "Each column must have an equivalent format.")

    formats = dict(zip(column_names, column_formats))
    columns = [lc[column_name] for column_name in column_names]
    fmt = "commented_header" if include_headers else "no_header"

    if len(column_names) <= 6:
        column_text = f"columns {column_names}"
    else:
        column_text = f"{len(column_names)} columns"
    print(f"Writing {len(lc)} rows(s) for {column_text} to '{file_name.name}'")

    io_ascii.write(columns, output = file_name, format = fmt,
                   names = column_names, formats = formats,
                   comment = comment_prefix, delimiter = delimiter,
                   overwrite = overwrite)


def write_task3_in_file(file_name: Path,
                        append_lines: List[str]=None,
                        **params):
    """
    Writes a JKTEBOP task3 .in file based on applying the passed params/token
    values to the task3.in.template file.

    :file_name: the name and path of the file to write

    :append_lines: lines to optionally append at the end of the in file

    :params: a dictionary of param tokens/keys and values
    """
    with open(file_name, mode="w", encoding="utf8") as of:
        print(f"Writing JKTEBOP task3 in file to '{file_name.name}'")

        with open(Path("./library/task3.in.template"),
                  "r", 
                  encoding="utf8") as tpf:
            template = Template(tpf.read())

        if "file_name_stem" not in params:
            params["file_name_stem"] = file_name.stem

        # Preempt JKTEBOP's validation rules
        if "L3" in params and params['L3'] < 0.:
            print(f"Increasing L3 from {params['L3']} to the",
                  "minimum input value supported by JKTEBOP of 0.0")
            params['L3'] = 0.

        if "rA_plus_rB" in params and params["rA_plus_rB"] > 0.8:
            print(f"Decreasing rA_plus_rB from {params['rA_plus_rB']} to the",
                  "maximum input value supported by JKTEBOP of 0.8")
            params["rA_plus_rB"] = 0.8

        # Will error if any expected tokens are not present.
        of.write(template.substitute(**params))

        # Add on any lines to be appended to the file
        if append_lines:
            # Newline so we don't append directly onto current final line.
            of.write("\n")
            of.writelines(append_lines)


def build_polies_for_lc(lc: LightCurve,
                        polies: List[dict] = None) -> List[str]:
    """
    Will build poly instruction lines for the passed light-curve based
    on the configuration in passed array of poly configuration items. There are
    two types of poly config differentiated by whether they have a "date_range"
    or "gap_threshold" parameter.

    The expectation is that zero or more "date_range" polies will be listed 
    first. These will be triggered if their date range overlaps with that of 
    the passed light-curve. 
    
    The last poly specified may be an optional "gap_threshold" poly to act as 
    a fall-back if no "date_range" polies are found or have been applied. 
    A "gap_threshold" poly will calculate suitable date ranges by splitting 
    the timeseries data on gaps > threshold and build a poly instruction for 
    each.  A single poly instruction covering the whole lightcurve will be 
    built if no gaps found.

    Once a "date_range" poly has been triggered subsequent "gap_threshold" 
    poly instructions are ignored and vice-versa.

    The defaults for a poly term, degree and gap_threshold are "sf", 1 & 0.5 (d) 

    :lc: the source light-curve

    :polies: an array of poly configurations to be evaluated in order
    """
    lines = []

    if polies and isinstance(polies, list):
        flags = {}  # Control application of mutually exclusive config per term
        MANUAL_POLY = 1
        AUTO_POLY = 2
        lc_start = lc.time.min()
        lc_end = lc.time.max()

        for ix, poly in enumerate(polies):
            term = poly.get("term") or "sf"
            deg = poly.get("degree") or 1

            # The two types of config are distinguished by the presence/absence
            # of a time_range. Either we have an excplicit time_range given...
            if "time_range" in poly:
                if flags.get(term) != AUTO_POLY:
                    rng = lightcurves.to_time([
                        np.min(poly["time_range"]),
                        np.max(poly["time_range"])
                    ], lc)

                    if lc_start < rng[0] < lc_end or lc_start < rng[1] < lc_end:
                        lines += ["\n" + build_poly_instr(rng, term, deg)]
                        flags[term] = MANUAL_POLY
                        print(f"Manual poly for {term} (degree={deg}) "\
                              + f"over JD {rng[0].jd:.2f} to {rng[1].jd:.2f}.")
                else:
                    print(f"Skipping poly[{ix}]: auto poly already applied")

            else:
                if not flags.get(term):
                    # ... or we use a gap threshold, where the lc will be split
                    # into time ranges separated by gaps > the threshold.
                    thold = TimeDelta((poly.get("gap_threshold") or 0.5) * u.d)
                    rng_ixs = lightcurves.find_indices_of_segments(lc, thold)

                    for (start_ix, end_ix) in rng_ixs:
                        rng = Time([
                            lc.time[start_ix] - .02 * u.d, # covers rounding err
                            lc.time[end_ix] + .02 * u.d
                        ])

                        lines += ["\n" + build_poly_instr(rng, term, deg)]
                        flags[term] = AUTO_POLY
                        print(f"Auto poly for '{term}' (degree={deg}) "\
                              + f"over JD {rng[0].jd:.2f} to {rng[1].jd:.2f} "\
                              + f"splitting on gaps > {thold} d.")
                elif flags.get(term) == AUTO_POLY:
                    print("Skipping auto poly - an auto poly already applied")
                else:
                    print("Skipping auto poly - manual poly already applied")
    return lines


def build_poly_instr(time_range: Tuple[Time, Time],
                     term: str = "sf",
                     degree: int = 1) -> str:
    """
    Builds up and returns a JKTEBOP 'in' file fitted poly instruction

    :time_range: the (from, to) range to fit it over - will pivot on the mean

    :term: the term to fit - defaults to sf

    :degree: the degree of the polynomial to fit - defaults to 1 (linear)
    """
    fit_flags = ' '.join(["1" if coef <= degree else "0" for coef in range(6)])
    time_from = np.min(time_range).jd - 2.4e6
    time_to = np.max(time_range).jd - 2.4e6
    pivot = np.mean([time_from, time_to])
    return f"poly  {term}  {pivot:.2f}  0.0 0.0 0.0 0.0 0.0 0.0  " \
            f"{fit_flags}  {time_from:.2f} {time_to:.2f}"
