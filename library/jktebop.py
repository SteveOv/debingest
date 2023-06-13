"""
Functions for interacting with or preparing data for JKTEBOP.
"""
from typing import List, Union, Callable, Tuple
from pathlib import Path
from string import Template
import numpy as np
from astropy.time import Time
from astropy.io import ascii
from lightkurve import LightCurve


def write_data_to_dat_file(lc: LightCurve, 
                           file_name: Path,
                           column_names: List[str] \
                            = ["time", "delta_mag", "delta_mag_err"],
                           column_formats: List[Union[str, Callable]] \
                            = [lambda t: f"{t.jd-2.4e6:.6f}", "%.6f", "%.6f"],
                           overwrite: bool = True,
                           include_headers: bool = False,
                           comment_prefix: str = "# ",
                           delimiter: str = "\t"): 
    """
    This will write the contents of the passed LightCurve columns to a
    JKTEBOP compatible delimited dat file.

    !lc! the source LightCurve

    !file_name! the name and location of the file to write

    !column_names! the names of the columns (in lc) to write

    !column_formats! the formats used to write the columns to file text

    !overwrite! whether to overwrite or append to the file

    !include_headers! whether to include a header row
    
    !comment_prefix! prefix to use for comments/header row

    !delimiter! the column delimiter to use 
    """

    # If these are not specified default to including all of them.
    if column_names is None:
        column_names = lc.colnames
    if column_formats is None:
        column_formats = ["%s"] * len(column_names)

    # Check and set up the formats.
    if len(column_names) != len(column_formats):
        raise ValueError("Different number of column_names to column_formats." 
                         + "Each column must have an equivalent format.")

    formats = dict(zip(column_names, column_formats))
    columns = [lc[column_name] for column_name in column_names]
    format = "commented_header" if include_headers else "no_header"

    if len(column_names) <= 6:
        column_text = f"columns {column_names}"
    else:
        column_text = f"{len(column_names)} columns"
    print(f"Writing {len(lc)} rows(s) for {column_text} to '{file_name.name}'")

    ascii.write(columns, output = file_name, format = format, 
                names = column_names, formats = formats, 
                comment = comment_prefix, delimiter = delimiter, 
                overwrite = overwrite)
    return


def write_task3_in_file(file_name: Path, append_lines: List[str]=[], **params):
    """
    Writes a JKTEBOP task3 .in file based on applying the passed params/token
    values to the task3.in.template file.

    !file_name! the name and path of the file to write

    !params! a dictionary of param tokens/keys and values
    """
    with open(file_name, mode="w") as of:
        with open(Path("./library/task3.in.template"), "r") as tpf:
            template = Template(tpf.read())

        if "file_name_stem" not in params:
            params["file_name_stem"] = file_name.stem

        # Will error if any expected tokens are not present.
        of.write(template.substitute(**params))

        # Add on any lines to be appended to the file
        if append_lines:
            # Newline so we don't append directly onto current final line.
            of.write("\n")
            of.writelines(append_lines)
        print(f"Writing JKTEBOP task3 in file to '{file_name.name}'")
    return


def build_poly_instr(time_range: Tuple[Time, Time],
                     term: str = "sf",
                     degree: int = 1) -> str:
    """
    Builds up and returns a JKTEBOP 'in' file fitted poly instruction

    !time_range! the (from, to) range to fit it over - will pivot on the mean

    !term! the term to fit - defaults to sf

    !degree! the degree of the polynomial to fit - defaults to 1 (linear)
    """
    fit_flags = ' '.join(["1" if coef <= degree else "0" for coef in range(6)])
    time_from = np.min(time_range).jd - 2.4e6
    time_to = np.max(time_range).jd - 2.4e6
    pivot = np.mean([time_from, time_to])
    return f"poly  {term}  {pivot:.2f}  0.0 0.0 0.0 0.0 0.0 0.0  " \
            f"{fit_flags}  {time_from:.2f} {time_to:.2f}"