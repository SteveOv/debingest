from typing import Union, Tuple, List, Callable
from pathlib import Path
from string import Template
import numpy as np
from astropy.time import Time
from astropy.units import Quantity
from astropy.io import ascii
from lightkurve.lightcurve import LightCurve

def fit_polynomial(times: Time, 
                   ydata: Quantity, 
                   degree: int = 2, 
                   iterations: int = 2, 
                   res_sigma_clip: float = 1., 
                   reset_const_coeff: bool = False, 
                   include_coeffs: bool = False) \
                    -> Union[Quantity, Tuple[Quantity, List]]:
    """
    Will calculate a polynomial fit over the requested time range and y data
    values. The fit is iterative.  After each iteration the residuals are 
    evaluated against a threshold defined by the StdDev of the residuals 
    multiplied by res_sigma_clip; any datapoints with residuals greater than 
    this are excluded from subsequent iterations.  This approach will exclude 
    large ydata excursions, such as eclipses, from influencing the final fit.

    !times! - the times (x data)

    !ydata! - the data to fit to

    !degree! - the degree of polynomial to fit.  Defaults to 2.

    !iterations! - number of fit iterations to run.

    !res_sigma_clip! - the factor applied to the residual StdDev to 
                    calculate the clipping threshold for each new iteration.

    !reset_const_coeff! - if True the const coefficient will be reset to zero 
                        before calculating the final fit.

    !include_coeffs! - if True the coefficients will be returned in addition 
                        to the fitted ydata.

    returns the fitted y data and optionally the coefficients used to 
    generate it.
    """
    time_from_jd, time_to_jd = np.min(times).jd, np.max(times).jd
    pivot_ix = int(np.floor(len(times) / 2))
    pivot_jd = times[pivot_ix].jd  
    time_values = times.jd - pivot_jd

    print(f"Fitting polynomial (degree = {degree}) where x is JD "
        + f"{time_from_jd} to {time_to_jd} (minus pivot at {pivot_jd}). ")

    print(f"\tRunning {iterations} iterations fitting data with residual "
        + f"within {res_sigma_clip} sigma.")
    
    fit_mask = [True] * len(ydata)
    for iter in np.arange(iterations, 0, -1):
        coeffs = np.polynomial.polynomial.polyfit(time_values[fit_mask], 
                                                  ydata.value[fit_mask], 
                                                  deg=degree, 
                                                  full=False)
        
        if iter > 1:
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values)
            resids = ydata.value - fit_ydata
            fit_mask &= (np.abs(resids) <= (np.std(resids)*res_sigma_clip))
        else:
            if reset_const_coeff:
                print(f"...final 'const' coeff reset to zero (on request)...")
                coeffs[0] = 0
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values) * ydata.unit
            print(f"\tCompleted; y={poly_func} "
                  + f"(sigma(fit_ydata)={np.std(fit_ydata):.6e})")

    return (fit_ydata, coeffs) if include_coeffs else fit_ydata


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


def write_task3_in_file(file_name: Path, **params):
    """
    Writes a JKTEBOP task3 .in file based on applying the passed params/token
    values to the task3.in.template file.

    !file_name! the name and path of the file to write

    !params! a dictionary of param tokens/keys and values
    """
    with open(file_name, mode="w") as of:
        with open(Path("./task3.in.template"), "r") as tpf:
            template = Template(tpf.read())

        if "file_name_stem" not in params:
            params["file_name_stem"] = file_name.stem

        # Will error if any expected tokens are not present.
        of.write(template.substitute(**params))
        print(f"Writing JKTEBOP task3 in file to '{file_name.name}'")
    return