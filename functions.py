from typing import Union, Tuple, List, Callable
from pathlib import Path
from string import Template
import numpy as np
from scipy.signal import find_peaks
from astropy.time import Time
import astropy.units as u
from astropy.io import ascii
from lightkurve.lightcurve import LightCurve, FoldedLightCurve


def clip_mask_from_time_range(lc: LightCurve,
                              time_range: Tuple[np.double, np.double]) \
                                -> np.ndarray:
    """
    Returns a mask over the passed LightCurve for the indicated time range.

    !lc! the LightCurve to mask

    !time_range! the (from, to) time range to mask
    """
    time_from = to_time(np.min(time_range), lc)
    time_to = to_time(np.max(time_range), lc)

    mask = ((lc.time >= time_from) & (lc.time <= time_to))
    print(f"Clip range over [{time_from.format} {time_from}, "\
          f"{time_to.format} {time_to}] masks {sum(mask)} row(s).")
    return mask


def to_time(value: Union[int, float, np.double], lc: LightCurve) -> Time:
    """
    Converts the passed numeric value to an astropy Time.
    The magnitude of the time will be used to interpret the format to match LC.
    """
    if value < 4e4 and lc.time.format == "btjd":
        return Time(value, format=lc.time.format, scale=lc.time.scale)
    else:
        if value < 2.4e6:
            value += 2.4e6
        return Time(value, format="jd", scale=lc.time.scale)
    

def fit_polynomial(times: Time, 
                   ydata: u.Quantity, 
                   degree: int = 2, 
                   iterations: int = 2, 
                   res_sigma_clip: float = 1., 
                   reset_const_coeff: bool = False, 
                   include_coeffs: bool = False) \
                    -> Union[u.Quantity, Tuple[u.Quantity, List]]:
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


def append_magnitude_columns(lc: LightCurve, 
                             name: str = "delta_mag",
                             err_name: str = "delta_mag_err"):
    """
    This will append a magnitude and corresponding error column
    to the passed LightCurve.

    !lc! the LightCurve to update

    !name! the name of the new magnitude column
    
    !err_name! the name of the corresponding magnitude error column
    """
    lc[name] = u.Quantity(-2.5 * np.log10(lc.flux.value) * u.mag)
    lc[err_name] = u.Quantity(
        2.5 
        * 0.5
        * np.abs(
            np.subtract(
                np.log10(np.add(lc.flux.value, lc.flux_err.value)),
                np.log10(np.subtract(lc.flux.value, lc.flux_err.value))
            )
        )
        * u.mag)
    return 


def find_primary_epoch(lc: LightCurve) -> Tuple[Time, int]:
    """
    Will find the primary epoch (Time and index) of the passed LightCurve.
    This will be the "most prominent" eclipse found.
    
    !lc! the LightCurve to interogate
    """
    # Look for eclipse peaks in the data. We'll take the most prominent.
    (peak_ixs, peak_props) = find_eclipses(lc)
    strongest_peak_ix = peak_ixs[peak_props["prominences"].argmax()]
    return lc.time[strongest_peak_ix], strongest_peak_ix


def find_period(lc: LightCurve, 
                primary_epoch: Time) -> u.Quantity:
    """
    Will find the best estimate of the orbital period for the passed LightCurve.

    !lc! the LightCurve to parse.

    !primary_epoch! the time of the best primary - used to test our findings
    """
    # Look for eclipse peaks in the data. Should give us approx periods.
    (peak_ixs, _) = find_eclipses(lc)
    eclipse_diffs = np.diff(lc.time[peak_ixs])

    # Now use a periodogram restricted to a frequency range based on 
    # the known peak/eclipse spacing to find potential period values.
    # LK docs recommend normalize("ppm") and oversample_factor=100.
    max_fr = np.reciprocal(np.min(eclipse_diffs))
    min_fr = np.reciprocal(np.multiply(np.max(eclipse_diffs), 2))
    pg = lc.normalize(unit="ppm").to_periodogram("ls", 
                                                 maximum_frequency=max_fr,
                                                 minimum_frequency=min_fr,
                                                 oversample_factor=100)

    # The period should be a harmonic of the periodogram's max-power peak
    # Set this candidates up with the most likely last so we fall through.
    periods = [np.multiply(pg.period_at_max_power, f) for f in [1., 2.]]
    for period in periods:
        # Test periods by looking for 2 peaks on a folded LC. Rotated it so that
        # primary is at 0.1 to make primary & secondary eclipses distinct.
        # If the test fails we'll fall through on the last period option.
        fold_lc = phase_fold_lc(lc, primary_epoch, period, 0.90)
        eclipse_count = len(find_eclipses(fold_lc)[0])
        print(f"\tTesting period {period}: found {eclipse_count} eclipse(s)")
        if eclipse_count == 2:
            break
    return period


def find_eclipses(lc: LightCurve, width: int = 5) -> Tuple[List[int], dict]:
    """
    Will find a list of the indices of the eclipses in the passed LightCurve.
    Also return a peak_properties dictionary (see scipy.stats.find_peaks)

    !lc! the LightCurve to parse.

    !width! the minimum sample width of any peak
    """
    # We'll use a Std Dev as the minimum prominence of any peaks to find.  
    # Unless the data is very noisey, this should rule out all but eclipses.
    std_mag = np.std(lc["delta_mag"].data)
    return find_peaks(lc["delta_mag"], prominence=(std_mag, ), width=width)
 

def phase_fold_lc(lc: LightCurve,
                  primary_epoch: Time,
                  period: u.Quantity,
                  wrap_phase: float=0.75) -> FoldedLightCurve:
    """
    Perform a normalized phase fold on the passed LightCurve.
    By default, the primary_epoch/phase 0 will rotated to the 0.25 position.

    !lc! the LightCurve to produce the fold from

    !primary_epoch! the reference to fold around

    !period! the period to fold on

    !wrap_phase! the amout to wrap the fold to control where the primary appears
    """
    if not isinstance(wrap_phase, u.Quantity):
        wrap_phase = u.Quantity(wrap_phase)
    return lc.fold(period, 
                   epoch_time=primary_epoch, 
                   normalize_phase=True, 
                   wrap_phase=wrap_phase)
   

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