"""
Functions for interacting with LightCurve data
"""
from typing import Union, Tuple, List, Generator
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate.interpolate import interp1d
from astropy.time import Time, TimeDelta
from astropy.units import Quantity, quantity_input
import astropy.units as u

from lightkurve.lightcurve import LightCurve, FoldedLightCurve

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

def mask_from_time_range(lc: LightCurve,
                         time_range: Tuple[np.double, np.double]) \
                            -> np.ndarray:
    """
    Returns a mask over the passed LightCurve for the indicated time range.

    :lc: the LightCurve to mask

    :time_range: the (from, to) time range to mask
    """
    time_from = to_time(np.min(time_range), lc)
    time_to = to_time(np.max(time_range), lc)

    mask = (lc.time >= time_from) & (lc.time <= time_to)
    print(f"Range over [{time_from.format} {time_from}, "\
          f"{time_to.format} {time_to}] selects {sum(mask)} row(s).")
    return mask


def to_time(value: Union[Time, np.double, Tuple[np.double], List[np.double,]],
            lc: LightCurve) \
                -> Time:
    """
    Converts the passed numeric value to an astropy Time.
    The magnitude of the time will be used to interpret the format to match LC.

    :value: the value to be converted

    :lc: the light-curve to match format with
    """
    if isinstance(value, list) or isinstance(value, tuple):
        return Time([to_time(v, lc) for v in value])
    if isinstance(value, Time):
        return value
    elif value < 4e4 and lc.time.format == "btjd":
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

    :times: - the times (x data)

    :ydata: - the data to fit to

    :degree: - the degree of polynomial to fit.  Defaults to 2.

    :iterations: - number of fit iterations to run.

    :res_sigma_clip: - the factor applied to the residual StdDev to 
                     calculate the clipping threshold for each new iteration.

    :reset_const_coeff: - if True the const coefficient will be reset to zero 
                        before calculating the final fit.

    :include_coeffs: - if True the coefficients will be returned in addition 
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
    for iteration in np.arange(iterations, 0, -1):
        coeffs = np.polynomial.polynomial.polyfit(time_values[fit_mask],
                                                  ydata.value[fit_mask],
                                                  deg=degree,
                                                  full=False)

        if iteration > 1:
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values)
            resids = ydata.value - fit_ydata
            fit_mask &= (np.abs(resids) <= (np.std(resids)*res_sigma_clip))
        else:
            if reset_const_coeff:
                print("...final 'const' coeff reset to zero (on request)...")
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

    :lc: the LightCurve to update

    :name: the name of the new magnitude column
    
    :err_name: the name of the corresponding magnitude error column
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

@quantity_input(period_hint="time")
def find_primary_epoch(lc: LightCurve,
                       pe_hint: Time = None,
                       period_hint: Quantity = None) -> Tuple[Time, int]:
    """
    Will find the primary epoch (Time and index) of the passed LightCurve.
    This will be the "most prominent" eclipse found. If both supplied, the two
    hint values will be used to test whether the peak found is a period multiple
    of the hinted primary epoch. This is particularly useful where the two 
    types of eclipse are of similar magnitude and the peak finder may select
    a secondary eclipse.
    
    :lc: the LightCurve to interogate
    :pe_hint: hint value for the primary epoch
    :period_hint: hint value of the period
    """
    (peak_ixs, peak_props) = find_eclipses(lc)
    if pe_hint is None or period_hint is None:
        # No hints so just take the most prominent peak
        strongest_peak_ix = peak_ixs[peak_props["prominences"].argmax()]
    else:
        # If we have hints we can use them to reverse down the list of prominent
        # peaks to make sure we are selecting the most prominent one that is
        # also near the orbital phase 0. Particularly useful if the two eclipse
        # types are of similar depth - can stop us setting the pe to a secondary
        strongest_peak_ix = None
        sorted_prom_ixs = reversed(np.argsort(peak_props["prominences"]))
        for prom_ix in sorted_prom_ixs:
            strong_peak_ix = peak_ixs[prom_ix]
            period_offset = (lc.time[strong_peak_ix] - pe_hint) / period_hint
            period_offset = period_offset.value

            if np.isclose(period_offset, np.floor(period_offset), atol=0.2):
                # This peak is near phase 0 - we'll use it.
                strongest_peak_ix = strong_peak_ix
                break

            if strongest_peak_ix is None:
                # Make sure we have a fallback position of the most prominent
                # peak for use if no subsequent peaks meet the hint criteria
                strongest_peak_ix = strong_peak_ix

    return lc.time[strongest_peak_ix], strongest_peak_ix


def find_period(lc: LightCurve,
                primary_epoch: Time) -> u.Quantity:
    """
    Will find the best estimate of the orbital period for the passed LightCurve.

    :lc: the LightCurve to parse.

    :primary_epoch: the time of the best primary - used to test our findings
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
    period = periods[-1]
    for test_per in periods:
        # Test periods by looking for 2 peaks on a folded LC. Rotated it so that
        # primary is at 0.1 to make primary & secondary eclipses distinct.
        # If the test fails we'll fall through on the last period option.
        fold_lc = phase_fold_lc(lc, primary_epoch, test_per, 0.90)
        eclipse_count = len(find_eclipses(fold_lc)[0])
        print(f"\tTesting period {test_per}: found {eclipse_count} eclipse(s)")
        if eclipse_count == 2:
            period = test_per
            break
    return period


def find_eclipses(lc: LightCurve, width: int = 5) -> Tuple[List[int], dict]:
    """
    Will find a list of the indices of the eclipses in the passed LightCurve.
    Also return a peak_properties dictionary (see scipy.stats.find_peaks)

    :lc: the LightCurve to parse.

    :width: the minimum sample width of any peak
    """
    # We'll use a Std Dev as the minimum prominence of any peaks to find.
    # Unless the data is very noisey, this should rule out all but eclipses.
    std_mag = np.std(lc["delta_mag"].data)
    return find_peaks(lc["delta_mag"], prominence=(std_mag, ), width=width)


def phase_fold_lc(lc: LightCurve,
                  primary_epoch: Time,
                  period: u.Quantity,
                  phase_pivot: float=0.75) -> FoldedLightCurve:
    """
    Perform a normalized phase fold on the passed LightCurve. By default, the
    primary_epoch (phase 0) will rotated to the 0.25 position giving returned 
    phase coverage of [-0.25, 0.75).

    :lc: the LightCurve to produce the fold from

    :primary_epoch: the reference to fold around

    :period: the period to fold on

    :phase_pivot: the pivot point above which to wrap phase round to < 0.
    """
    if phase_pivot is not None and not isinstance(phase_pivot, u.Quantity):
        phase_pivot = u.Quantity(phase_pivot)
    return lc.fold(period,
                   epoch_time=primary_epoch,
                   normalize_phase=True,
                   wrap_phase=phase_pivot)


def get_reduced_folded_lc(flc: FoldedLightCurve,
                          num_bins: int = 1024,
                          phase_pivot: float=0.75,
                          flc_rollover: int = 200) \
                                -> Tuple[u.Quantity, u.Quantity]:
    """
    A data reduction function which gets a reduced set of phase folded 
    delta magnitude data in equal size bins of the requested number. 
    
    The data is sourced by sampling the passed FoldedLightCurve.  In case this 
    does not extend over a complete phase, rows are copied over from opposite
    ends of the phase space data to extend the coverage.  The number of rows 
    copied is controlled by the flc_rollover argument.

    :flc: the source FoldedLightCurve

    :num_bins: the number of equally spaced rows to return

    :phase_pivot: the pivot point about which the fold phase was wrapped to < 0.

    :flc_rollover: the number of row to extend the ends of the source phases by

    returns a tuple with requested number or phases and delta magnitudes
    """
    source_phases = np.concatenate([
        flc.phase[-flc_rollover:] -1.,
        flc.phase,
        flc.phase[:flc_rollover] +1.
    ])

    source_delta_mags = np.concatenate([
        flc["delta_mag"][-flc_rollover:],
        flc["delta_mag"],
        flc["delta_mag"][:flc_rollover]
    ])

    # If there is a phase wrap then phases above the pivot will have been
    # wrapped around to <0. Work out what the expected minimum phase should be.
    expected_min_phase = 0 if not phase_pivot else phase_pivot - 1.
    if not isinstance(expected_min_phase, u.Quantity):
        expected_min_phase = u.Quantity(expected_min_phase)

    min_phase = np.max((expected_min_phase, source_phases.min()))
    interp = interp1d(source_phases, source_delta_mags, kind="linear")
    reduced_phases = np.linspace(min_phase, min_phase + 1., num_bins + 1)[:-1]
    reduced_mags = interp(reduced_phases)
    return (reduced_phases, reduced_mags)


def find_indices_of_segments(lc: LightCurve,
                             threshold: TimeDelta) \
                                -> Generator[Tuple[int, int], any, None]:
    """
    Finds the indices of contiguous segments in the passed LightCurve. These are
    subsets of the LC where the gaps between bins does not exceed the passed
    threshold. Gaps > threshold are treated as boundaries between segments.

    :lc: the source LightCurve to parse for gaps/segments.

    :threshold: the threshold gap time beyond which a segment break is triggered

    Returns an generator of segment (start, end) indices. If no gaps found this 
    will yield a single entry for the (first, last) indices in the LightCurve.
    """
    if not isinstance(threshold, TimeDelta):
        threshold = TimeDelta(threshold * u.d)

    # Much quicker if we use primatives - make sure we work in days
    threshold = threshold.to(u.d).value
    times = lc.time.value

    last_ix = len(lc) - 1
    segment_start_ix = 0
    for this_ix, previous_time in enumerate(times, start = 1):
        if this_ix > last_ix or times[this_ix] - previous_time > threshold:
            yield (segment_start_ix, this_ix - 1)
            segment_start_ix = this_ix
