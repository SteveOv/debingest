from typing import Union, Tuple, List
import numpy as np
from astropy.time import Time
from astropy.units import Quantity

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

    print(f"\tRunning {iterations} iterations, fitting data with residual "
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
            residuals = ydata.value - fit_ydata
            fit_mask &= (np.abs(residuals) <= (np.std(residuals)*res_sigma_clip))
        else:
            if reset_const_coeff:
                print(f"...final 'const' coeff reset to zero (on request)...")
                coeffs[0] = 0
            poly_func = np.polynomial.Polynomial(coeffs)
            fit_ydata = poly_func(time_values) * ydata.unit
            print(f"\tFit complete; y={poly_func} (sigma(y)={np.std(fit_ydata)})")

    return (fit_ydata, coeffs) if include_coeffs else fit_ydata