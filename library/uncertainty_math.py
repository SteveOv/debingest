""" Some basic mathematical functions for values with uncertainties. """
import numpy as np

# pylint: disable=invalid-name

def add(x, dx, y, dy):
    """
    Calculate: z ± dz = (x ± dx) + (y ± dy)
    """
    z = np.add(x, y)
    dz = uncertainty_add_or_subtract(dx, dy)
    return z, dz


def subtract(x, dx, y, dy):
    """
    Calculate: z ± dz = (x ± dx) - (y ± dx)
    """
    z = np.subtract(x, y)
    dz = uncertainty_add_or_subtract(dx, dy)
    return z, dz


def multiply(x, dx, y, dy):
    """
    Calculate: z ± dz = (x ± dx) * (y ± dy)
    """
    z = np.multiply(x, y)
    dz = uncertainty_multiply_or_divide(z, x, dx, y, dy)
    return z, dz


def divide(x, dx, y, dy):
    """
    Calculate: z ± dz = (x ± dx) / (y ± dy)
    """
    z = np.divide(x, y)
    dz = uncertainty_multiply_or_divide(z, x, dx, y, dy)

    return z, dz


def power(x, dx, y, dy):
    """
    Calculate: z ± dz = (x ± dx)^(y ± dy)
    """
    z = np.power(x, y)
    dz_of_dx = np.multiply(np.multiply(y, z), np.divide(dx, x))
    dz_of_dy = np.multiply(np.multiply(dy, z), np.log10(np.abs(x)))
    dz = np.sqrt(np.add(np.power(dz_of_dx, 2), np.power(dz_of_dy, 2)))
    return z, dz


def ln(x, dx=0):
    """
    Calculate: z ± dz = ln(x ± dx)
    """
    z = np.log(x)
    dz = np.divide(dx, x)
    return z, dz


def log10(x, dx=0):
    """
    Calculate: z ± dz = log10(x ± dx)
    """
    z = np.log10(x)
    dz = np.multiply(np.divide(dx, x), np.divide(1, np.log(10)))
    return z, dz


def cos(x, dx):
    """
    Calculate: z ± dz = cos(x ± dx)
    """
    z = np.cos(x)
    # dz = sqrt((d/dx(cos(x))*dx)^2) where d/dx(cos(x)) = -sin(x)
    ddx = -np.sin(x)
    dz = np.sqrt(np.power(np.multiply(ddx, dx), 2))
    return z, dz


def sin(x, dx):
    """
    Calculate: z ± dz = sin(x ± dx)
    """
    z = np.sin(x)
    # dz = sqrt((d/dx(sin(x))*dx)^2) where d/dx(sin(x)) = cos(x)
    ddx = np.cos(x)
    dz = np.sqrt(np.power(np.multiply(ddx, dx), 2))
    return z, dz


def tan(x, dx):
    """
    Calculate: z ± dz = tan(x ± dx)
    """
    z = np.tan(x)
    # dz = sqrt((d/dx(tan(x))*dx)^2) where d/dx(tan(x)) = sec^2(x)
    ddx = np.power(np.sec(x), 2)
    dz = np.sqrt(np.power(np.multiply(ddx,  dx), 2))
    return z, dz


def arccos(x, dx):
    """
    Calculate: z ± dz = arccos(x ± dx)
    """
    z = np.arccos(x)
    # dz = sqrt((d/dx(arccos(x))*dx)^2) where d/dx(arccos(x)) = -1/\sqrt(1-x^2)
    ddx = np.divide(-1, np.sqrt(np.subtract(1, np.power(x, 2))))
    dz = np.sqrt(np.power(np.multiply(ddx, dx), 2))
    return z, dz


def arcsin(x, dx):
    """
    Calculate: z ± dz = arcsin(x ± dx)
    """
    z = np.arcsin(x)
    # dz = sqrt((d/dx(arcsin(x))*dx)^2) where d/dx(arcsin(x)) = 1/\sqrt(1-x^2)
    ddx = np.divide(1, np.sqrt(np.subtract(1, np.power(x, 2))))
    dz = np.sqrt(np.power(np.multiply(ddx, dx), 2))
    return z, dz


def arctan(x, dx):
    """
    Calculate: z ± dz = arctan(x ± dx)
    """
    z = np.arctan(x)
    # dz = sqrt((d/dx(arctan(x))*dx)^2) where d/dx(arcsin(x)) = 1/(1+x^2)
    ddx = np.divide(1, np.add(1, np.power(x, 2)))
    dz = np.sqrt(np.power(np.multiply(ddx, dx), 2))
    return z, dz


def rad2deg(x, dx):
    """
    Calculate: z ± dz = rad2deg(x ± dx)
    """
    coeff = np.divide(180, np.pi)
    return multiply(x, dx, coeff, 0)


def uncertainty_add_or_subtract(dx=0, dy=0):
    """
    Calculate the uncertainty associated with a sum or difference calc 
    based on the passed error values.
    """
    dz_of_dx = np.power(dx, 2)
    dz_of_dy = np.power(dy, 2)
    dz = np.sqrt(np.add(dz_of_dx, dz_of_dy))
    return dz


def uncertainty_multiply_or_divide(z, x, dx, y, dy):
    """
    Calculate the uncertainty associated with a multiplication or division
    based on the passed value and error values. z will be the value resulting
    from the initial multiplication or division of x and y.
    """
    dz_of_dx = np.power(np.divide(dx, x), 2)
    dz_of_dy = np.power(np.divide(dy, y), 2)
    dz = np.multiply(np.sqrt(np.add(dz_of_dx, dz_of_dy)), z)
    return dz
