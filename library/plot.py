"""
Functions for plotting to matplotlib Axes.
"""
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import astropy.units as u
from lightkurve import LightCurve, FoldedLightCurve

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

def plot_light_curve_on_axes(lc: LightCurve,
                             ax: Axes = None,
                             title: str = None,
                             column: str = "delta_mag",
                             label: str = None,
                             zorder: int = 0) -> Axes:
    """
    Plots the passed LightCurve time and delta_mag data on a set of Axes.
    The Axes may be passed in if plotting to an existing figure, otherwise
    a new (8, 4) figure and single axes will be created.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)

    # Will set up useable ticks and tick labels.
    # And a default ylabel. The label specified here is for a legend.
    lc.scatter(column=column, ax=ax, s=2., label=label, zorder=zorder)

    if lc[column].unit == u.mag:
        ax.invert_yaxis()
        ax.set_ylabel(
            f"{'Relative m' if 'delta' in column else 'M'}agnitude [mag]")

    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in",
                   bottom=True, top=True, left=True, right=True)
    ax.set_title(title)
    return ax


def plot_folded_light_curve_on_axes(flc: FoldedLightCurve,
                                    ax: Axes = None,
                                    title: str = None,
                                    column: str = "delta_mag",
                                    label: str = None,
                                    zorder: int = 0) -> Axes:
    """
    Plots the passed FoldedLightCurve time and delta_mag data on a set of Axes.
    The Axes may be passed in if plotting to an existing figure, otherwise
    a new (8, 4) figure and single axes will be created.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)

    # Will set up useable ticks and tick labels.
    # And a default ylabel. The label specified here is for a legend.
    flc.scatter(
        column=column, ax=ax, s=4, alpha=.25, label=label, zorder=zorder)

    if flc[column].unit == u.mag:
        ax.invert_yaxis()
        ax.set_ylabel(
            f"{'Relative m' if 'delta' in column else 'M'}agnitude [mag]")

    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in",
                   bottom=True, top=True, left=True, right=True)
    ax.set_title(title)
    return ax
