"""
helpers.py

Script containing helper funcions for plotting and command line.

Authors: John E. Parker (2026)
"""

# import python libraries
import subprocess
import string
import numpy as np


def run_cmd(cmd, print_out=True):
    """
    Runs command on the command line.

    Parameters
    ----------
    cmd : string
        command to be provided to command line

    print_out : boolean
        (optional, True) prints cmd if True
    """
    if print_out:
        print(cmd)
    subprocess.run(cmd)


def makeNice(axes, labelsize=6, lw=1, width=0):
    """
    Helper script to clean up plots. Removes right and top spines. Adjusts
    labelsizes, linewidths, and tickmarks of spines.

    Parameters
    ----------
    axes : list or single axes plot
        axes to clean up

    labelsize : int
        (optional, 6) size of tickmarks

    lw : int
        (optional, 1) linewidth of spines

    width : int
        (optional, 0) width of tickmarks
    """
    if type(axes) == list:
        for ax in axes:
            for i in ["left", "right", "top", "bottom"]:
                if i != "left" and i != "bottom":
                    ax.spines[i].set_visible(False)
                    ax.tick_params("both", width=0, labelsize=labelsize)
                else:
                    ax.spines[i].set_linewidth(lw)
                    ax.tick_params("both", width=width, labelsize=labelsize)
    else:
        for i in ["left", "right", "top", "bottom"]:
            if i != "left" and i != "bottom":
                axes.spines[i].set_visible(False)
                axes.tick_params("both", width=0, labelsize=labelsize)
            else:
                axes.spines[i].set_linewidth(lw)
                axes.tick_params("both", width=width, labelsize=labelsize)


def add_fig_labels(axes, fontsize=10):
    """
    Helper script to add panel label to axes.

    Parameters
    ----------
    axes : list or single axes plot
        axes to clean up

    fontsize : int
        (optional, 10) size of subplot labels

    """
    labels = string.ascii_uppercase
    for i in range(len(axes)):
        axes[i].text(
            -0.15,
            1.05,
            labels[i],
            fontsize=fontsize,
            transform=axes[i].transAxes,
            fontweight="bold",
            color="gray",
        )


def match_axis(axes, type="both"):
    """
    Helper script to match axes for multiple plots.

    Parameters
    ----------
    axes : list or single axes plot
        axes to match axis limits

    type : string
        (optional, "both") "x" or "y" to match respective axis, otherwise both
    """
    if type == "x":
        min = np.min([ax.get_xlim()[0] for ax in axes])
        max = np.max([ax.get_xlim()[1] for ax in axes])
        for ax in axes:
            ax.set_xlim([min, max])
    elif type == "y":
        min = np.min([ax.get_ylim()[0] for ax in axes])
        max = np.max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_ylim([min, max])
    else:
        match_axis(axes, type="x")
        match_axis(axes, type="y")
