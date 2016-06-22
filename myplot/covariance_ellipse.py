# -*- coding: utf-8 -*-
"""
It sounds like you just want the 2-sigma ellipse of the scatter of points?

If so, consider something like this (From some code for a paper here: 
   https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
    
        cov
            The 2x2 covariance matrix to base the ellipse on.
        pos
            The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd
            The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax
            The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip