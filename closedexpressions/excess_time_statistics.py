"""
Excess time statisitics
In all cases, the signal z should have been normalized as (z-<z>)/z_rms
"""

import numpy as np
import mpmath as mm
import warnings


def eT(X, g):
    """
    Returns the fraction of time above threshold for the normalized shot noise process X.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
    Output:
        F: The fraction of time above threshold. The total time is T*F.
    """
    F = np.ones(len(X))
    assert g > 0
    g = mm.mpf(g)
    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = mm.gammainc(g, a=np.sqrt(g) * X[i] + g, regularized=True)
    return F


def eX(X, g, l):
    """
    Returns the rate of upwards level crossings above threshold for the normalized shot noise process X.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
        l: pulse asymmetry parameter, float.
    Output:
        F: The rate of upward crossings above threshold. The total number of crossings is td*F/T.
    """
    assert g > 0
    assert l >= 0
    assert l <= 1
    l = mm.mpf(l)
    g = mm.mpf(g)
    F = np.zeros(len(X))

    def eXtmp(x, g, l):
        if (l > 0) & (l < 1):
            return (
                (
                    l ** (g * l - 1)
                    * (1 - l) ** (g * (1 - l) - 1)
                    * g ** (g / 2 - 1)
                    / (mm.gamma(g * l) * mm.gamma(g * (1 - l)))
                )
                * (x + np.sqrt(g)) ** g
                * mm.exp(-np.sqrt(g) * x - g)
            )
        else:
            return (
                g ** (g / 2)
                * (x + np.sqrt(g)) ** g
                * mm.exp(-np.sqrt(g) * x - g)
                / mm.gamma(g)
            )

    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = eXtmp(X[i], g, l)
    return F


def eX_l0(X, g):
    """
    Returns the rate of upwards level crossings above threshold for the normalized shot noise process X with a one sided pulse shape (l=0).
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
    Output:
        F: The rate of upward crossings above threshold. The total number of crossings is td*F/T.
    """
    warnings.warn("The functionality of eX_l0 has been added to eX.")
    assert g > 0
    g = mm.mpf(g)
    F = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = (
                g ** (g / 2)
                * (X[i] + np.sqrt(g)) ** g
                * mm.exp(-np.sqrt(g) * X[i] - g)
                / mm.gamma(g)
            )
    return F


def avT(X, g, l):
    """
    Returns the normalized average time above threshold for the normalized shot noise process X.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
        l: pulse asymmetry parameter, float.
    Output:
        F: The normalized average time above threshold. The unnormalized version is F/td.
    """
    assert g > 0
    assert l >= 0
    assert l <= 1
    l = mm.mpf(l)
    g = mm.mpf(g)
    F = np.zeros(len(X))

    def avTtmp(x, g, l):
        if (l > 0) & (l < 1):
            return (
                (
                    mm.gamma(g * l)
                    * mm.gamma(g * (1 - l))
                    * l ** (1 - g * l)
                    * (1 - l) ** (1 - g * (1 - l))
                    * g ** (1 - g / 2)
                )
                * mm.gammainc(g, a=np.sqrt(g) * x + g, regularized=True)
                * (x + np.sqrt(g)) ** (-g)
                * mm.exp(np.sqrt(g) * x + g)
            )
        else:
            return (
                (mm.gamma(g) * g ** (-g / 2))
                * mm.gammainc(g, a=np.sqrt(g) * X[i] + g, regularized=True)
                * (x + np.sqrt(g)) ** (-g)
                * mm.exp(np.sqrt(g) * x + g)
            )

    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = avTtmp(X[i], g, l)
    return F


def avT_l0(X, g):
    """
    Returns the normalized average time above threshold for the normalized shot noise process X with pulse asymmetry parameter l=0.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
    Output:
        F: The normalized average time above threshold. The unnormalized version is F/td.
    """
    warnings.warn("The functionality of avT_l0 has been added to avT.")
    assert g > 0
    g = mm.mpf(g)
    F = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = (
                (mm.gamma(g) * g ** (-g / 2))
                * mm.gammainc(g, a=np.sqrt(g) * X[i] + g, regularized=True)
                * (X[i] + np.sqrt(g)) ** (-g)
                * mm.exp(np.sqrt(g) * X[i] + g)
            )
    return F
