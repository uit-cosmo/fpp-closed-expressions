""" Autocorrelation function and power spectral density (positive half-line) """

import numpy as np
import mpmath as mm


def acorr(T, td, l):
    """
    Returns the normalized autocorrelation of a shot noise process.
    Input:
        T:  ndarray, float. Time lag.
        td: float, pulse duration time.
        l:  float, pulse asymmetry parameter. Related to pulse rise time by tr = l * td and pulse fall time by tf = (1-l) * tf.
    Output:
        R: ndarray, float. Autocorrelation at time lag tau.
    """
    R = np.zeros(T.shape[0], dtype="float64")
    assert td > 0.0
    assert l >= 0.0
    assert l <= 1.0

    eps = 1e-8

    td = mm.mpf(td)
    l = mm.mpf(l)
    inv_td = mm.mpf(1.0 / td)

    if np.abs(l) < eps or np.abs(l - 1.0) < eps:
        fun = lambda t, td, l: mm.exp(-t * inv_td)

    elif np.abs(l - 0.5) < eps:
        fun = lambda t, td, l: (1.0 + 2.0 * t * inv_td) * mm.exp(-2.0 * t * inv_td)

    else:
        fun = lambda t, td, l: (
            (1.0 - l) * mm.exp(-t * inv_td / (1.0 - l)) - l * mm.exp(-t * inv_td / l)
        ) / (1.0 - 2.0 * l)

    for i in range(len(T)):
        R[i] = fun(T[i], td, l)

    return R


def psd(omega, td, l):
    """
    Returns the normalized power spectral density of a shot noise process,
    given by
    PSD(omega) = 2.0 * taud / [(1 + (1 - l)^2 omega^2 taud^2) (1 + l^2 omega^2 taud^2)]
    Input:
        omega...: ndarray, float: Angular frequency
        td......: float, pulse duration time
        l.......: float, pulse asymmetry parameter.
                  Related to pulse rise time by
                  tr = l*td and pulse fall time by tf = (1-l)*tf.
    Output:
        psd.....: ndarray, float: Power spectral density
    """
    psd = np.zeros(omega.shape[0])
    assert td > 0
    assert l >= 0
    assert l <= 1
    # td = mm.mpf(td)
    # l = mm.mpf(l)
    if l in [0, 1]:
        # fun = lambda o, td, l: 4 * td / (1 + (td * o)**2)
        psd = 4.0 * td / (1.0 + (td * omega) * (td * omega))
    elif l == 0.5:
        # fun = lambda o, td, l: 64 * td / (4 + (td * o)**2)**2
        psd = 64.0 * td / (4.0 + (td * omega) * (td * omega)) ** 2.0
    else:
        # fun = lambda o, td, l: 4 * td / \
        #    ((1 + ((1 - l) * td * o)**2) * (1 + (l * td * o)**2))
        psd = (
            4.0
            * td
            / (
                (1.0 + ((1.0 - l) * td * omega) * (1.0 - l) * td * omega)
                * (1.0 + (l * td * omega) * (l * td * omega))
            )
        )

    # for i in range(len(O)):
    #    S[i] = fun(O[i], td, l)
    return psd
