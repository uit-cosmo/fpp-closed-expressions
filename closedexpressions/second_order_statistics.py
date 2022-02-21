""" Autocorrelation function and power spectral density (positive half-line) """

import numpy as np
import mpmath as mm
import warnings

warnings.filterwarnings("ignore")


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

    if l in [0, 1]:
        psd = 4.0 * td / (1.0 + (td * omega) * (td * omega))
    elif l == 0.5:
        psd = 64.0 * td / (4.0 + (td * omega) * (td * omega)) ** 2.0
    else:
        psd = (
            4.0
            * td
            / (
                (1.0 + ((1.0 - l) * td * omega) * (1.0 - l) * td * omega)
                * (1.0 + (l * td * omega) * (l * td * omega))
            )
        )

    return psd


def PSD_periodic_arrivals(omega, td, gamma, A_rms, A_mean, dt, norm=True):
    """Calculates the closed expression of the power spectral density  of a process
    of periodic Lorentzian pulses with duration time td = 1

    Args:
        omega: array[floats], frequency array
        gamma: float, intermittency parameters
        A_rms: float, rms value of amplitudes
        A_mean: float: mean amplotide
        dt: float, time step of time array correcponding to omega
        norm: bool, if True, expression for normalized process returned

    Returns:
        Power spectral density of process

    """
    I_2 = 1 / (2 * np.pi)
    first_term = td * gamma * A_rms ** 2 * I_2 * Lorentz_PSD(td * omega)
    tmp = np.zeros(omega.size)
    index = np.zeros(1000)
    for n in range(1, 1000):
        index = 2 * np.pi * n * gamma
        tmp = np.where(np.abs(omega - find_nearest(omega, index)) > 0.001, tmp, 1)

    PSD = (
        2 * np.pi * td * gamma ** 2 * A_mean ** 2 * I_2 * Lorentz_PSD(td * omega) * tmp
    )

    # imitate finite amplitude for delta functions in PSD finite
    # amplitudes occur due to the finite resolution of the time
    # series and the numerical method used to calculate the PSD
    PSD = 2 * (first_term + PSD / dt)

    if norm:
        Phi_rms = Phi_rms_periodic_lorentz(gamma, A_rms, A_mean)
        Phi_mean = Phi_mean_periodic_lorentz(gamma, A_mean)
        PSD[0] = PSD[0] - Phi_mean ** 2 * 2 * np.pi
        return PSD / Phi_rms ** 2
    return PSD


def autocorr_periodic_arrivals(t, gamma, A_mean, A_rms, norm=True):
    """Calculates the closed expression of the autocorrelation function of a process
    of periodic Lorentzian pulses with duration time td = 1

    Args:
        time: array[floats], time array
        gamma: float, intermittency parameters
        A_mean: float: mean amplotide
        A_rms: float, rms value of amplitudes
        norm: bool, if True, expression for normalized process returned

    Returns:
        autocorrelation function of process

    """
    I_2 = 1 / (2 * np.pi)
    central_peak = gamma * A_rms ** 2 * I_2 * Lorentz_pulse(t)
    oscillation = (
        gamma
        * np.pi
        * (
            1 / np.tanh(2 * np.pi * gamma - 1j * gamma * np.pi * t)
            + 1 / np.tanh(2 * np.pi * gamma + 1j * gamma * np.pi * t)
        )
    )
    R = central_peak + gamma * A_mean ** 2 * I_2 * oscillation.astype("float64")
    if norm:
        Phi_rms = Phi_rms_periodic_lorentz(gamma, A_rms, A_mean)
        Phi_mean = Phi_mean_periodic_lorentz(gamma, A_mean)
        return (R - Phi_mean ** 2) / Phi_rms ** 2
    return R


def Phi_rms_periodic_lorentz(gamma, A_rms, A_mean):
    """returns the rms values of a process of periodic Lorentz pulses
    with duration time td =1"""
    I_2 = 1 / (2 * np.pi)
    return (
        gamma * A_rms ** 2 * I_2
        + gamma
        * A_mean ** 2
        * I_2
        * (2 * np.pi * gamma * (1 / np.tanh(2 * np.pi * gamma)) - gamma / I_2)
    ) ** 0.5


def Phi_mean_periodic_lorentz(gamma, A_mean):
    """returns the mean values of a process of periodic Lorentz pulses
    with duration time td = 1"""
    I_1 = 1
    return gamma * A_mean * I_1


def Lorentz_pulse(theta):
    """spatial discretisation of Lorentz pulse with duration time td = 1 """
    return 4 * (4 + theta ** 2) ** (-1)


def Lorentz_PSD(theta):
    """PSD of a single Lorentz pulse with duration time td = 1"""
    return 2 * np.pi * np.exp(-2 * np.abs(theta))


def find_nearest(array, value):
    """returns array of parks in PSD"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

