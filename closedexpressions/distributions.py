""" Distributions """

import numpy as np
import mpmath as mm


def shot_noise_dist(X, g, A, cdf=False):
    """
    Returns the pdf or cdf of a gamma distributed variable.
    Input:
        X: Variable values, 1d numpy array
        g: shape parameter
        A: scale parameter
        cdf: toggles pdf(default) or cdf.
    Output:
        F: The pdf or cdf of X.
    """
    F = np.zeros(len(X))
    if not cdf:
        f = lambda x, g, A: x ** (g - 1) * mm.exp(-x / A) / (mm.gamma(g) * A ** g)
    else:
        f = lambda x, g, A: mm.gammainc(g, a=x / A, regularized=True)
    assert g > 0
    assert A > 0
    for i in range(len(X)):
        if X[i] >= 0:
            F[i] = f(X[i], g, A)
    return F


def norm_shot_noise_dist(X, g, cdf=False):
    """
    Returns the pdf or cdf of a normalized gamma distributed variable.
    If x is gamma distributed, X=(x-<x>)/x_rms
    Input:
        X: Variable values, 1d numpy array
        g: shape parameter
        cdf: toggles pdf(default) or cdf.
    Output:
        F: The pdf or cdf of X.
    """
    F = np.zeros(len(X))
    assert g > 0
    if not cdf:
        f = (
            lambda x, g: g ** (g * 0.5)
            * (x + g ** (0.5)) ** (g - 1)
            * mm.exp(-(g ** (0.5)) * x - g)
            / mm.gamma(g)
        )
    else:
        f = lambda x, g: mm.gammainc(g, a=g ** (0.5) * x + g, regularized=True)
    for i in range(len(X)):
        if X[i] > -(g ** (1 / 2)):
            F[i] = f(X[i], g)
    return F


def noisy_shot_noise(X, g, e):
    """
    Returns the pdf of a normalized gamma distributed process with additive noise.
    Let z ~ Gamma(g,A), y ~ Normal(0,s^2), x = z+y.
    Input:
        X: The normalized variable X = (x-<x>)/x_rms, 1d numpy array
        g: shape parameter
        e: noise parameter, e=y_rms^2 / z_rms^2.
    Output:
        F: The pdf of X.
    """
    F = np.zeros(len(X))
    assert g > 0
    assert e > 0
    g = mm.mpf(g)
    e = mm.mpf(e)
    for i in range(len(X)):
        x = mm.mpf(X[i])

        F[i] = (
            (g * 0.5) ** (g * 0.5)
            * e ** (g * 0.5 - 1.0)
            * (1.0 + e) ** (0.5)
            * mm.exp(-(((1.0 + e) ** (0.5) * x + g ** (0.5)) ** (2.0)) / (2.0 * e))
            * (
                e ** (0.5)
                * mm.hyp1f1(
                    0.5 * g,
                    0.5,
                    ((1.0 + e) ** (0.5) * x + g ** (0.5) * (1.0 - e)) ** 2 / (2.0 * e),
                )
                / (2.0 ** (0.5) * mm.gamma((1.0 + g) * 0.5))
                + ((1.0 + e) ** (0.5) * x + g ** (0.5) * (1.0 - e))
                * mm.hyp1f1(
                    (1.0 + g) * 0.5,
                    1.5,
                    ((1.0 + e) ** (0.5) * x + g ** (0.5) * (1.0 - e)) ** 2 / (2.0 * e),
                )
                / mm.gamma(g * 0.5)
            )
        )
    return F


def norm_sym_dsn_dist(X, g):
    """
    Returns the normalized pdf of the derivative of a symmetric shot noise process, (td/2)*dS(t)/dt, lambda = 1/2.
    Input:
        X: The normalized variable X = (x-<x>)/x_rms, 1d numpy array
        g: shape parameter
    Output:
        F: The pdf of X.
    """
    F = np.zeros(len(X))
    assert g > 0
    g = mm.mpf(g)

    for i in range(len(X)):
        x = mm.mpf(np.abs(X[i]))
        F[i] = (
            mm.sqrt(2.0 * g / mm.pi)
            * 2.0 ** (-g / 2.0)
            * (mm.sqrt(g) * x) ** ((g - 1.0) / 2.0)
            * mm.besselk((1.0 - g) / 2.0, mm.sqrt(g) * x)
            / mm.gamma(g / 2.0)
        )

    return F


def joint_pdf_shot_noise(X, dX, g, A, l):
    """
    The joint PDF of X and the normalized derivative of X, dX.
    X and dX are assumed to be 1d arrays. The returned joint PDF has
    X on the first axis, and the returned meshgrids have 'ij'-indexing.
    len(X) = n, len(dX) = m, shape(J) = (n,m)
    """

    J = np.zeros([len(X), len(dX)])
    xX, dxX = np.meshgrid(X, dX, indexing="ij")
    pos = (xX + (1 - l) * dxX > 0) & (xX - l * dxX > 0)
    J[pos] = (
        l ** (g * l)
        * (1 - l) ** (g * (1 - l))
        * A ** (-g)
        / (mm.gamma(g * l) * mm.gamma(g * (1 - l)))
    )
    J[pos] *= (
        np.exp(-xX[pos] / A)
        * (xX[pos] + (1 - l) * dxX[pos]) ** (g * l - 1)
        * (xX[pos] - l * dxX[pos]) ** (g * (1 - l) - 1)
    )

    return J, xX, dxX


def shot_noise_laplace_A(X, g, a):
    """
    Returns the pdf of a shot noise process with laplace distributed amplitudes, A~Laplace(0,a)
    Input:
        X: Variable values, 1d numpy array.
        g: shape parameter
        a: scale parameter
    Output:
        F: The pdf
    """
    F = np.zeros(len(X))
    assert g > 0
    assert a > 0
    g = mm.mpf(g)
    a = mm.mpf(a)
    for i in range(len(X)):
        x = abs(X[i])
        F[i] = (
            (x / (2 * a)) ** ((g - 1) / 2)
            * mm.besselk((1 - g) / 2, x / a)
            / (a * np.sqrt(np.pi) * mm.gamma(g / 2))
        )
    return F


def shot_noise_laplace_A_norm(X, g):
    """
    Returns the normalized pdf of a shot noise process with laplace distributed amplitudes, A~Laplace(0,a)
    Input:
        X: Variable values, 1d numpy array.
        g: shape parameter
    Output:
        F: The pdf
    """
    F = np.zeros(len(X))
    assert g > 0
    g = mm.mpf(g)
    for i in range(len(X)):
        x = abs(X[i])
        F[i] = (
            (np.sqrt(g) * x / 2) ** ((g - 1) / 2)
            * mm.besselk((1 - g) / 2, np.sqrt(g) * x)
            * np.sqrt(g / np.pi)
            / mm.gamma(g / 2)
        )
    return F


def shotnoise_PDF_laplaceA(phi_rg, gamma_val, phi_rms):
    """
    Computes the PDF for a shotnoise process with Laplace distributed Amplitudes
    A ~ Laplace(0, a)
    See O.E. Garcia and A. Theodorsen, https://arxiv.org/abs/1702.00105
    phi_rms PDF(Phi) = sqrt(gamma / pi) / Gamma(gamma / 2) * (sqrt(gamma) |Phi| / Phi_rms) ^ ((gamma - 1) / 2) * Kv((gamma-1) / 2, sqrt(gamma) |Phi| / Phi_rms)
    Input:
    ======
    phi_rg...... ndarray, float: Domain of the PDF
    gamma_val... float, intermittency parameter
    phi_rms..... float, root mean squre value of the underlying sample
    Returns:
    =======
    res......... ndarray, float: The PDF on the domain
    """

    from scipy.special import gamma as gamma_func
    from scipy.special import kv

    t1 = np.sqrt(gamma_val / np.pi) / gamma_func(0.5 * gamma_val)
    t2 = (0.5 * np.sqrt(gamma_val) * np.abs(phi_rg) / phi_rms) ** (
        0.5 * (gamma_val - 1.0)
    )
    t3 = kv(0.5 * (gamma_val - 1.0), np.sqrt(gamma_val) * np.abs(phi_rg) / phi_rms)
    return t1 * t2 * t3
