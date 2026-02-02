import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


def hypo(x, c, a):
    return a * np.sqrt(x*x + c)


def double_exp(x, a, b, c, d):
    return a*np.exp(b*x) + c*np.exp(d*x)


def fit_exp(x, y):

    popt, pcov = curve_fit(double_exp, x, y, p0=[0.02, 1/600., -0.05, 1./600.],
                           bounds=([-np.inf, -np.inf, -np.inf, -np.inf],
                                   [np.inf, np.inf, np.inf, np.inf]),
                           maxfev=100000)

    a_fit = popt[0]
    b_fit = popt[1]
    c_fit = popt[2]
    d_fit = popt[3]
    a_err = np.sqrt(np.diag(pcov))[0]
    b_err = np.sqrt(np.diag(pcov))[1]
    c_err = np.sqrt(np.diag(pcov))[2]
    d_err = np.sqrt(np.diag(pcov))[3]

    print(f"Fitted a = {a_fit:.4f} ± {a_err:.4f}")
    print(f"Fitted b = {b_fit:.4f} ± {b_err:.4f}")
    print(f"Fitted c = {c_fit:.4f} ± {c_err:.4f}")
    print(f"Fitted d = {d_fit:.4f} ± {d_err:.4f}")

    def dexp(x, a=a_fit, b=b_fit, c=c_fit, d=d_fit):
        return a * np.exp(x * b) + c * np.exp(x * d)

    params = [a_fit, b_fit, c_fit, d_fit]
    return params, dexp


def fit_hypo(fc, fobs):

    popt, pcov = curve_fit(hypo, fc, fobs, p0=[0.1, 0.1],
                           bounds=([0, -np.inf],
                                   [2000, np.inf]))

    c_fit = popt[0]
    a_fit = popt[1]
    c_err = np.sqrt(np.diag(pcov))[0]
    a_err = np.sqrt(np.diag(pcov))[1]

    print(f"Fitted c = {c_fit:.4f} ± {c_err:.4f}")
    print(f"Fitted a = {a_fit:.4f} ± {a_err:.4f}")

    def hyperbola(x, c=c_fit, a=a_fit):
        return a * np.sqrt(x*x + c)

    return a_fit, hyperbola


def find_best_scale(Fo, Fc):
    """Find scaling factor that minimizes R1 between Fo and Fc"""
    Fo = np.asarray(Fo)
    Fc = np.asarray(Fc)

    up_bound = 5 * Fo.mean() / Fc.mean()

    # Reasonable search range for scale
    result = minimize_scalar(r1_factor, bounds=(0, up_bound),
                             args=(Fo, Fc), method='bounded')

    k_best = result.x
    r1_best = result.fun
    print(f"R1 scaling - R1 = {r1_best:.3f}  scale = {k_best:.3f}")
    return k_best, r1_best


def r1_factor(k, Fo, Fc):
    down = np.sum(np.abs(Fo))
    up = np.abs(Fo) - k * np.abs(Fc)
    return np.sum(np.abs(up)) / down
