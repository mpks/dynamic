import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
from scitbx.array_family import flex


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


def scale_intensities(I1, I2, keep_first=True, return_r1=False):

    I1 = np.array(I1)
    I2 = np.array(I2)

    I1[I1 < 0] = 0
    I2[I2 < 0] = 0

    F1 = np.sqrt(I1)
    F2 = np.sqrt(I2)

    if keep_first:
        scale, r1 = find_best_scale(F1, F2)
        F2 = scale * F2
        I2 = F2**2
    else:
        scale, r1 = find_best_scale(F2, F1)
        F1 = scale * F1
        I1 = F1**2

    if return_r1:
        return I1, I2, r1
    else:
        return I1, I2


def find_best_scale(Fo, Fc):
    """
    Find scaling factor that minimizes R1 between Fo and Fc
    (finds best k for which Fo is approx Fc * k)
    """
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


def s1_distance(spot_a, spot_b):

    # The spots need to be at the same, or at least at the neigboring images

    assert abs(spot_a.z - spot_b.z) <= 1

    dx = (spot_a.s1_x - spot_b.s1_x)**2
    dy = (spot_a.s1_y - spot_b.s1_y)**2
    dz = (spot_a.s1_z - spot_b.s1_z)**2

    return np.sqrt(dx + dy + dz)


def compute_s1(x, y, experiment):
    """Set observed s1 vectors for reflections if required, return the number
    of reflections that have been set."""

    detector = experiment.detector
    panel = detector[0]
    beam = experiment.beam
    xy = flex.vec2_double(flex.double([x]), flex.double([y]))
    s1 = panel.get_lab_coord(xy)
    s1 = s1 / s1.norms() * (1 / beam.get_wavelength())

    s1 = np.array(s1)

    return s1[0]


def scale_intensities_wr(I1, I2, sigma1, sigma2,
                         keep_first=True, return_wr=False):
    """
    Scale intensities minimizing wR factor using sigma weights.

    wR = sqrt( sum(w * (Fo - k*Fc)^2) / sum(w * Fo^2) )
    where w = 1/sigma^2

    Parameters
    ----------
    I1, I2 : array-like, observed and calculated intensities
    sigma1, sigma2 : array-like, sigmas corresponding to I1 and I2
    keep_first : if True, scale I2 to I1; if False, scale I1 to I2
    """
    I1 = np.array(I1, dtype=float)
    I2 = np.array(I2, dtype=float)
    sigma1 = np.array(sigma1, dtype=float)
    sigma2 = np.array(sigma2, dtype=float)

    I1[I1 < 0] = 0
    I2[I2 < 0] = 0

    if keep_first:
        scale, wr = find_best_scale_wr(I1, I2, sigma1)
        I2 = scale * I2
    else:
        scale, wr = find_best_scale_wr(I2, I1, sigma2)
        I1 = scale * I1

    if return_wr:
        return I1, I2, wr
    else:
        return I1, I2


def find_best_scale_wr(Iobs, Icalc, sigma):
    """
    Find scaling factor k minimizing wR between Iobs and k*Icalc.

    wR = sqrt( sum(w*(Iobs - k*Icalc)^2) / sum(w*Iobs^2) )
    where w = 1/sigma^2

    Note: this has an analytical solution, so no optimizer needed.
    k = sum(w * Iobs * Icalc) / sum(w * Icalc^2)
    """
    Iobs = np.asarray(Iobs, dtype=float)
    Icalc = np.asarray(Icalc, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Avoid division by zero
    valid = sigma > 0
    Iobs, Icalc, sigma = Iobs[valid], Icalc[valid], sigma[valid]

    w = 1.0 / sigma**2

    # Analytical optimal scale
    k_best = np.sum(w * Iobs * Icalc) / np.sum(w * Icalc**2)

    wr_best = wr_factor(k_best, Iobs, Icalc, w)
    print(f"wR scaling - wR = {wr_best:.4f}  scale = {k_best:.4f}")
    return k_best, wr_best


def wr_factor(k, Iobs, Icalc, w):
    """
    wR = sqrt( sum(w*(Iobs - k*Icalc)^2) / sum(w*Iobs^2) )
    """
    numerator = np.sum(w * (Iobs - k * Icalc)**2)
    denominator = np.sum(w * Iobs**2)
    return np.sqrt(numerator / denominator)
