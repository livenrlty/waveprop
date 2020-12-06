import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess  # Python library with AR processes


def generate_source_coordinate(nx, nz, nabs):
    """
    Generate random source coordinate (NOT in immerse boundaries)
    Input:
    nx, nz - number of grid nodes on uniform spatial grid
    nabs - numbers of immerse boundary grid nodes
    Output:
    (srcx, srcz) - tuple with source coordinates (int, int).
    """
    srcx = nabs + np.random.randint(0, high=nx - 2 * nabs, dtype=int)
    srcz = nabs + np.random.randint(0, high=nz - 2 * nabs, dtype=int)

    return srcx, srcz


def generate_stoch_gauss(nt, r_corr):
    """
    Generate stoch Gauss process with Gauss filtering (5*sigma).
    Input:
    nt - number of timesteps.
    r_corr - correlation radius.
    10*r_corr+1 should be < than nt!
    Output:
    np.array with values of size (nt,).
    """
    signal = np.random.randn(nt)
    x = np.arange(-5 * r_corr, 5 * r_corr + 1)
    gauss_filter = np.exp(-x ** 2 / (2 * (r_corr ** 2)))
    filtered = np.convolve(signal, gauss_filter, 'same')

    return filtered / np.std(filtered)


def generate_stoch_ar1(nt, a1):
    """
    Generate AR1 process for strong noisy source
    Input:
    nt - number of timesteps.
    a1 - interior parameter.
    Output:
    np.array with values of size (nt,).
    """
    ar_par = np.array([1.0, a1])
    ma = np.array([1.0])
    simulated_ar1 = ArmaProcess(ar_par, ma).generate_sample(nsample=nt)

    return simulated_ar1 / np.std(simulated_ar1)
