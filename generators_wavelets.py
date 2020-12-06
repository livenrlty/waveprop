import numpy as np
import pywt
from scipy.interpolate import interp1d

def cut_first_zeros(arr, tol):
    # function for generate_wavelets
    for i in range(len(arr)):
        if abs(arr[i]) > tol:
            break
    return arr[i:]

def cut_last_zeros(arr, tol):
    # function for generate_wavelets
    for i in range(len(arr)):
        if abs(arr[len(arr) - 1 - i]) > tol:
            break
    return arr[:len(arr) - 1 - i]

def transform_wav(arr, tol, nt):
    # function for generate_wavelets
    arr = cut_first_zeros(arr, tol)
    arr = cut_last_zeros(arr, tol)

    support_len = int(nt*(np.random.uniform()*0.2 + 0.7))

    interp_arr = interp1d(np.arange(arr.shape[0]), arr, kind='nearest')
    interp_arr = interp_arr(np.arange(support_len)*arr.shape[0]/support_len)
    wav = np.zeros(nt)
    wav[:interp_arr.shape[0]] = interp_arr
    return wav

def generate_wavelets(nt, wav_name):
    """
    Generate standard wavelets for impulsive source
    Input:
    nt - number of timesteps.
    wav_name - name of standard wavelet.
    Output:
    tuple with two np.arrays with values of size (nt,).
    """
    wav = pywt.Wavelet(wav_name)
    (phi, psi, x) = wav.wavefun()

    tol = 1e-4 # tolerance for cutting of first zeros
    phi = transform_wav(phi, tol, nt)
    psi = transform_wav(psi, tol, nt)

    return phi, psi