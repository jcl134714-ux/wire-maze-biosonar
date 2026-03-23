"""
NLFM pulse generation.
See Supporting Information Section S2 for the mathematical formulation.
"""

import numpy as np


def generate_nlfm_pulse(fmax, bw, n, dur, fs):
    """
    Generate a Hanning-windowed NLFM pulse.

    Parameters
    ----------
    fmax : float
        Maximum frequency (Hz), i.e., f_center + bw/2.
    bw : float
        Bandwidth (Hz).
    n : float
        Sweep-shaping parameter controlling nonlinearity of the frequency sweep.
    dur : float
        Pulse duration (s).
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    t : ndarray
        Time vector (s).
    f_inst : ndarray
        Instantaneous frequency (Hz).
    y : ndarray
        Pulse waveform.
    """
    dt = 1 / fs
    N = int(round(dur * fs))
    t_norm = np.linspace(0, 1, N)

    k_param = 1 - np.exp(np.log(bw) / n)
    f_inst = fmax - (t_norm - k_param) ** n
    t = t_norm * dur
    phi = 2 * np.pi * np.cumsum(f_inst) * dt

    win = np.hanning(N)
    y = np.cos(phi) * win
    return t, f_inst, y
