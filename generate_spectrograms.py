"""
Echo spectrogram generation from MSM frequency-domain scattering data.

Reconstructs time-domain echoes via inverse Fourier transform (Section 2.2.1)
and computes STFT spectrograms (Section 2.4.1) for CNN classification.

Usage:
    python generate_spectrograms.py --input data/csr_random_results28.mat \
                                    --output spectrograms/csr_random.npz
"""

import argparse
import numpy as np
import hdf5storage
from scipy.interpolate import interp1d
from scipy.signal import spectrogram as scipy_spectrogram
from nlfm_pulse import generate_nlfm_pulse


# ---------- Default signal parameters (Table in Section 2.3.3) ----------
FMAX = 80000       # f_center(70kHz) + BW(20kHz)/2
BW = 2e4            # 20 kHz
N_SWEEP = 20        # sweep-shaping parameter
DUR = 2e-3          # 2 ms
FS = 1e6            # 1 MHz

# MSM frequency grid
F_SCAT = np.linspace(58480, 81500, 1152)

# Time-domain reconstruction
T_TOTAL = 0.05      # 50 ms reconstruction window

# STFT parameters (Section 2.4.1)
STFT_WIN_LEN = 2000   # 2 ms Hamming window
STFT_OVERLAP = 1400   # 70% overlap
STFT_NFFT = 5000      # -> Δf = 200 Hz

# Frequency band for extraction
FREQ_LOW = 58000
FREQ_HIGH = 82000


def compute_pulse_spectrum(fmax, bw, n, dur, fs, f_scat):
    """Compute NLFM pulse spectrum on the MSM frequency grid."""
    _, _, y = generate_nlfm_pulse(fmax, bw, n, dur, fs)
    df = f_scat[1] - f_scat[0]
    nfft = int(round(fs / df))
    Y = np.fft.fft(y, nfft)
    f_bins = np.arange(nfft) * fs / nfft
    return interp1d(f_bins, Y, kind='linear',
                    bounds_error=False, fill_value=0)(f_scat)


def reconstruct_echo(S_pulse, G_freq, f_scat, fs, T):
    """
    Reconstruct time-domain echo: P(w) = S(w)*G(w), p(t) = IFFT{P(w)}.
    """
    nfft = int(round(T * fs))
    f_full = np.arange(nfft) * fs / nfft
    half = nfft // 2 + 1

    Y = S_pulse * G_freq
    Y_full = np.zeros(nfft, dtype=complex)
    Y_full[:half] = interp1d(f_scat, Y, kind='linear',
                             bounds_error=False, fill_value=0)(f_full[:half])
    if nfft % 2 == 0:
        Y_full[half:] = np.conj(Y_full[half-2:0:-1])
    else:
        Y_full[half:] = np.conj(Y_full[half-1:0:-1])

    return np.fft.ifft(Y_full)[::-1].real


def echo_to_spectrogram(signal, fs):
    """Compute dB spectrogram and extract the target frequency band."""
    window = np.hamming(STFT_WIN_LEN)
    f, t, Sxx = scipy_spectrogram(signal, fs, window=window,
                                   noverlap=STFT_OVERLAP, nfft=STFT_NFFT,
                                   scaling='density', mode='magnitude')
    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    mask = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
    return Sxx_db[mask, :], f[mask], t


def main(input_path, output_path):
    data = hdf5storage.loadmat(input_path)
    p_scat_all = data['p_scat_all'].flatten()
    n_samples = len(p_scat_all)

    S_pulse = compute_pulse_spectrum(FMAX, BW, N_SWEEP, DUR, FS, F_SCAT)

    # Get output dimensions
    sig0 = reconstruct_echo(S_pulse, p_scat_all[0][0, :], F_SCAT, FS, T_TOTAL)
    spec0, _, _ = echo_to_spectrogram(sig0, FS)
    n_freq, n_time = spec0.shape

    spectrograms = np.zeros((n_samples, 2, n_freq, n_time))

    for i in range(n_samples):
        p_scat = p_scat_all[i]  # (2, 1152): two receiver channels
        for ch in range(2):
            sig = reconstruct_echo(S_pulse, p_scat[ch, :], F_SCAT, FS, T_TOTAL)
            spec, _, _ = echo_to_spectrogram(sig, FS)
            spectrograms[i, ch] = spec
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_samples}")

    np.savez(output_path, spectrograms=spectrograms)
    print(f"Saved {n_samples} samples to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.input, args.output)
