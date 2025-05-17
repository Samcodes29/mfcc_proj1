#!/usr/bin/env python3
"""
Compute MFCC features for automatic speech recognition.
Filled in all sections marked “####”.
"""

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

def freq2mel(freq):
    return 2595 * np.log10(1 + freq / 700.0)

def mel2freq(mel):
    return 700 * (10**(mel / 2595.0) - 1)


# 1. Read waveform (from the audio_samples folder)
fs_hz, signal = wav.read('audio_samples/SA1.wav')
signal = signal.astype(float)
signal_length = len(signal)

# 2. Pre-emphasis
pre_emphasis = 0.97
emphasised = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# 3. Frame parameters (25 ms windows, 10 ms step)
frame_length_ms = 25
frame_step_ms   = 10
frame_length = int(round(frame_length_ms / 1000.0 * fs_hz))
frame_step   = int(round(frame_step_ms   / 1000.0 * fs_hz))
num_frames   = int(np.ceil(float(signal_length - frame_length) / frame_step)) + 1
print(f"number of frames is {num_frames}")

# 4. Pad signal to fit exactly num_frames * frame_step + frame_length
pad_signal_length = num_frames * frame_step + frame_length
pad_zeros = np.zeros(pad_signal_length - signal_length)
pad_signal = np.append(emphasised, pad_zeros)

# 5. Hamming window
win = np.hamming(frame_length)

# 6. Framing: slice into overlapping frames & apply window
frames = np.zeros((num_frames, frame_length))
for t in range(num_frames):
    start = t * frame_step
    frame = pad_signal[start : start + frame_length]
    frames[t, :] = frame * win

# 7. FFT & power spectrum
#    Next power of two for zero-padding
NFFT = 1 << (frame_length - 1).bit_length()
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

# 8. Mel-filterbank (26 filters between 0 and 8000 Hz)
low_freq_hz  = 0
high_freq_hz = min(fs_hz/2, 8000)
num_filters  = 26

low_mel  = freq2mel(low_freq_hz)
high_mel = freq2mel(high_freq_hz)
mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
hz_points  = mel2freq(mel_points)
bin_indices = np.floor((NFFT + 1) * hz_points / fs_hz).astype(int)

fbank = np.zeros((num_filters, NFFT//2 + 1))
for m in range(1, num_filters + 1):
    f_m_minus = bin_indices[m - 1]
    f_m       = bin_indices[m]
    f_m_plus  = bin_indices[m + 1]
    for k in range(f_m_minus, f_m):
        fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
    for k in range(f_m, f_m_plus):
        fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
filter_banks = 20 * np.log10(filter_banks)

# 9. MFCC via DCT (take 1–12; omit C0)
num_ceps = 12
mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps+1)]

# 10. Liftering
cep_lifter = 22
(nframes, ncoeff) = mfccs.shape
n = np.arange(ncoeff)
lift = 1 + (cep_lifter/2) * np.sin(np.pi * n / cep_lifter)
mfccs *= lift

# 11. Optional CMVN (mean-var normalize)
eps = 1e-8
mfccs -= (np.mean(mfccs, axis=0) + eps)
mfccs /= (np.std(mfccs, axis=0) + eps)

# 12. Plot results
time_axis = np.linspace(0, signal_length/fs_hz, num_frames)
plt.figure(figsize=(10, 6))

plt.subplot(3,1,1)
plt.title("Power Spectrum (log)")
plt.imshow(np.log(pow_frames.T + eps), origin='lower', aspect='auto',
           extent=(0, signal_length/fs_hz, 0, fs_hz/2))
plt.ylabel("Freq (Hz)")

plt.subplot(3,1,2)
plt.title("Mel-filterbank Energies (dB)")
plt.imshow(filter_banks.T, origin='lower', aspect='auto',
           extent=(0, signal_length/fs_hz, 0, num_filters))
plt.ylabel("Filter #")

plt.subplot(3,1,3)
plt.title("MFCCs")
plt.imshow(mfccs.T, origin='lower', aspect='auto',
           extent=(0, signal_length/fs_hz, 1, num_ceps))
plt.ylabel("Cepstral Coeff")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig('mfcc_outputs/day1_plots.png')
plt.show()
