import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = '/Users/Yogesh/Downloads/music_speech/music_wav/beatles.wav'

# waveform
signal, sr = librosa.load(file, sr=22050)  # sr * T
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# time domain > frequency domain using FFT
# fft = np.fft.fft(signal)
# magnitude = np.abs(fft)
# frequency = np.linspace(0, sr, len(magnitude))
# left_frequency = frequency[:int(len(frequency)/2)]
# left_magnitude = magnitude[:int(len(magnitude)/2)]
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()

# how this frequencies contributing to the original sound over a time
# to know this perform STFT > short time fourier transform and plot STFT spectrogram
n_fft = 2048  # no of samples (window) to perform single FFT
hop_length = 512  # no of samples (how much to slide a window) for FFT
# stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)
# spectrogram = np.abs(stft)
# # log spectrogram
# log_spectrogram = librosa.amplitude_to_db(spectrogram)
# librosa.display.specshow(data=log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar()
# plt.show()

# extract mfcc and plot them to see how they are evolving
mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length)
plt.xlabel('Time')
plt.ylabel('mfcc')
plt.colorbar()
plt.show()