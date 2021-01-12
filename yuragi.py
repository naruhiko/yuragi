import wave as wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.fft as sf

sample_wave_file = "./sounds/roji5.wav"

wav = wave.open(sample_wave_file)

print("sampling freq[Hz]: ", wav.getframerate())
print("sampling size[Byte]: ", wav.getsampwidth())
print("sampling num",wav.getnframes())
print("channels: ",wav.getnchannels())

data = wav.readframes(wav.getnframes())
data = np.frombuffer(data, dtype=np.int16)

wav.close()

data = data/np.iinfo(np.int16).max

x = np.array(range(wav.getnframes()))/wav.getframerate()

plt.figure(figsize=(10, 4))
plt.xlabel("Time[sec]")
plt.ylabel("Value[-1,1]")

plt.plot(x,data)
plt.savefig("./waveform.png")



f, t, stft_data = sp.stft(data, fs=wav.getframerate(), window="hann", nperseg=512, noverlap=256)
print("shape: ", np.shape(stft_data))
print("freq: ",f)
print("sec: ",t)

fig = plt.figure(figsize=(10,4))
spectrum, freqs, t, im = plt.specgram(data, NFFT=512, noverlap=512/16*15, Fs=wav.getframerate(), cmap="gray")
fig.colorbar(im).set_label('Intensity [dB]')
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.savefig("./spectrogram.png")
