import wave as wave
import numpy as np

sample_wave_file = "./sounds/roji5.wav"

wav = wave.open(sample_wave_file)

print("sampling freq[Hz]: ", wav.getframerate())
print("sampling size[Byte]: ", wav.getsampwidth())
print("sampling num",wav.getnframes())
print("channels: ",wav.getnchannels())
