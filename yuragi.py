import wave as wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.fft as sf

# from pydub import AudioSegment
# sound = AudioSegment.from_wav("./sounds/s5.wav")
# sound = sound.set_channels(1)
# sound.export("./sounds/s5_mono.wav", format="wav")

sample_wave_file = "./sounds/s5_mono.wav"

wav = wave.open(sample_wave_file)

print("sampling freq[Hz]: ", wav.getframerate())
print("sampling size[Byte]: ", wav.getsampwidth())
print("sampling num",wav.getnframes())
print("channels: ",wav.getnchannels())

data = wav.readframes(wav.getnframes())
if wav.getsampwidth() == 4:
    data = np.frombuffer(data, dtype=np.int32)
    data = data/np.iinfo(np.int32).max
elif wav.getsampwidth() == 2:
    data = np.frombuffer(data, dtype=np.int16)
    data = data/np.iinfo(np.int16).max

wav.close()


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



def FFT_main(t, x, dt, split_t_r, overlap, window_F, output_FN, y_label, y_unit):
 
    #$B%G!<%?$r%*!<%P!<%i%C%W$7$FJ,3d$9$k!#(B
    split_data = data_split(t, x, split_t_r, overlap)
 
    #FFT$B$r9T$&!#(B
    FFT_result_list = []
    for split_data_cont in split_data:
        FFT_result_cont = FFT(split_data_cont, dt, window_F)
        FFT_result_list.append(FFT_result_cont)
 
    #$B3F%U%l!<%`$N%0%i%U2=(B
    IDN = 0
    for split_data_cont, FFT_result_cont in zip(split_data, FFT_result_list):
        IDN = IDN+1
        plot_FFT(split_data_cont[0], split_data_cont[1], FFT_result_cont[0], FFT_result_cont[1], output_FN, IDN, 0, y_label, y_unit)
 
    #$BJ?6Q2=(B
    fq_ave = FFT_result_list[0][0]
    F_abs_amp_ave = np.zeros(len(fq_ave))
    for i in range(len(FFT_result_list)):
        F_abs_amp_ave = F_abs_amp_ave + FFT_result_list[i][1]
    F_abs_amp_ave = F_abs_amp_ave/(i+1)
 
    plot_FFT(t, x, fq_ave, F_abs_amp_ave, output_FN, "ave", 1, y_label, y_unit)
 
    return fq_ave, F_abs_amp_ave
 
def plot_FFT(t, x, fq, F_abs_amp, output_FN, IDN, final_graph, y_label, y_unit):
    fig = plt.figure(figsize=(12, 4))
    ax2 = fig.add_subplot(121)
    title1 = "time_" + output_FN[:-4]
    plt.plot(t, x)
    plt.xlabel("time [s]")
    plt.ylabel(y_label+"["+y_unit+"]")
    plt.title(title1)
 
    ax2 = fig.add_subplot(122)
    title2 = "freq_" + output_FN[:-4]
    plt.xlabel('freqency(Hz)')
    plt.ylabel(y_label+"["+y_unit+"/rtHz]")
    # plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0,10000)
    plt.plot(fq, F_abs_amp)
    plt.title(title2)
 
    if final_graph == 0:
        plt.savefig(output_FN[:-4]+"_"+str(IDN)+"_FFTtemp"+output_FN[-4:], dpi=300)
    elif final_graph == 1:
        plt.savefig(output_FN, dpi=300)
 
    return 0
 
def FFT(data_input, dt, window_F):
 
    N = len(data_input[0])
 
    #$BAk$NMQ0U(B
    if window_F == "hanning":
        window = np.hanning(N)          # $B%O%K%s%0Ak(B
    elif window_F == "hamming":
        window = np.hamming(N)          # $B%O%_%s%0Ak(B
    elif window_F == "blackman":
        window = np.blackman(N)         # $B%V%i%C%/%^%sAk(B
    else:
        print("Error: input window function name is not sapported. Your input: ", window_F)
        print("Hanning window function is used.")
        hanning = np.hanning(N)          # $B%O%K%s%0Ak(B
 
    #$BAk4X?t8e$N?.9f(B
    x_windowed = data_input[1]*window
 
    #FFT$B7W;;(B
    F = np.fft.fft(x_windowed)
    F_abs = np.abs(F)
    F_abs_amp = F_abs / N * 2
    fq = np.linspace(0, 1.0/dt, N)
 
    #$BAkJd@5(B
    acf=1/(sum(window)/N)
    F_abs_amp = acf*F_abs_amp
 
    #$B%J%$%-%9%HDj?t$^$GCj=P(B
    fq_out = fq[:int(N/2)+1]
    F_abs_amp_out = F_abs_amp[:int(N/2)+1]
 
    return [fq_out, F_abs_amp_out]
 
def data_split(t, x, split_t_r, overlap):
 
    split_data = []
    one_frame_N = int(len(t)*split_t_r) #1$B%U%l!<%`$N%5%s%W%k?t(B
    overlap_N = int(one_frame_N*overlap) #$B%*!<%P!<%i%C%W$9$k%5%s%W%k?t(B
    start_S = 0
    end_S = start_S + one_frame_N
 
    while True:
        t_cont = t[start_S:end_S]
        x_cont = x[start_S:end_S]
        split_data.append([t_cont, x_cont])
 
        start_S = start_S + (one_frame_N - overlap_N)
        end_S = start_S + one_frame_N
 
        if end_S > len(t):
            break
 
 
    return np.array(split_data)

dt = 1 / wav.getframerate() #This value should be correct as real.
output_FN = "test.png"

split_t_r = 0.1 #1$B$D$NOH$GA4BN$N$I$N3d9g$N%G!<%?$rJ,@O$9$k$+!#(B
overlap = 0.5 #$B%*!<%P!<%i%C%WN((B
window_F = "hanning" #$BAk4X?tA*Br(B: hanning, hamming, blackman
y_label = "amplitude"
y_unit = "V"
FFT_main(x, data, dt, split_t_r, overlap, window_F, output_FN, y_label, y_unit)
