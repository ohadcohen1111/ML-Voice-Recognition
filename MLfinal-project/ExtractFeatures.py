import librosa
import numpy as np
import wave
import statistics
import scipy.io.wavfile

def spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean

def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.log(X + 1e-20)

    vtf = np.exp(X.mean(axis=0, keepdims=True)) / norm

    return np.squeeze(vtf, axis=0)



win_s = 4096
hop_s = 512

y, sr = librosa.load("b0019.wav")
# print("Average frequency = " + str(np.array(pitches).mean()/1000) + " hz")
print("std = ", np.std(y)*10)
print("Median of data-set is : % s " % (statistics.median(y)))

fs = (1/sr)*512
# print("fs: ", fs)
# file = wave.open('b0019.wav')
# print("Frame rate: ", file.getframerate())
# print("n Frames: ", file.getnframes())
# print(FeatureSpectralFlatness(y, 1))

def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    result_d = {
        'mean': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt
    }

    return result_d

print(spectral_properties(y, fs))
samplerates, data = scipy.io.wavfile.read("b0019.wav")
print("S: ", samplerates)
print("data: ", data)
print("length: ", len(data))
