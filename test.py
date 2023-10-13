import numpy as np
from scipy.fftpack import fft,ifft
import scipy.signal
import librosa

signal = np.array([1]*64, dtype=float)
stft = librosa.stft(signal, n_fft=8, hop_length=2, center=True)
stft = stft.T
a = list(stft[1])
b = a + [np.conj(ele) for ele in a[-2:0:-1]]
print(b)
print(list(ifft(b)))

def STFT(x, n_fft=2048, hop_length=512, window='hann'):
    # x - signal
    # n_fft - fft window size 
    # hop_length - step size between ffts
    # window - window type. See scipy.signal.get_window
    # return spectrogram 
    p = 0
    res = []
    windf = scipy.signal.get_window(window, n_fft)
    y = np.pad(x, n_fft//2)
    while True:
        if p>=len(x)+hop_length:
            break
        window = y[p:p+n_fft]
        window = windf * window
        window = fft(window)
        res.append(window[:n_fft//2+1])
        p += hop_length
    res = np.array(res, dtype=complex)
    res = res.T
    return res

stft1 = STFT(signal, n_fft=8, hop_length=2)
stft1 = stft1.T
a1 = list(stft1[1])
b1 = a1 + [np.conj(ele) for ele in a1[-2:0:-1]]
print(b1)
print(list(ifft(b1)))