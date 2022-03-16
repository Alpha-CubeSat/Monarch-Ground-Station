import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import dask.array as da
from dask.diagnostics import ProgressBar
import h5py
import time
import sys
from tqdm import tqdm

fs = 200e3 # sampling rate
fc = 50e3 # chip rate
step = 20
N_FFT = int(512*fs/fc)
print(N_FFT)

prn0 = np.memmap('prng0.c64' , mode='r', dtype='complex64')[0:2049]
prn1 = np.memmap('prng1.c64' , mode='r', dtype='complex64')[0:2049]

prns_fft = np.empty((2, N_FFT), dtype = 'complex64')
prns_fft[1,:] = np.fft.fft(prn1, N_FFT).conjugate()
prns_fft[0,:] = np.fft.fft(prn0, N_FFT).conjugate()

doppler_range = 100

def do_correlations(x):
    c_t = np.zeros(2);
    x_f = np.fft.fft(x).astype('complex64')
    for k in range(-doppler_range,doppler_range):
        c_t[0] = max(c_t[0],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[0]))).max())
        c_t[1] = max(c_t[1],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[1]))).max())

    result = c_t
    result=result / np.abs(x_f.sum())
    return result

signal = np.memmap(sys.argv[1] , mode='r', dtype='complex64')

y = np.empty(((signal.size-N_FFT)+1,2))
dy = np.zeros((signal.size-N_FFT)+1)
for i in tqdm(range(0,signal.size-N_FFT,step)):
    y[i,:] = do_correlations(signal[i:i+N_FFT])
    dy[i] = y[i,1]-y[i,0]
plt.title('Corrrelation')
plt.xlabel('Sample')
plt.ylabel('Normalized correlation magnitude difference');
plt.plot(dy)
plt.show()
