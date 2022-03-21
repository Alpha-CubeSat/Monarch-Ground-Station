import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import dask.array as da
from dask.diagnostics import ProgressBar
import h5py
import time
import sys
from tqdm import tqdm
from joblib import Parallel, delayed

fs = 200e3 # sampling rate
fc = 50e3 # chip rate
step = 20
N_FFT = int(512*fs/fc)

prn0 = np.memmap('prng0.c64' , mode='r', dtype='complex64')[0:2049]
prn1 = np.memmap('prng1.c64' , mode='r', dtype='complex64')[0:2049]

prns_fft = np.empty((2, N_FFT), dtype = 'complex64')
prns_fft[1,:] = np.fft.fft(prn1, N_FFT).conjugate()
prns_fft[0,:] = np.fft.fft(prn0, N_FFT).conjugate()

doppler_range = 100

def do_correlations(x):
    c_t = np.zeros(2)
    x_f = np.fft.fft(x).astype('complex64')
    for k in range(-doppler_range,doppler_range):
        c_t[0] = max(c_t[0],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[0]))).max())
        c_t[1] = max(c_t[1],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[1]))).max())

    result = c_t
    result=result / np.abs(x_f.sum())
    return result

signal = np.memmap(sys.argv[1] , mode='r', dtype='complex64')

#y = np.empty(((signal.size-N_FFT)+1,2))
#dy = np.zeros((signal.size-N_FFT)+1)
prn_int = int(N_FFT/step)
print(prn_int)

dmax = np.zeros(0,dtype='int')
data = np.empty(0)
def process(i):
    y = do_correlations(signal[i:i+N_FFT])
    return y[1]-y[0]

dy = Parallel(n_jobs=6)(delayed(process)(i) for i in tqdm(range(0,signal.size-N_FFT,step)))
for i in range(0,int(signal.size/N_FFT)):
    if((i+1)*N_FFT < signal.size):
        dmax = np.append(dmax,i*prn_int + np.abs(dy[i*prn_int:(i+1)*prn_int]).argmax())
    else:
        dmax = np.append(dmax,i*prn_int + np.abs(dy[i*prn_int:signal.size]).argmax())
lastArg = - prn_int
for arg in dmax:
    if(arg > lastArg + prn_int):
        if(dy[arg] > 0):
            data = np.append(data,1)
        else:
            data = np.append(data,0)
        lastArg=arg
print(data)

plt.title('Corrrelation')
plt.xlabel('Sample')
plt.ylabel('Normalized correlation magnitude difference')
plt.plot(dy)
plt.show()
