from os import pread
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal
# import dask.array as da
# from dask.diagnostics import ProgressBar
# import h5py
import time
import sys
import scipy.signal
from tqdm import tqdm
from joblib import Parallel, delayed

fs = 200e3 # sampling rate
fc = 50e3 # chip rate
step = 100
smooth_range = 1
N_FFT = int(512*fs/fc)
N_PRN_LEN = 596*fs/fc
N_BYTE_PER_PACKET = int(8)
N_BIT_PER_BYTE = 14

prn0 = np.memmap('prng0.c64' , mode='r', dtype='complex64')[0:2049]
prn1 = np.memmap('prng1.c64' , mode='r', dtype='complex64')[0:2049]
preamble = np.memmap('preambleg.c64' , mode='r', dtype='complex64')[0:2049]

prns_fft = np.empty((2, N_FFT), dtype = 'complex64')
prns_fft[1,:] = np.fft.fft(prn1, N_FFT).conjugate()
prns_fft[0,:] = np.fft.fft(prn0, N_FFT).conjugate()

preamble = np.fft.fft(preamble, N_FFT).conjugate()

doppler_range = 100

def do_correlations(x):
    c_t = np.zeros(3)
    x_f = np.fft.fft(x).astype('complex64')
    for k in range(-doppler_range,doppler_range):
        c_t[0] = max(c_t[0],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[0]))).max())
        c_t[1] = max(c_t[1],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[1]))).max())
        c_t[2] = max(c_t[2],(np.abs(np.fft.ifft(np.roll(x_f,k) * preamble))).max())

    result = c_t
    result=result / np.abs(x_f.sum())
    return result

signal = np.memmap(sys.argv[1] , mode='r', dtype='complex64')

#y = np.empty(((signal.size-N_FFT)+1,2))
#dy = np.zeros((signal.size-N_FFT)+1)

def process(i):
    y = do_correlations(signal[i:i+N_FFT])
    if(y[2] - y[0] < 0):
        return [y[1] -y[0],0]
    return [y[1]-y[0],y[2]-y[0]]

dy = np.array(Parallel(n_jobs=6)(delayed(process)(i) for i in tqdm(range(0,signal.size-N_FFT,step))))
#find all peaks in dy[1] as possible start of a packet
peaks = scipy.signal.find_peaks(dy[:,1], distance=int(N_FFT/step),prominence=0.02)
print(peaks)
dy_spaced = np.zeros(len(signal))
preamble_spaced = np.zeros(len(signal))

preamble_candidate = peaks[0]
for i in range(len(preamble_candidate)):
    preamble_candidate[i]*=step
for i in range(len(dy)):
    dy_spaced[i*step] = dy[i,0]
    preamble_spaced[i*step] = dy[i,1]
data = np.zeros((len(preamble_candidate),N_BYTE_PER_PACKET*N_BIT_PER_BYTE))

for i in range(preamble_candidate.size):
    #if i + packet length reaches end of signal, ignore it
    if(preamble_candidate[i] + N_BYTE_PER_PACKET*N_BIT_PER_BYTE*N_PRN_LEN > len(signal)):
        continue
    plt.axvline(preamble_candidate[i],ymin=0.8,ymax=1.0,color='r')
    offset = 0
    for j in range(step,int(N_PRN_LEN),step):
        preamble_corr_sum=0
        dy_corr_sum=0
        for v in range(preamble_candidate[i]+j-smooth_range*step,preamble_candidate[i]+j+smooth_range*step,step):
            preamble_corr_sum += np.abs(preamble_spaced[v])
            dy_corr_sum += np.abs(dy_spaced[v])
        if preamble_corr_sum < dy_corr_sum:
            offset = j
            break
    plt.axvline(x=preamble_candidate[i]+offset,ymin=0.8,ymax=1.0,color='g')
    for k in range(int(N_BYTE_PER_PACKET*N_BIT_PER_BYTE)):
        window_start = int(preamble_candidate[i]+offset+k*N_PRN_LEN)
        window_end = int(preamble_candidate[i]+offset+(k+1)*N_PRN_LEN)
        if(window_end > signal.size):
            break
        sum = dy_spaced[window_start:window_end].sum()
        if sum > 0:
            data[i,int(k)] = 1
        else:
            data[i,int(k)] = 0
        #plt.axvline(x = preamble_candidate[i]+offset + int(k*N_PRN_LEN),ymin=0.3,ymax=0.7,color = 'y')

#Print bits data
print("Bits data:")
np.set_printoptions(threshold=np.inf)
print(data)

plt.title('Corrrelation')
plt.xlabel('Sample')
plt.ylabel('Normalized correlation magnitude difference')
    
plt.plot(dy_spaced)
plt.plot(preamble_spaced)
plt.show()
