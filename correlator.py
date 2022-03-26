import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal
# import dask.array as da
# from dask.diagnostics import ProgressBar
# import h5py
import time
import sys
from tqdm import tqdm
from joblib import Parallel, delayed

fs = 200e3 # sampling rate
fc = 50e3 # chip rate
step = 100
N_FFT = int(512*fs/fc)
N_PRN_LEN = int(598*fs/fc)
N_BYTE_PER_PACKET = int(8)

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
    if i%step != 0:
        return [0,0]
    y = do_correlations(signal[i:i+N_FFT])
    return [y[1]-y[0],y[2]-y[0]]

dy = np.array(Parallel(n_jobs=6)(delayed(process)(i) for i in tqdm(range(0,signal.size-N_FFT))))

#find the index of values in x that are 100 times greater than the positive average
def get_pos_index(x):
    #turn negative value in x to 0
    x = np.array( [ num if num > 0 else 0 for num in x ] )
    return np.where(x>2000*x.mean())


#get all preamble_candidates
preamble_candidates = get_pos_index(dy[:,1])[0]
preamble_final=np.zeros(0,dtype="int")

preamble_wave_start = -N_PRN_LEN*2

#if a preamble is larger than N_FFT + preamble_wave_start, push the averageof preamble_wave_start and preamble_wave_end to the final list, and set average to this preamble
#if a preamble is smaller than N_FFT + preamble_wave_start, it's new preamble_wave_end
index_sum = 0
index_count = 0
for i in preamble_candidates:
    if i > 2*N_PRN_LEN + preamble_wave_start:
        if preamble_wave_start > 0:
            preamble_final = np.append(preamble_final,int(index_sum/index_count))
        preamble_wave_start = i
        index_sum = i*signal[i]
        index_count = signal[i]
    else:
        index_sum+=i*signal[i]
        index_count+=signal[i]
preamble_final = np.append(preamble_final,int(index_sum/index_count))

#print preamble locations
print("Preamble locations:")
print(preamble_final)
data = np.zeros((len(preamble_final),N_BYTE_PER_PACKET*8))
for i in range(preamble_final.size):
    plt.axvline(preamble_final[i],ymin=0,ymax=0.5,color='r')
    for k in range(int(N_BYTE_PER_PACKET*8)):
        window_start = int(preamble_final[i]+N_PRN_LEN/2+k*N_PRN_LEN)
        window_end = int(preamble_final[i]+N_PRN_LEN/2+(k+1)*N_PRN_LEN)
        if(window_end > signal.size):
            break
        sum = dy[window_start:window_end,0].sum()
        if sum > 0:
            data[i,int(k)] = 1
        else:
            data[i,int(k)] = 0
        #plt.axvline(x = preamble_final[i]+N_PRN_LEN/2 + k*N_PRN_LEN,color = 'y')
    if(preamble_final[i] + N_PRN_LEN/2 + N_PRN_LEN*8 <= signal.size):
        plt.axvline(x = preamble_final[i]+N_PRN_LEN/2 + (N_BYTE_PER_PACKET*8)*N_PRN_LEN,ymin=0,ymax=0.5,color = 'y')
        
#Print bits data
print("Bits data:")
print(data)

#Accumulate bits in  data to bytes
def get_byte(data):
    bytes = np.zeros(N_BYTE_PER_PACKET,dtype="int")
    for i in range(N_BYTE_PER_PACKET):
        for k in range(8):
            bytes[i] = bytes[i] + data[i*8+k]*int(2**(k))
    return bytes

#print data to bytes in hex for each row
print("Bytes data:")
for i in range(data.shape[0]):
    print(" ".join(["%02X" % x for x in get_byte(data[i])]))



plt.title('Corrrelation')
plt.xlabel('Sample')
plt.ylabel('Normalized correlation magnitude difference')
    
plt.plot(dy)
plt.show()
