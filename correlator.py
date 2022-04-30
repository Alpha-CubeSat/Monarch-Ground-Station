from os import pread
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal
# import dask.array as da
# from dask.diagnostics import ProgressBar
# import h5py
import time
import sys
import hamming
import scipy.signal
from tqdm import tqdm
from joblib import Parallel, delayed

G = 9.81
dps = 245

fs = 200e3 # sampling rate
fc = 50e3 # chip rate
step = 100

doppler_range = 100
doppler_center = 0

smooth_range = 1

N_FFT = int(512*fs/fc)
N_PRN_LEN = 596*fs/fc
N_BYTE_PER_PACKET = int(7)
N_BIT_PER_BYTE = 14

prn0 = np.memmap('prng0.c64' , mode='r', dtype='complex64')[0:2049]
prn1 = np.memmap('prng1.c64' , mode='r', dtype='complex64')[0:2049]
preamble = np.memmap('preambleg.c64' , mode='r', dtype='complex64')[0:2049]

prns_fft = np.empty((2, N_FFT), dtype = 'complex64')
prns_fft[1,:] = np.fft.fft(prn1, N_FFT).conjugate()
prns_fft[0,:] = np.fft.fft(prn0, N_FFT).conjugate()

preamble = np.fft.fft(preamble, N_FFT).conjugate()

def do_correlations(x):
    c_t = np.zeros(3)
    x_f = np.fft.fft(x).astype('complex64')
    for k in range(-doppler_range+doppler_center,doppler_range+doppler_center):
        c_t[0] = max(c_t[0],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[0]))).max())
        c_t[1] = max(c_t[1],(np.abs(np.fft.ifft(np.roll(x_f,k) * prns_fft[1]))).max())
        c_t[2] = max(c_t[2],(np.abs(np.fft.ifft(np.roll(x_f,k) * preamble))).max())

    result = c_t
    result=result / np.abs(x_f.sum())
    return result

def process(i):
    y = do_correlations(signal[i:i+N_FFT])
    if(y[2] - y[0] < 0):
        return [y[1] -y[0],0]
    return [y[1]-y[0],y[2]-y[0]]

if(len(sys.argv) != 2):
    print("Usage: ./correlator.py <file_name>")
    exit()
#check if the file name ends with .c64
if(sys.argv[1][-4:] != '.c64'):
    #read dy from npy file
    dy = np.load(sys.argv[1])
else:
    signal = np.memmap(sys.argv[1] , mode='r', dtype='complex64')
    dy = np.array(Parallel(n_jobs=6)(delayed(process)(i) for i in tqdm(range(0,signal.size-N_FFT,step))))
    #save dy to file
    np.save(sys.argv[1][:-4]+'.npy',dy)
    
#find all peaks in dy[1] as possible start of a packet
peaks = scipy.signal.find_peaks(dy[:,1], distance=int(2*N_FFT/step),prominence=0.05)

dy_spaced = np.zeros(len(dy)*step)
preamble_spaced = np.zeros(len(dy)*step)

preamble_candidate = peaks[0]
for i in range(len(preamble_candidate)):
    preamble_candidate[i]*=step
for i in range(len(dy)):
    dy_spaced[i*step] = dy[i,0]
    preamble_spaced[i*step] = dy[i,1]
data = np.zeros((len(preamble_candidate),N_BYTE_PER_PACKET*N_BIT_PER_BYTE))
data_start = np.zeros(len(preamble_candidate))

for i in range(preamble_candidate.size):
    #if i + packet length reaches end of signal, ignore it
    if(preamble_candidate[i] + N_BYTE_PER_PACKET*N_BIT_PER_BYTE*N_PRN_LEN > len(dy_spaced)):
        #resize data
        data = data[0:i,:]
        break
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
    data_start[i] = preamble_candidate[i] + offset
    for k in range(int(N_BYTE_PER_PACKET*N_BIT_PER_BYTE)):
        window_start = int(preamble_candidate[i]+offset+k*N_PRN_LEN)
        window_end = int(preamble_candidate[i]+offset+(k+1)*N_PRN_LEN)
        if(window_end > dy_spaced.size):
            break
        sum = dy_spaced[window_start:window_end].sum()
        if sum > 0:
            data[i,int(k)] = 1
        else:
            data[i,int(k)] = 0
        #plt.axvline(x = preamble_candidate[i]+offset + int(k*N_PRN_LEN),ymin=0.3,ymax=0.7,color = 'y')

#parse the data
def parse_msg(msg):
    #turn msg data type to signed int
    msg = np.array(msg,dtype='int8')
    ax = msg[1]/128.0 * 2 * G
    ay = msg[2]/128.0 * 2 * G
    az = msg[3]/128.0 * 2 * G
    gx = msg[4]/128.0 * dps
    gy = msg[5]/128.0 * dps
    gz = msg[6]/128.0 * dps
    #print ax, ay, az, gx, gy, gz
    print("ax: %f m/s^s, ay: %f m/s^s, az: %f m/s^s, gx: %f d/s, gy: %f d/s, gz: %f d/s" % (ax, ay, az, gx, gy, gz))

for i in range(data.shape[0]):
    out = hamming.decode(data[i])
    #print out in hex format
    np.set_printoptions(formatter={'int':hex})
    if(out[0] == 0x1E):
        plt.axvline(x=data_start[i],color='r')
        for k in range(1,int(N_BYTE_PER_PACKET*N_BIT_PER_BYTE+1)):
            plt.axvline(x=data_start[i]+k*N_PRN_LEN,color='y')
        print(out)
        parse_msg(out) 


plt.title('Corrrelation')
plt.xlabel('Sample')
plt.ylabel('Normalized correlation magnitude difference')
    
plt.plot(dy_spaced)
plt.plot(preamble_spaced)
plt.show()
