import numpy as np

H = np.array([[0,0,0,1,1,1,1],
                 [0,1,1,0,0,1,1],
                 [1,0,1,0,1,0,1]])

#decode, detect error and correct them
def decodeBlock(arr):
	Hy = np.dot(H,arr)
    	#mod Hy by 2 and turn it into from binary to decimal
	Hy = np.mod(np.array(Hy),2)
	err = int(4*Hy[0]+2*Hy[1]+Hy[2])
	if err == 0:
		return [arr[2],arr[4],arr[5],arr[6]]
	else:
		arr[err-1] = 1 - arr[err-1]
		return [arr[2],arr[4],arr[5],arr[6]]

def decode(arr):
	#devide arr into blocks of 7, decode each block and concatenate the result, assemble the resulte to bytes
	out = np.zeros(int(len(arr)/14),dtype=np.uint8)
	arr = arr.reshape(int(len(arr)/7),7)
	for i in range(len(out)):
		arr[2*i] = np.flip(arr[2*i])
		arr[2*i+1] = np.flip(arr[2*i+1])
		bins = np.zeros(8)
		bins[:4] = decodeBlock(arr[2*i+1,:]) 
		bins[4:8] = decodeBlock(arr[2*i,:])
		out[i] = int(bins[0]*128+bins[1]*64+bins[2]*32+bins[3]*16+bins[4]*8+bins[5]*4+bins[6]*2+bins[7])
	return out
