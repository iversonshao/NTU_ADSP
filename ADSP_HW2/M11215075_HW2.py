import numpy as np
import matplotlib.pyplot as plt
from math import floor


k = 16
N = 2*k + 1
N_pts = 10000

def hilbert_transform(x):
    #step1
    if x == 0:
        return 0
    elif 0 < x < 0.5: 
        return -1j
    elif 0.5 < x < 1:
        return 1j
    
Hd = [] #sample
for i in np.arange(0, 1, 1/N):
    Hd.append(hilbert_transform(i))

#transision band
Hd[0] = -0.9j #H(1/33)
Hd[k] = -0.7j #H(16/33)
Hd[k+1] = 0.7j #H(17/33)
Hd[-1] = 0.9j #H(32/33)

Hd = np.array(Hd)
#step2
r1n = np.fft.ifft(Hd)

#step3
rn = np.roll(r1n, floor(N/2))

#step4

RF = []
F = np.arange(0.0, 1.0, 1/N_pts)
for F_idx in F:
    i = 0
    for n in range(-k, k+1):
        i += rn[n+k]*np.exp(-1j*2*np.pi*n*F_idx)
    RF.append(i)
RF = np.array(RF)



#Impulse response
plt.figure(figsize=(10,5))
plt.stem(np.array(range(-k, k+1)), rn.real,linefmt='DarkCyan', markerfmt='ko', basefmt='GoldenRod')
plt.title("Impulse Response")
plt.xlabel("N")
plt.savefig('Impulse_Response.png')
plt.show()

#imaginary part of frequency response
plt.figure(figsize=(10,5))
plt.plot(np.arange(0, 1, 1/N), Hd.imag, color='r', linestyle=':')
plt.plot(F, RF.imag)
plt.title("Frequency Response")
plt.xlabel("F")
plt.savefig('Frequency_Response.png')
plt.show()

