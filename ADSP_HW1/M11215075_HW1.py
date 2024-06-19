# -*- coding: utf-8 -*-
import numpy as np
import math 
import matplotlib.pyplot as plt

fs = 6000
N = 17
delta = 1e-4

k = int((N-1)/2)
N_ext = k + 2
Fn = np.array([0, 0.05, 0.1, 0.19, 0.26, 0.3, 0.35, 0.4, 0.45, 0.5])

WEIGHT = ((0, 0.2, 1), (0.25, 0.5, 0.6))
Hd_pass = (0, 0.225)

def W(x):
    # Pass band
    if WEIGHT[0][0] <= x <= WEIGHT[0][1]:  
        return WEIGHT[0][2]
    # Stop band
    elif WEIGHT[1][0] <= x <= WEIGHT[1][1]:  
        return WEIGHT[1][2]
    # Transition band
    else: 
        return 0

# Hd(Fm) = s[n]cos(2£kFm(n))+(-1)^mW^-1(Fm)e
def Hd(x):
    if Hd_pass[0] <= x <= Hd_pass[1]:
        return 1
    else:
        return 0

# R(Fn) function =    
def R_Fn(F, s):
    a = 0
    for n in range(k+1):
        a += s[n]* math.cos(2*math.pi* n * F)
    return a

# err(F) function
def err(F, s):
    return (R_Fn(F, s) - Hd(F)) * W(F)
max_errl = float('inf')
iteration = 0
while True:
    iteration += 1
    #step2
    # As = b
    A = []
    for m in range(k+2):
        row = []
        for n in range(k+2):
            if n == 0: row.append(1)
            elif n == k+1: row.append(((-1)**m) / W(Fn[m]))
            else: row.append(math.cos(2 * math.pi * Fn[m] * n))
        A.append(row)

    A = np.array(A)
    A_inv = np.linalg.inv(A)

    b = []
    for m in range(k+2):
        b.append(Hd(Fn[m]))
    b = np.array(b)

    s = A_inv @ b
    #step3 find error local maximum
    n_extreme = []
    max_err = -1

    F_ll  = None
    F_l = None
    for i in range(int(0.5 / delta) +2):
        if i == 0: continue
        if i == 1: 
            F_ll = 0
            F_l = err(0 * delta, s)
        if i == int(0.5 / delta) +1:
            F = 0 
        else:
            F = err(i * delta, s)
        
        if F_l - F > 0 and F_l - F_ll > 0 or F_l - F < 0 and F_l - F_ll < 0:
            n_extreme.append((i-1)*delta)
            if max_err < abs(F_l): max_err = abs(F_l)

        F_ll = F_l
        F_l = F
    print(f'Iteration {iteration} - Maximal Error: {max_err}')
    if 0 <= max_errl-max_err < delta: break
    max_errl = max_err
    Fn = n_extreme[:k+2]
    
    
    
#step6 h[k] = s[0]
h = []
for i in range(N):
    if i < k:
        h.append(s[k-i]/2)
    elif i == k:
        h.append(s[0])
    else:
        h.append(s[i-k]/2)

plt.figure(figsize=(10,5))

t = np.arange(0.0, 0.5, delta)
plt.plot(t, np.array([ R_Fn(t[i], s) for i in range(len(t)) ]), lw=2, color = 'm')
plt.plot(t, np.array([ Hd(t[i])    for i in range(len(t)) ]), lw=2, color = 'c')
plt.legend(['FIR Filter', 'Desire Filter']) 
plt.title("Frequency Response")
plt.savefig('Frequency_Response.png')
plt.show()


plt.figure(figsize=(10,5))
t = np.arange(0, N)
plt.stem(t, np.array([ h[i] for i in range(len(t)) ]),linefmt='DarkCyan', markerfmt='ko', basefmt='GoldenRod')
plt.legend(['h[n]']) 
plt.title("Impulse Response")
plt.savefig('Impulse_Response.png')
plt.show()





