from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import cwt

t = np.linspace(-1, 1, 200, endpoint=False)
a = np.cos(2 * np.pi * 3 * t)
b = signal.gausspulse(t - 0.4, fc=10)*5
c = signal.gausspulse(t + 0.4, fc=7)*5
sig  = a + b + c 
#widths = np.arange(1, 31, step = 0.1)
#cwtmatr = signal.cwt(sig, signal.ricker, widths)
Wx_k, scales = cwt(sig, 'morlet')
cwtmatr = Wx_k 

freqs = pywt.scale2frequency('morl', scales) / np.mean(np.diff(t))

fig,ax = plt.subplots(3,1)
ax[0].plot(a)
ax[0].plot(b)
ax[0].plot(c)
ax[1].plot(sig)
#ax[2].imshow(np.abs(cwtmatr), cmap='jet', aspect='auto')
ax[2].pcolormesh(t, freqs, np.abs(cwtmatr), cmap='jet')
#fig.colorbar(im, ax=ax[2])
plt.show()

imshow(cwtmatr, yticks=scales, abs=1,
       title="abs(CWT) | Morlet wavelet",
       ylabel="scales", xlabel="samples")

# https://dsp.stackexchange.com/questions/74032/scalograms-in-python

##############################################################
##############################################################

import numpy as np
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow

#%%# Helper fn + params #####################################################
def exp_am(t, offset):
    return np.exp(-pi*((t - offset) / .1)**10)

pi = np.pi
v1, v2, v3 = 64, 128, 32

#%%# Make `x` & plot #########################################################
t = np.linspace(0, 1, 2048, 1)
x = (np.sin(2*pi * v1 * t) * exp_am(t, .2) +
     (np.sin(2*pi * v1 * t) + 2*np.cos(2*pi * v2 * t)) * exp_am(t, .5)  + 
     (2*np.sin(2*pi * v2 * t) - np.cos(2*pi * v3 * t)) * exp_am(t, .8))
plot(x, title="x(t) | t=[0, ..., 1], %s samples" % len(x), show=1)

#%%# Take CWT & plot #########################################################
Wx, scales = cwt(x, 'morlet')
imshow(Wx, yticks=scales, abs=1,
       title="abs(CWT) | Morlet wavelet",
       ylabel="scales", xlabel="samples")

##############################################################
##############################################################
import pywt
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(512)
y = np.sin(2*np.pi*x/32)
coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
fig,ax = plt.subplots(2,1)
ax[0].plot(x,y)
ax[1].matshow(coef) 
plt.show() 
#----------

import pywt
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')

fig,ax = plt.subplots(2,1)
ax[0].plot(t,sig)
ax[1].imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
plt.show() 
