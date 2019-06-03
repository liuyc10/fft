
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1200)

y = np.sin(2*np.pi*50*x) + np.sin(2*np.pi*120*x)  #+ 2*np.random.randn(x.size)

Y = np.fft.fft(y[:600])

yreal = Y.real
yimag = Y.imag

yf = abs(Y)
yf1 = abs(Y / (len(x)/2))
yf2 = yf1[range(int(len(x)/2))]

xf = np.arange(len(y)/2)
xf1 = xf
xf2 = xf[range(int(len(x)/2))]

plt.subplot(221)
plt.plot(x[0:150], y[0:150])

plt.subplot(222)
plt.plot(xf,yf, 'r')

plt.subplot(223)
plt.plot(xf1, yf1, 'g') 

plt.subplot(224) 
plt.plot(xf2, yf2, 'b')

plt.show()