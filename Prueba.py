import matplotlib.pyplot as plt
from numpy.fft import *
import numpy as np
# Creamos la exponencial
x = np.linspace(0,10,1001)
dx = x[1]-x[0]
spread = 1
delay = 5

y = np.exp(-(x-delay)**2 / (2*spread**2))
plt.plot(x,y)
plt.show()
# Hacemos transformada de Fourier
fq = fftshift(fftfreq(len(x)/dx))
y = fftshift(fft(y))
