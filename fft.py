# %% 
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *

x = np.linspace(0, 10, 100001)
dx = x[1] - x[0]
spread = .5
delay = 5

y = np.exp( - np.power(x - delay,2) / (2 * np.power(spread, 2)))

plt.plot(x, y)
# %%

fq = fftshift(fftfreq(len(x))/dx)
Y = fftshift(fft(y))

plt.plot(fq, np.abs(Y), '.-')
plt.xlim(-1, 1)