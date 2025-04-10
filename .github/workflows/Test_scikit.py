import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

# Frecuencia de 1 a 10 GHz
freq = rf.Frequency(1, 10, 201, unit='GHz')

# Constantes físicas
c = 299_792_458  # m/s
mu0 = 4e-7 * np.pi
epsilon0 = 8.854187817e-12

# Parámetros del material
er = 1           # permitividad relativa (puede ser 1 para un conductor)
mur = 1          # permeabilidad relativa
sigma = 5.8e7    # conductividad [S/m] (Ej: cobre ~ 5.8e7 S/m)
thickness = 1e-3 # grosor de la capa (1 mm)

# Frecuencia angular (vector)
omega = 2 * np.pi * freq.f  # rad/s

# Permitividad compleja con conductividad
epsilon_complex = epsilon0 * er - 1j * sigma / omega

# Permeabilidad
mu = mu0 * mur

# Constante de propagación γ
gamma = 1j * omega * np.sqrt(mu * epsilon_complex)

# Impedancia característica Z0
Z0 = np.sqrt(mu / epsilon_complex)

# Crear el medio con gamma y Z0 definidos
medium = rf.media.DefinedGammaZ0(
    frequency=freq,
    gamma=gamma,
    Z0=Z0
)

# Crear la capa delgada como línea de transmisión
layer = medium.line(d=thickness, unit='m')

# Obtener S11 y S21
s11 = layer.s[:, 0, 0]
s21 = layer.s[:, 1, 0]

# Graficar
plt.plot(freq.f, 20*np.log10(np.abs(s11)), label='|S11| (dB)')
plt.plot(freq.f, 20*np.log10(np.abs(s21)), label='|S21| (dB)')
plt.xlabel('Frecuencia (GHz)')
plt.ylabel('Magnitud (dB)')
plt.title('S11 y S21 de una capa con conductividad')
plt.legend()
plt.grid()
plt.show()
