import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

# — Constantes físicas y definición de pulso — #
def gaussian_pulse_frequency(omega, t0, sigma):
    """
    Espectro de un pulso gaussiano desplazado en el tiempo:
      G(ω) = sqrt(2π)·σ·exp(−(σ²·ω²)/2)·exp(−j ω t0)
    """
    return np.sqrt(2*np.pi)*sigma * np.exp(- (sigma**2 * omega**2)/2) * np.exp(-1j*omega*t0)

dt    = 0.05     # paso temporal (c=1)
N     = 1001     # puntos
t     = (np.arange(N) - (N-1)/2) * dt

# 1) Espectro completo
f_full     = np.fft.fftfreq(N, d=dt)      # incluye f<0
omega_full = 2*np.pi * f_full
P_full     = gaussian_pulse_frequency(omega_full, t0=0.0, sigma=0.25)

# 2) Sólo frecuencias positivas (excluimos f=0 para evitar división por cero)
mask_pos = f_full > 0
mask_neg = f_full < 0
f_pos    = f_full[mask_pos]
omega_pos= omega_full[mask_pos]
P_pos    = P_full[mask_pos]

# 3) Objeto Frequency de SKRF
freq = rf.Frequency.from_f(f_pos, unit='Hz')

# 4) Definición del slab
eps1     = 1.0    # εr real
cond1    = 2.0   # σ (S/m)
thickness= 0.2    # m

# ε compleja = ε1 – j·σ/ω
epsilon_complex = eps1 - 1j * cond1 / omega_pos
gamma           = 1j * omega_pos * np.sqrt(epsilon_complex)
Z0              = np.sqrt(1.0 / epsilon_complex)


phi_11 = np.cosh(gamma * thickness)
phi_12 = Z0 * np.sinh(gamma * thickness)
phi_21 = 1 / Z0 * np.sinh(gamma * thickness)
phi_22 = np.cosh(gamma * thickness)

T = (2) / (phi_11 + phi_12 + phi_21 + phi_22)
R = (phi_11 + phi_12 - phi_21 - phi_22) / (phi_11 + phi_12 + phi_21 + phi_22)

# 5) S-parameters en f>0
S11_pos = R
S21_pos = T
# imprimimos los parámetros S
plt.plot(f_pos, np.abs(S11_pos), label='S11')
plt.plot(f_pos, np.abs(S21_pos), label='S21')
plt.legend()
plt.show()

plt.plot(f_full[:N//2], np.abs(P_full[:N//2]), label='P_full')
plt.plot(f_full[:N//2], np.abs(S11_pos*P_full[:N//2]), label='Reflejado')
plt.plot(f_full[:N//2], np.abs(S21_pos*P_full[:N//2]), label='Transmitido')
plt.legend()
plt.show()

# añadimos un 0 en el medio para que la FFT sea simétrica
S11 = np.concatenate((S11_pos, [S11_pos[-1]], S11_pos[::-1]))
S21 = np.concatenate((S21_pos, [S21_pos[-1]], S21_pos[::-1]))

# 7) Señales en el tiempo
pulse_inc = np.fft.irfft(P_pos, n=N) / dt
pulse_ref = np.fft.irfft(S11_pos * P_pos, n=N) / dt
pulse_trn = np.fft.irfft(S21_pos * P_pos, n=N) / dt


# centramos en t=0
pulse_inc = np.fft.fftshift(pulse_inc)
pulse_ref = np.fft.fftshift(pulse_ref)
pulse_trn = np.fft.fftshift(pulse_trn)

print('R='+ str(np.max(np.abs(pulse_ref))/np.max(pulse_inc)))
print('T='+ str(np.max(pulse_trn)/np.max(pulse_inc)))


# 8) Graficar
plt.figure(figsize=(8,5))
plt.plot(t,    pulse_inc, label='Incidente')
plt.plot(t,    pulse_ref, '--', label='Reflejado')
plt.plot(t,    pulse_trn, ':',  label='Transmitido')
plt.xlabel('t (c=1)')
plt.ylabel('Amplitud')
plt.title('Pulso EM con slab en x=[7, 7.2]')
plt.legend()
plt.tight_layout()
plt.show()
