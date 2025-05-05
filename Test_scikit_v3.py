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
eps1     = 5.0    # εr real
cond1    = 10.0    # σ (S/m)
thickness= 0.2    # m

# ε compleja = ε1 – j·σ/ω
epsilon_complex = eps1 - 1j * cond1 / omega_pos
gamma           = 1j * omega_pos * np.sqrt(epsilon_complex)
Z0              = np.sqrt(1.0 / epsilon_complex)

# ⚠️ Nota: usar z0 en lugar de Z0 para evitar deprecación
medium = rf.media.DefinedGammaZ0(frequency=freq,
                                 gamma=gamma,
                                 z0=Z0)
layer = medium.line(d=thickness, unit='m')

# 5) S-parameters en f>0
S11_pos = layer.s[:,0,0]
S21_pos = layer.s[:,1,0]

# imprimimos los parámetros S
plt.plot(f_pos, np.abs(S11_pos), label='S11')
plt.plot(f_pos, np.abs(S21_pos), label='S21')
plt.show()

plt.plot(f_full[:N//2], np.abs(P_full[:N//2]), label='P_full')
plt.plot(f_full[:N//2], np.abs(S11_pos*P_full[:N//2]), label='Reflejado')
plt.plot(f_full[:N//2], np.abs(S21_pos*P_full[:N//2]), label='Transmitido')
plt.legend()
plt.show()

# añadimos un 0 en el medio para que la FFT sea simétrica
S11 = np.concatenate((S11_pos, [0], S11_pos[::-1]))
S21 = np.concatenate((S21_pos, [0], S21_pos[::-1]))

# 7) Señales en el tiempo
pulse_inc = np.real(np.fft.ifft(P_full))          / dt
pulse_ref = np.real(np.fft.ifft(S11 * P_full)) / dt
pulse_trn = np.real(np.fft.ifft(S21 * P_full)) / dt

# centramos en t=0
pulse_inc = np.fft.fftshift(pulse_inc)
pulse_ref = np.fft.fftshift(pulse_ref)
pulse_trn = np.fft.fftshift(pulse_trn)

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
