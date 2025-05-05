import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from skrf.media.freespace import Freespace

# — Constantes físicas — #
eps0_vac = 1.0  # permitividad relativa del vacío (c=1 ⇒ eps0_rel=1)

def gaussian_pulse(t, t0, sigma):
    return np.exp(-((t - t0)**2) / (2 * sigma**2))

# 1) Discretización temporal y pulso
dt    = 1e-3       # paso temporal (c = 1 ⇒ dt en “distancia”)
N     = 2**12      # número de puntos
t     = np.arange(N) * dt
t0    = 0.5        # centro del pulso (colocado en x = –0.5 a t=0)
sigma = 0.25       # ancho
pulse = 0.5 * gaussian_pulse(t, t0, sigma)

# 2) FFT del pulso y vector de frecuencias
P_full = np.fft.fft(pulse)
P_full[0] = 0                      # anulamos DC para no generar NaN
f_full  = np.fft.fftfreq(N, d=dt)

plt.figure(figsize=(8,5))
plt.plot(f_full, np.abs(P_full), label='FFT del pulso')
plt.xlim(0, 5)
plt.xlabel('Frecuencia (Hz)')
plt.legend()
plt.show()

# 3) Frecuencias positivas
pos = f_full >= 0
f   = f_full[pos]
P   = P_full[pos]

# 4) Creamos el objeto Frequency de scikit-rf
freq = rf.Frequency.from_f(f, unit='Hz')

# 5) Definición de medios
eps0_rel = 1.0    # medio 0: vacío sin pérdidas
eps1     = 5.0    # permitividad real del slab
cond1    = 1.0    # conductividad del slab (S/m)
omega    = 2 * np.pi * f

# medio 0: Freespace con eps_r=1
med0 = Freespace(frequency=freq, ep_r=eps0_rel + 0j)

# medio 1: Freespace con eps_r complejo para incluir pérdidas
ep_complex1 = np.where(
    f > 0,
    eps1 - 1j * (cond1 / (omega * eps0_vac)),
    eps1 + 0j
)
med1 = Freespace(frequency=freq, ep_r=ep_complex1)

# 6) Creamos antes del slab una sección libre (offset) de longitud 2.0
offset = 2.0    # x donde empieza el slab
line0  = med0.line(d=offset, unit='m')

# 7) Línea interna del slab
thickness = 0.2   # grosor del slab
line1     = med1.line(d=thickness, unit='m')

# 8) Interfaces (salto de impedancias)
Z0  = med0.z0
Z1  = med1.z0
t01 = med0.impedance_mismatch(Z0, Z1)
t10 = med1.impedance_mismatch(Z1, Z0)

# 9) Montaje de la red completa: espacio libre → entrada slab → slab → salida slab
network = line0 ** t01 ** line1 ** t10

# 10) Extraemos S₁₁ y S₂₁
S11 = network.s[:,0,0]
S21 = network.s[:,1,0]

plt.plot(f, np.abs(S11), label='|S11|')
plt.plot(f, np.abs(S21), label='|S21|')
plt.xlabel('Frecuencia (Hz)')
plt.legend()
plt.show()

# 11) Reconstrucción del espectro completo aplicando S-parámetros
P_ref_full = np.zeros_like(P_full, dtype=complex)
P_trn_full = np.zeros_like(P_full, dtype=complex)
idx_pos    = np.where(pos)[0]

P_ref_full[idx_pos] = P * S11
P_trn_full[idx_pos] = P * S21
for i in idx_pos[1:]:
    j = (-i) % N
    P_ref_full[j] = np.conj(P_ref_full[i])
    P_trn_full[j] = np.conj(P_trn_full[i])

# 12) IFFT de vuelta al dominio del tiempo
pulse_ref = np.real(np.fft.ifft(P_ref_full))
pulse_trn = np.real(np.fft.ifft(P_trn_full))

# 13) Cálculo de reflectancia R y transmitancia T
E_inc = np.sum(np.abs(pulse)**2)
E_ref = np.sum(np.abs(pulse_ref)**2)
E_trn = np.sum(np.abs(pulse_trn)**2)

R = E_ref / E_inc
T = E_trn / E_inc

print(f"Reflectance R = {R:.6f}")
print(f"Transmittance T = {T:.6f}")
print(f"Comprobación R + T = {R + T:.6f}")

A_inc = np.max(np.abs(pulse))
A_ref = np.max(np.abs(pulse_ref))
A_trn = np.max(np.abs(pulse_trn))

r_amp = A_ref / A_inc
t_amp = A_trn / A_inc

print(f"Coeficiente de amplitud r = {r_amp:.6f}")
print(f"Coeficiente de amplitud t = {t_amp:.6f}")
print(f"Equivalente R_amp^2 = {r_amp**2:.6f},  T_amp^2 = {t_amp**2:.6f}")

# 14) Gráfica final
plt.figure(figsize=(8,5))
plt.plot(t,    pulse,     label='Incidente')
plt.plot(t,    pulse_ref, '--', label='Reflejado')
plt.plot(t,    pulse_trn, ':',  label='Transmitido')
plt.xlabel('t (unidades c=1)')
plt.ylabel('Amplitud')
plt.title('Pulso EM: reflexión y transmisión con slab en x=[2,2.2]')
plt.legend()
plt.tight_layout()
plt.show()
