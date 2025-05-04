import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import skrf as rf

# Constante c=1 en unidades naturalizadas
C0 = 1.0

def gaussian_pulse(t, t0, sigma):
    """Gaussian pulse in time."""
    return np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

def RT_coeffs_scikit(pulse, dt, eps0, cond0, eps1, cond1, thickness):
    """
    Compute R and T of a slab using scikit-rf in c=1, μ0=ε0=1.
    pulse     : array of samples (half-amplitude toward panel)
    dt        : the FDTD time step (which is 0.5*dx/C0)
    eps0, cond0: properties of incident medium
    eps1, cond1: properties of slab
    thickness : slab thickness in same units (i.e. retardo en s)
    """

    # --- 0) Recover true sampling for the pulse ---
    # dt was 0.5*dx/C0; the pulse really is sampled at dt_pulse=dx/C0=2*dt
    dt_pulse = 2 * dt

    # 1) FFT two-sided with correct dt_pulse
    N     = len(pulse)
    G_all = fft(pulse)
    f_all = fftfreq(N, dt_pulse)   # Hz naturalizados

    # 2) Keep only f>0
    pos   = f_all > 0
    f_pos = f_all[pos]
    G_pos = G_all[pos]
    omega = 2 * np.pi * f_pos
    omega[omega == 0] = 1e-12      # avoid div by zero

    # 3) Slab EM parameters
    mu    = 1.0
    er    = eps1 / eps0
    sigma = cond1

    # complex perm, prop constant, impedance
    eps_c = er - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(mu * eps_c)
    Zc    = np.sqrt(mu / eps_c)

    # 4) Build scikit-rf Frequency (Hz)
    freq_obj = rf.Frequency.from_f(f_pos, unit='hz')

    # 5) Homogeneous slab as delay line
    medium  = rf.media.DefinedGammaZ0(
        frequency=freq_obj,
        gamma=gamma,
        z0=Zc
    )
    section = medium.line(d=thickness, unit='s')

    # 6) Manual interface S-matrices
    rho1 = (Zc - 1)/(Zc + 1); tau1 = 2*Zc/(Zc + 1)
    rho2 = (1 - Zc)/(1 + Zc); tau2 = 2      /(1 + Zc)

    s_if1 = np.zeros((len(f_pos),2,2), complex)
    s_if2 = np.zeros_like(s_if1)
    s_if1[:,0,0], s_if1[:,0,1] = rho1, tau1
    s_if1[:,1,0], s_if1[:,1,1] = tau1, -rho1
    s_if2[:,0,0], s_if2[:,0,1] = rho2, tau2
    s_if2[:,1,0], s_if2[:,1,1] = tau2, -rho2

    if1 = rf.Network(s=s_if1, frequency=freq_obj)
    if2 = rf.Network(s=s_if2, frequency=freq_obj)

    # 7) Cascade interfaces + slab
    total = if1 ** section ** if2
    s11   = total.s[:,0,0]
    s21   = total.s[:,1,0]

    # Debug: plot S11/S21
    plt.figure()
    plt.plot(f_pos, np.abs(s11), label='|S11|')
    plt.plot(f_pos, np.abs(s21), label='|S21|')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.title('Slab S-parameters')
    plt.show()

    # 8) Spectral integration for R, T
    P_inc = np.abs(G_pos)**2
    P_ref = np.abs(s11 * G_pos)**2
    P_tra = np.abs(s21 * G_pos)**2

    R = np.sum(P_ref) / np.sum(P_inc)
    T = np.sum(P_tra) / np.sum(P_inc)

    print(f"Max|S11|={np.max(np.abs(s11)):.3f}, Max|S21|={np.max(np.abs(s21)):.3f}")
    print(f"R = {R:.3f}, T = {T:.3f}")

    return R, T

# ----------------------------
# Example with your test params
# ----------------------------
nx      = 1001
L       = 10.0
xE      = np.linspace(-L/2, L/2, nx)
xPanel  = 2.0
wPanel  = 0.2
x0      = 0.0
sigma   = 0.25

dx  = xE[1] - xE[0]
dt  = 0.5 * dx / C0    # same as in your test

# Generate the half-amplitude pulse vs “time” (mapping x→t)
pulse = gaussian_pulse(xE, x0, sigma) / 2

plt.figure()
plt.plot(xE, pulse)
plt.xlabel('t (or x/C0)')
plt.title('Incident Half-Pulse')
plt.show()

R, T = RT_coeffs_scikit(
    pulse=pulse, dt=dt,
    eps0=1.0, cond0=0.0,
    eps1=5.0, cond1=1.0,
    thickness=wPanel
)

print("Final R, T:", R, T)
