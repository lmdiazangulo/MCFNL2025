import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

MU1 = 1.0
EPS1 = 2.0
C1 = 1 / np.sqrt(MU1*EPS1)

R=(np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
T=2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# x = np.linspace(0, 100, 1001)
# plt.plot(x, gaussian_pulse(x, 50, 10))

def fdtd_1d_solver(initial_condition, xE, dt, Tf, nx, L, d):
    nxE = len(initial_condition)
    xH = (xE[:1] + xE[1:]) / 2.0
    dx = xE[1] - xE[0]

    e = np.zeros_like(initial_condition)
    h = np.zeros_like(xH)

    e[:] = initial_condition[:]
    nx1 = int(nx/L*(L-d))
    for n in range(0, int(Tf / dt)):

        h[:nx1] = h[:nx1] - dt / dx / MU0 * (e[1:nx1+1] - e[:nx1])
        h[nx1:] = h[nx1:] - dt / dx / MU1 * (e[nx1 + 1:] - e[nx1:-1])

        e[0] = 0.0
        e[-1] = 0.0
        e[1:nx1] = e[1:nx1] - dt / dx / EPS0 * (h[1:nx1] - h[:nx1-1])
        e[nx1:-1] = e[nx1:-1] - dt / dx / EPS1 * (h[nx1:] - h[nx1-1:-1])

        #For debugging
        # plt.plot(xE, e)
        # plt.axvline(x=L/2-d, color='grey', linestyle='--')
        # plt.pause(0.001)
        # plt.cla()
    return e

def test_fdtd_1d_solver_permitivity():
    nx = 1001
    L=10
    d=2
    xE = np.linspace(-L/2, L/2, nx)
    nx1 = int(nx/L*(L-d))

    x0 = 0
    sigma = 0.25
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 4

    initial_condition = gaussian_pulse(xE, x0, sigma)
    final_condition = fdtd_1d_solver(initial_condition, xE, dt, Tf, nx, L, d )

    expected_condition = np.zeros(nx)
    expected_condition[:nx1] = \
        0.5 * gaussian_pulse(xE[:nx1], x0 - C0 * Tf, sigma) + \
        0.5 * R * gaussian_pulse(xE[:nx1], L/2-d - C0 * (Tf-(L/2-d)/C0) , sigma)
    expected_condition[nx1+1:] = \
        0.5 * T * gaussian_pulse(xE[nx1+1:], L/2-d + C1 * (Tf-(L/2-d)/C0), sigma*np.sqrt(EPS0/EPS1)) \


    plt.plot(xE, initial_condition, label='initial')
    plt.plot(xE, expected_condition, label='expected')
    plt.plot(xE, final_condition, label='final')
    plt.axvline(x=L/2-d, color='grey', linestyle='--')
    plt.legend()
    plt.show()

    assert np.corrcoef(final_condition, expected_condition)[0,1] >= 0.99

if __name__ == "__main__":
    test_fdtd_1d_solver_permitivity()
