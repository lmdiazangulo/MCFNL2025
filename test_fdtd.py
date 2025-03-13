import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))



# x = np.linspace(0, 100, 1001)
# plt.plot(x, gaussian_pulse(x, 50, 10))

def fdtd_1d_solver(initial_condition, xE, dt, Tf):
    nxE = len(initial_condition)
    xH = (xE[:1] + xE[1:]) / 2.0
    dx = xE[1] - xE[0]

    e = np.zeros_like(initial_condition)
    h = np.zeros_like(xH)

    e[:] = initial_condition[:]

    for n in range(0, int(Tf / dt)):
        h[:] = h[:] - dt / dx / MU0 * (e[1:] - e[:-1])

        e[0] = 0.0
        e[-1] = 0.0
        e[1:-1] = e[1:-1] - dt / dx / EPS0 * (h[1:] - h[:-1])

    return e

def test_fdtd_1d_solver():
    nx = 101
    xE = np.linspace(-5, 5, nx)
    

    x0 = 0.5
    sigma = 0.25
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 2

    initial_condition = gaussian_pulse(xE, x0, sigma)
    final_condition = fdtd_1d_solver(initial_condition, xE, dt, Tf)

    expected_condition = \
        0.5 * gaussian_pulse(xE, x0 - C0 * Tf, sigma) + \
        0.5 * gaussian_pulse(xE, x0 + C0 * Tf, sigma)

    assert np.corrcoef(final_condition, expected_condition)[0,1] >= 0.99

if __name__ == "__main__":
    test_fdtd_1d_solver()