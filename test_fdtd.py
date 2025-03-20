import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def fdtd_1d_solver(initial_condition, xE, dt, Tf, bounds=('pec', 'pec')):
    nxE = len(initial_condition)
    xH = (xE[:1] + xE[1:]) / 2.0
    dx = xE[1] - xE[0]

    e = np.zeros_like(initial_condition)
    h = np.zeros_like(xH)

    e[:] = initial_condition[:]

    e_old = np.zeros(2) 
    e_old[0] = e[0]  
    e_old[1] = e[-1]  

    for n in range(0, int(Tf / dt)):
        h[:] = h[:] - dt / dx / MU0 * (e[1:] - e[:-1])

        e[1:-1] = e[1:-1] - dt / dx / EPS0 * (h[1:] - h[:-1])

        if (bounds[0] == 'pec'):
            e[0] = 0.0
        elif (bounds[0] == 'mur'):
            e[0] = e[1] + (dt - dx)/(dt + dx)*(e[1] - e_old[0]) 
            e_old[0] = e[0] 
        else:
            raise ValueError("Unknown boundary condition: {}".format(bounds[0]))

        if (bounds[1] == 'pec'):
            e[-1] = 0.0
        elif (bounds[1] == 'mur'):
            e[-1] = e[-2] + (dt - dx)/(dt + dx)*(e[-2] - e_old[1])
            e_old[1] = e[-1] 
        else:
            raise ValueError("Unknown boundary condition: {}".format(bounds[1]))

        # For debugging
        # plt.plot(xE, e, '.-')
        # plt.ylim(-1, 1)
        # plt.pause(0.01)
        # plt.cla()


    return e

def test_fdtd_1d_solver_basic_propagation():
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

def test_fdtd_1d_solver_pec():
    nx = 101
    xE = np.linspace(-1, 1, nx)
    x0 = 0.0
    sigma = 0.1
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 2

    initial_condition = gaussian_pulse(xE, x0, sigma)
    final_condition = fdtd_1d_solver(initial_condition, xE, dt, Tf)

    expected_condition = -initial_condition

    assert np.corrcoef(final_condition, expected_condition)[0,1] >= 0.99


def test_fdtd_mur_conditions():
    nx = 101
    xE = np.linspace(-5, 5, nx)
    x0 = 0.5
    sigma = 0.25
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 100

    initial_e = gaussian_pulse(xE, x0, sigma)
    solved_e = fdtd_1d_solver(initial_e, xE, dt, Tf, ('mur', 'mur'))

    # Assert that the maximum e is less than 0.01
    assert np.max(solved_e) < 0.01, "Test failed: Maximum e after 4000 steps is greater than or equal to 0.01"
