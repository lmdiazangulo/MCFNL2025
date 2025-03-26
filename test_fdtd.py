import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd import FDTD1D, FDTD2D, gaussian_pulse, C0


def test_fdtd_1d_solver_basic_propagation():
    nx = 101
    xE = np.linspace(-5, 5, nx)
    x0 = 0.5
    sigma = 0.25

    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 2

    initial_condition = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE)
    solver.set_initial_condition(initial_condition)
    final_condition = solver.run_until(Tf, dt)

    expected_condition = \
        0.5 * gaussian_pulse(xE, x0 - C0 * Tf, sigma) + \
        0.5 * gaussian_pulse(xE, x0 + C0 * Tf, sigma)

    assert np.corrcoef(final_condition, expected_condition)[0, 1] >= 0.99


def test_fdtd_1d_solver_pec():
    nx = 101
    xE = np.linspace(-1, 1, nx)
    x0 = 0.0
    sigma = 0.1

    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 2

    initial_condition = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE, bounds=('pec', 'pec'))
    solver.set_initial_condition(initial_condition)
    final_condition = solver.run_until(Tf, dt)

    expected_condition = -initial_condition

    assert np.corrcoef(final_condition, expected_condition)[0, 1] >= 0.99


def test_fdtd_mur_conditions():
    """Test that Mur boundary conditions properly absorb the field."""
    nx = 101
    xE = np.linspace(-5, 5, nx)
    x0 = 0.5
    sigma = 0.25

    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 10

    initial_e = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE, bounds=('mur', 'mur'))
    solver.set_initial_condition(initial_e)
    solved_e = solver.run_until(Tf, dt)


    # Plot for debugging
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(xE, initial_e, 'b-', label='Initial')
    # ax.plot(xE, solved_e, 'r-', label='Final')
    # ax.set_title('Mur Boundary Conditions Test')
    # ax.set_xlabel('Position')
    # ax.set_ylabel('Field Amplitude')
    # ax.legend()
    # plt.grid(True)
    # plt.show()

    # The RMS should be very close to zero (less than 0.01)
    rms = np.sqrt(np.mean(solved_e**2))
    assert rms < 0.01


def test_fdtd_2d_solver_band_x():
    nx, ny = 101, 101
    xE = np.linspace(-5, 5, nx)
    yE = np.linspace(-5, 5, ny)

    x0, y0 = 0.0, 0.0
    sigma = 0.25

    dx = xE[1] - xE[0]
    dy = yE[1] - yE[0]
    dt = 0.5 * np.sqrt(dx**2 + dy**2) / C0
    Tf = 2

    initial_condition_x = np.zeros((nx, ny))
    for i in range(nx):
        gaussian_value = gaussian_pulse(xE[i], x0, sigma)
        for j in range(ny):
            initial_condition_x[i, j] = gaussian_value

    solver = FDTD2D(xE, yE)
    solver.set_initial_condition(initial_condition_x)
    Hz_final = solver.run_until(Tf, dt)

    mid_y = ny // 2
    section_x = Hz_final[:, mid_y]

    expected_x = \
        0.5 * gaussian_pulse(xE, x0 - C0 * Tf, sigma) + \
        0.5 * gaussian_pulse(xE, x0 + C0 * Tf, sigma)

    # # Plot for debugging
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(xE, section_x, label='Section in X (2D)')
    # ax.plot(xE, expected_x, '--', label='Expected in X (1D)')
    # ax.set_title('Comparison in X axis (band in X)')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    assert np.corrcoef(section_x, expected_x)[0, 1] >= 0.99


def test_fdtd_2d_solver_band_y():
    nx, ny = 101, 101
    xE = np.linspace(-5, 5, nx)
    yE = np.linspace(-5, 5, ny)

    x0, y0 = 0.0, 0.0
    sigma = 0.25

    dx = xE[1] - xE[0]
    dy = yE[1] - yE[0]
    dt = 0.5 * np.sqrt(dx**2 + dy**2) / C0
    Tf = 2

    initial_condition_y = np.zeros((nx, ny))
    for j in range(ny):
        gaussian_value = gaussian_pulse(yE[j], y0, sigma)
        for i in range(nx):
            initial_condition_y[i, j] = gaussian_value

    solver = FDTD2D(xE, yE)
    solver.set_initial_condition(initial_condition_y)
    Hz_final_y = solver.run_until(Tf, dt)

    mid_x = nx // 2
    section_y = Hz_final_y[mid_x, :]

    expected_y = 0.5 * gaussian_pulse(yE, y0 - C0 * Tf, sigma) + \
        0.5 * gaussian_pulse(yE, y0 + C0 * Tf, sigma)

    # # Plot for debugging
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(yE, section_y, label='Section in Y (2D)')
    # ax.plot(yE, expected_y, '--', label='Expected in Y (1D)')
    # ax.set_title('Comparison in Y axis (band in Y)')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    assert np.corrcoef(section_y, expected_y)[0, 1] >= 0.99

@pytest.mark.skip("Not really a test")
def test_fdtd_2d_solver_gaussian():
    """Test 2D FDTD solver with a 2D Gaussian pulse initial condition."""
    nx, ny = 101, 101
    xE = np.linspace(-5, 5, nx)
    yE = np.linspace(-5, 5, ny)

    x0, y0 = 0.0, 0.0
    sigma = 0.25

    dx = xE[1] - xE[0]
    dy = yE[1] - yE[0]
    dt = 0.5 * np.sqrt(dx**2 + dy**2) / C0
    Tf = 2

    initial_condition = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            initial_condition[i, j] = gaussian_pulse(xE[i], x0, sigma) * gaussian_pulse(yE[j], y0, sigma)

    solver = FDTD2D(xE, yE)
    solver.set_initial_condition(initial_condition)
    Hz_final = solver.run_until(Tf, dt)

    mid_x = nx // 2
    mid_y = ny // 2
    section_x = Hz_final[:, mid_y]
    section_y = Hz_final[mid_x, :]

    expected_x = 0.5 * gaussian_pulse(xE, x0 - C0 * Tf, sigma) + \
        0.5 * gaussian_pulse(xE, x0 + C0 * Tf, sigma)
    expected_y = 0.5 * gaussian_pulse(yE, y0 - C0 * Tf, sigma) + \
        0.5 * gaussian_pulse(yE, y0 + C0 * Tf, sigma)

    fig = plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(131)
    im1 = ax1.imshow(initial_condition, extent=[yE[0], yE[-1], xE[0], xE[-1]], 
                     aspect='equal', cmap='RdBu', origin='lower')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Initial Condition')
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    
    ax2 = plt.subplot(132)
    im2 = ax2.imshow(Hz_final, extent=[yE[0], yE[-1], xE[0], xE[-1]], 
                     aspect='equal', cmap='RdBu', origin='lower')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Final Condition')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')
    
    ax3 = plt.subplot(133)
    ax3.plot(xE, section_x, label='Section in X (2D)')
    ax3.plot(xE, expected_x, '--', label='Expected in X (1D)')
    ax3.plot(yE, section_y, label='Section in Y (2D)')
    ax3.plot(yE, expected_y, '--', label='Expected in Y (1D)')
    ax3.set_title('1D Sections')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_fdtd_1d_solver_permittivity():
    """Test FDTD solver with different permittivity regions."""
    MU0 = 1.0
    EPS0 = 1.0
    C0 = 1 / np.sqrt(MU0*EPS0)

    MU1 = 1.0
    EPS1 = 2.0
    C1 = 1 / np.sqrt(MU1*EPS1)

    R=(np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
    T=2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))

    nx = 1001
    L = 10
    d = 2
    xE = np.linspace(-L/2, L/2, nx)
    nx1 = int(nx/L*(L-d))

    x0 = 0
    sigma = 0.25
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf = 4

    initial_condition = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE)
    solver.set_initial_condition(initial_condition)
    
    # Set different permittivity regions
    solver.set_permittivity_regions([
        (-L/2, L/2-d, EPS0),  # First region with EPS0
        (L/2-d, L/2, EPS1)    # Second region with EPS1
    ])
    
    final_condition = solver.run_until(Tf, dt)

    expected_condition = np.zeros(nx)
    expected_condition[:nx1] = \
        0.5 * gaussian_pulse(xE[:nx1], x0 - C0 * Tf, sigma) + \
        0.5 * R * gaussian_pulse(xE[:nx1], L/2-d - C0 * (Tf-(L/2-d)/C0), sigma)
    expected_condition[nx1+1:] = \
        0.5 * T * gaussian_pulse(xE[nx1+1:], L/2-d + C1 * (Tf-(L/2-d)/C0), sigma*np.sqrt(EPS0/EPS1))

    assert np.corrcoef(final_condition, expected_condition)[0,1] >= 0.99

if __name__ == "__main__":
    pytest.main([__file__])
