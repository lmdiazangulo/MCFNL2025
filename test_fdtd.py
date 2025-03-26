import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd import fdtd_1d_solver, fdtd_2d_solver, gaussian_pulse, C0


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
    final_condition = fdtd_1d_solver(initial_condition, xE, dt, Tf)

    expected_condition = -initial_condition

    assert np.corrcoef(final_condition, expected_condition)[0, 1] >= 0.99


@pytest.mark.skip("Does not work")
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
    assert np.max(
        solved_e) < 0.01, "Test failed: Maximum e after 4000 steps is greater than or equal to 0.01"


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

    # Solve in 2D
    Hz_final = fdtd_2d_solver(initial_condition_x, xE, yE, dt, Tf)

    # Extract section in X (y = 0)
    mid_y = ny // 2
    section_x = Hz_final[:, mid_y]

    # Expected solution in 1D for X
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

    # Validate coherence in X
    assert np.corrcoef(section_x, expected_x)[
        0, 1] >= 0.99, "Profile in X does not match 1D solution"

def test_fdtd_2d_solver_band_y():
    # Test 2: Gaussian band in Y (constant in X)
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

    # Solve in 2D
    Hz_final_y = fdtd_2d_solver(initial_condition_y, xE, yE, dt, Tf)

    # Extract section in Y (x = 0)
    mid_x = nx // 2
    section_y = Hz_final_y[mid_x, :]

    # Expected solution in 1D for Y
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

    # Validate coherence in Y
    assert np.corrcoef(section_y, expected_y)[
        0, 1] >= 0.99, "Profile in Y does not match 1D solution"
  

if __name__ == "__main__":
    pytest.main([__file__])
