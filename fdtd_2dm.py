import numpy as np
import matplotlib.pyplot as plt

# Constants
MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0 * EPS0)  # Speed of light in vacuum

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def fdtd_2d_solver(nx, ny, zE, dx, dy, dt, Tf):
    Ez = np.copy(zE)
    Hx = np.zeros((nx, ny))
    Hy = np.zeros((nx, ny))

    for n in range(int(Tf / dt)):
        Hx[:, :-1] -= dt / dy * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += dt / dx * (Ez[1:, :] - Ez[:-1, :])

        Ez[1:-1, 1:-1] += (dt / dx) * ((Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) -
                                       (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]))

        Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0

    return Ez

def test_fdtd_2d_solver():
    nx, ny = 100, 100
    x0 = 0.0
    sigma = 0.25

    dx = dy = 1.0
    dt = 0.5 * dx / C0
    Tf = 2.0

    x = np.linspace(-5, 5, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    initial_condition = gaussian_pulse(X, x0, sigma)

    final_condition = fdtd_2d_solver(nx, ny, initial_condition, dx, dy, dt, Tf)

    expected_condition = 0.5 * gaussian_pulse(X, x0 - C0 * Tf, sigma) + \
                         0.5 * gaussian_pulse(X, x0 + C0 * Tf, sigma)

    correlation = np.corrcoef(final_condition.flatten(), expected_condition.flatten())[0,1]

    print(f"Test Correlation: {correlation:.5f}")
    assert correlation >= 0.99, "Test failed! The solution does not match the expected behavior."

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(initial_condition, cmap="RdBu", origin="lower")
    plt.colorbar(label="Ez field")
    plt.title("Initial Condition")

    plt.subplot(1, 2, 2)
    plt.imshow(final_condition, cmap="RdBu", origin="lower")
    plt.colorbar(label="Ez field")
    plt.title("Final Condition After FDTD")

    plt.show()

if __name__ == "__main__":
    test_fdtd_2d_solver()

