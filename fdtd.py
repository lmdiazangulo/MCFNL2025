import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

# Constants for permittivity regions test
EPS1 = 2.0
C1 = 1 / np.sqrt(MU0*EPS1)
R = (np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
T = 2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


class FDTD1D:
    def __init__(self, xE, bounds=('pec', 'pec')):
        self.xE = np.array(xE)
        self.xH = (self.xE[:1] + self.xE[1:]) / 2.0
        self.dx = self.xE[1] - self.xE[0]
        self.bounds = bounds
        self.e = np.zeros_like(self.xE)
        self.h = np.zeros_like(self.xH)
        self.eps = np.ones_like(self.xE)  # Default permittivity is 1 everywhere
        self.initialized = False

    def set_initial_condition(self, initial_condition):
        self.e[:] = initial_condition[:]
        self.initialized = True

    def set_permittivity_regions(self, regions):
        """Set different permittivity regions in the grid.
        
        Args:
            regions: List of tuples (start_x, end_x, eps_value) defining regions
                    with different permittivity values.
        """
        for start_x, end_x, eps_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)
            self.eps[start_idx:end_idx] = eps_value

    def step(self, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        self.e_old_left = self.e[1]
        self.e_old_right = self.e[-2]

        self.h[:] = self.h[:] - dt / self.dx / MU0 * (self.e[1:] - self.e[:-1])
        self.e[1:-1] = self.e[1:-1] - dt / self.dx / self.eps[1:-1] * (self.h[1:] - self.h[:-1])

        if self.bounds[0] == 'pec':
            self.e[0] = 0.0
        elif self.bounds[0] == 'mur':
            self.e[0] = self.e_old_left + (C0*dt - self.dx) / \
                (C0*dt + self.dx)*(self.e[1] - self.e[0])
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[0]}")

        if self.bounds[1] == 'pec':
            self.e[-1] = 0.0
        elif self.bounds[1] == 'mur':
            self.e[-1] = self.e_old_right + (C0*dt - self.dx) / \
                (C0*dt + self.dx)*(self.e[-2] - self.e[-1])
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[1]}")

    def run_until(self, Tf, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        n_steps = int(Tf / dt)
        for _ in range(n_steps):
            self.step(dt)

            # # For debugging
            # plt.plot(self.xE, self.e, '.-')
            # plt.ylim(-1, 1)
            # plt.pause(0.01)
            # plt.cla()

        return self.e


class FDTD2D:
    def __init__(self, xE, yE):
        self.xE = np.array(xE)
        self.yE = np.array(yE)
        self.nx = len(xE)
        self.ny = len(yE)
        self.dx = xE[1] - xE[0]
        self.dy = yE[1] - yE[0]

        # TE mode fields: Hz (magnetic field in z), Ex (electric field in x), Ey (electric field in y)
        self.Hz = np.zeros((self.nx, self.ny))
        self.Ex = np.zeros((self.nx, self.ny-1))
        self.Ey = np.zeros((self.nx-1, self.ny))
        self.initialized = False

    def set_initial_condition(self, initial_condition):
        self.Hz[:] = initial_condition[:, :]
        self.initialized = True

    def step(self, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        self.Ex[:, :] = self.Ex[:, :] + dt / EPS0 / \
            self.dy * (self.Hz[:, 1:] - self.Hz[:, :-1])
        self.Ey[:, :] = self.Ey[:, :] - dt / EPS0 / \
            self.dx * (self.Hz[1:, :] - self.Hz[:-1, :])

        self.Hz[1:-1, 1:-1] = self.Hz[1:-1, 1:-1] + dt / MU0 * (
            (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1]) / self.dy -
            (self.Ey[1:, 1:-1] - self.Ey[:-1, 1:-1]) / self.dx
        )

        # PEC boundary conditions
        self.Hz[0, :] = 0.0
        self.Hz[-1, :] = 0.0
        self.Hz[:, 0] = 0.0
        self.Hz[:, -1] = 0.0

    def run_until(self, Tf, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        n_steps = int(Tf / dt)
        for _ in range(n_steps):
            self.step(dt)

            # # For debugging
            # plt.imshow(self.Hz, extent=[self.yE[0], self.yE[-1], self.xE[0], self.xE[-1]],
            #           aspect='equal', cmap='RdBu')
            # plt.colorbar()
            # plt.pause(0.01)
            # plt.cla()

        return self.Hz
