import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)


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

        return self.Hz 