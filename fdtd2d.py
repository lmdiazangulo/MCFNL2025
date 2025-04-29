import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.condx = np.zeros_like(self.xE)  # Default conductivity is 0 everywheree
        self.condy = np.zeros_like(self.yE)
        self.condPMLx=np.zeros_like(self.xE) # Fake PML magnetic conductivty
        self.condPMLy=np.zeros_like(self.yE)

        # TE mode fields: Hz (magnetic field in z), Ex (electric field in x), Ey (electric field in y)
        self.Hzx = np.zeros((self.nx, self.ny))
        self.condPMLx = 1*self.Hzx
        self.Hzy = np.zeros((self.nx, self.ny))
        self.condPMLy = 1*self.Hzy
        self.Ex = np.zeros((self.nx, self.ny-1))
        self.condy = 1*self.Ex
        self.Ey = np.zeros((self.nx-1, self.ny))
        self.condx = 1*self.Ey
        self.Hz = np.zeros((self.nx, self.ny))
        self.initialized = False

    def set_initial_condition(self, initial_condition):
        self.Hz[:] = initial_condition[:, :]
        self.Hzx[:] = initial_condition[:, :]
        self.Hzy[:] = initial_condition[:, :]
        self.initialized = True

    def set_PML(self, thicknessPML, m, R0, dx):
        sigmaMax = (-np.log(R0) * (m + 1)) / (2 * thicknessPML * dx)
        
        for i in range(1, thicknessPML):
            sigmai = sigmaMax * ((thicknessPML - i) / thicknessPML) ** m
            right_index= int(len(self.condPMLx)-1-i)
            # Apply PML to the X-axis boundaries
            self.condPMLx[i, :] += sigmai
            self.condPMLx[right_index, :] += sigmai
            self.condx[i, :] += sigmai
            self.condx[right_index, :] += sigmai

            # Apply PML to the Y-axis boundaries
            self.condPMLy[:, i] += sigmai
            self.condPMLy[:, right_index] += sigmai
            self.condy[:, i] += sigmai
            self.condy[:, right_index] += sigmai

    def step(self, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        # Update electric field
        self.Ex[:, :] = ( 1 / (2 * EPS0 + self.condy * dt) ) *  ( (2 * EPS0 - self.condy * dt) * self.Ex[:, :] + ( ( 2 * dt ) / \
            self.dy ) * (self.Hz[:, 1:] - self.Hz[:, :-1]) )
        self.Ey[:, :] = ( 1 / (2 * EPS0 + self.condx * dt) ) * ( (2 * EPS0 - self.condx * dt) * self.Ey[:, :] - ( ( 2 * dt ) / \
            self.dx ) * (self.Hz[1:, :] - self.Hz[:-1, :]) )

        # Update magnetic field
        self.Hzx[1:-1, 1:-1] = ( 1 / (2 * MU0 + self.condPMLx[1:-1, 1:-1] * dt) ) * ( (2 * MU0 - self.condPMLx[1:-1, 1:-1] * dt) * self.Hzx[1:-1, 1:-1] - ( (2*dt) / self.dx ) * 
            (self.Ey[1:, 1:-1] - self.Ey[:-1, 1:-1]) )
        self.Hzy[1:-1, 1:-1] = ( 1 / (2 * MU0 + self.condPMLy[1:-1, 1:-1] * dt) ) * ( (2 * MU0 - self.condPMLy[1:-1, 1:-1] * dt) * self.Hzy[1:-1, 1:-1] + ( (2*dt) / self.dy ) *
            (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1]) )

        # Combine magnetic fields
        self.Hz = self.Hzx + self.Hzy

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
    
    # Added function to visualize the simulation
    def simulate_and_plot(self, Tf, dt, interval=10):
        """
        Simulates the evolution of the Hz field and visualizes it using matplotlib.

        Parameters:
        - Tf: Final simulation time.
        - dt: Time step.
        - interval: Number of steps between frames in the animation.
        """
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        n_steps = int(Tf / dt)

        fig, ax = plt.subplots()
        cax = ax.imshow(self.Hz, cmap='viridis', origin='lower', extent=[self.xE[0], self.xE[-1], self.yE[0], self.yE[-1]])
        fig.colorbar(cax, ax=ax)
        ax.set_title("Hz Field Evolution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(frame):
            for _ in range(interval):
                self.step(dt)
            cax.set_array(self.Hz)
            return cax,

        ani = animation.FuncAnimation(fig, update, frames=n_steps // interval, blit=True)
        plt.show()