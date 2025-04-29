import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)
COND0 = 0.0

# Constants for permittivity regions test
EPS1 = 1.0
C1 = 1 / np.sqrt(MU0*EPS1)
COND1 = 3.0
R = (np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
T = 2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


class FDTD1D:
    def __init__(self, xE, bounds):
        self.xE = np.array(xE)
        self.xH = (self.xE[:1] + self.xE[1:]) / 2.0
        self.dx = self.xE[1] - self.xE[0]
        self.bounds = bounds
        self.e = np.zeros_like(self.xE)
        self.h = np.zeros_like(self.xH)
        self.eps = np.ones_like(self.xE)  # Default permittivity is 1 everywhere
        self.cond = np.zeros_like(self.xE)
        self.initialized = False
        #We define the regions for the permittivity and conductivity. We define the x position of the start of the region and its width.

        self.regions = {
            'eps': [(0.7, 0.72, EPS1)],
            'cond': [(0.7, 0.72, COND1)]
        }
        # Set the permittivity and conductivity regions
        self.set_permittivity_regions(self.regions['eps'])
        self.set_conductivity_regions(self.regions['cond'])

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

    def set_conductivity_regions(self, regions):
        """Set different conductivity regions in the grid.
        
        Args:
            regions: List of tuples (start_x, end_x, cond_value) defining regions
                    with different conductivity values.
        """
        for start_x, end_x, cond_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)
            self.cond [start_idx:end_idx] = cond_value

    def step(self, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        self.e_old_left = self.e[1]
        self.e_old_right = self.e[-2]
        
        #self.h[:] = self.h[:] - dt / self.dx / MU0 * (self.e[1:] - self.e[:-1]) + self.cond[1:] * dt / 2 * (self.e[1:] + self.e[:-1])

        self.h[:] = self.h[:] - dt / self.dx / MU0 * (self.e[1:] - self.e[:-1])
        self.e[1:-1] = (self.eps[1:-1] - self.cond[1:-1] * dt/2) / (self.eps[1:-1] + self.cond[1:-1] * dt/2) * self.e[1:-1] - dt / self.dx / (self.eps[1:-1]+self.cond[1:-1]*dt/2) * (self.h[1:] - self.h[:-1])

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
            raise RuntimeError("Initial condition not set.")
        
        plt.ion()  # Activar modo interactivo
        fig, ax = plt.subplots()
        line, = ax.plot(self.xE, self.e)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('E field')
        # Lets put a vertical line at the boundary of the region with different cond.
        for start_x, end_x, eps_value in self.regions['cond']:
            ax.axvline(x=start_x, color='r', linestyle='--', label='Boundary')
            ax.axvline(x=end_x, color='r', linestyle='--')
        ax.set_title('FDTD 1D Simulation')
        ax.legend() 

        n_steps = int(Tf / dt)
        for _ in range(n_steps):
            self.step(dt)
            line.set_ydata(self.e)
            plt.pause(0.01)

        plt.ioff()
        plt.show()

# Now lets save the results of the simulation with the different regions.
# We will use the gaussian pulse as initial condition. The gaussian pulse is defined by its center and width.
def main():
    xE = np.linspace(0, 1, 200)
    fdtd = FDTD1D(xE, bounds=('mur', 'mur'))
    x0 = 0.5
    sigma = 0.05
    initial_condition = gaussian_pulse(xE, x0, sigma)
    fdtd.set_initial_condition(initial_condition)

    dt = 0.005
    Tf = 2.0
    fdtd.run_until(Tf, dt)

main()