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

        # # For debugging
        # plt.plot(xE, e, '.-')
        # plt.ylim(-1, 1)
        # plt.pause(0.01)
        # plt.cla()

    return e

def fdtd_2d_solver(initial_condition, xE, yE, dt, Tf):
    """
    2D FDTD solver in TE mode (Hz, Ex, Ey)
    Uses PEC boundary conditions
    """
    nx = len(xE)
    ny = len(yE)
    dx = xE[1] - xE[0]
    dy = yE[1] - yE[0]
    
    # TE mode fields: Hz (magnetic field in z), Ex (electric field in x), Ey (electric field in y)
    Hz = np.zeros((nx, ny))  # Hz at integer grid points
    Ex = np.zeros((nx, ny-1))  # Ex at half grid points in y
    Ey = np.zeros((nx-1, ny))  # Ey at half grid points in x

    Hz[:] = initial_condition[:, :]

    n_steps = int(Tf / dt)
    for n in range(n_steps):
        # Update Ex (electric field in x)
        Ex[:, :] = Ex[:, :] + dt / EPS0 / dy * (Hz[:, 1:] - Hz[:, :-1])
        
        # Update Ey (electric field in y)
        Ey[:, :] = Ey[:, :] - dt / EPS0 / dx * (Hz[1:, :] - Hz[:-1, :])
        
        # Update Hz (magnetic field in z)
        Hz[1:-1, 1:-1] = Hz[1:-1, 1:-1] + dt / MU0 * (
            (Ex[1:-1, 1:] - Ex[1:-1, :-1]) / dy - (Ey[1:, 1:-1] - Ey[:-1, 1:-1]) / dx
        )

        # PEC boundary conditions for Hz
        Hz[0, :] = 0.0  # Left boundary
        Hz[-1, :] = 0.0  # Right boundary
        Hz[:, 0] = 0.0  # Bottom boundary
        Hz[:, -1] = 0.0  # Top boundary

    return Hz 