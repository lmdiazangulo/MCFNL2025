import numpy as np
import matplotlib.pyplot as plt

def fdtd_simulation(nt):
    # Define simulation parameters
    L = 1  # Length of the box
    nx = 1001  # Number of grid points
    dx = L / nx  # Spatial step size
    dt = 0.5 * dx  # Time step size (Courant-Friedrichs-Lewy condition)
    eps_0 = 1 #Permittivity of free space
    mu_0 = 1 #Permeability of free space

    # Initialize fields and Ez_old
    Ez = np.exp(-(np.linspace(0, L, nx) - L / 2)*2 / (0.1 * L)*2)  # Gaussian pulse
    Hy = np.zeros(nx)  # Magnetic field
    Ez_old = np.zeros(2) 
    Ez_old[0] = Ez[0]  
    Ez_old[1] = Ez[-1]  

    # Implement the FDTD update equations
    for n in range(nt):
        # Update Hy field
        Hy[:-1] = Hy[:-1] + (dt / (mu_0 * dx)) * (Ez[1:] - Ez[:-1])

        # Update Ez field with Mur's ABC
        Ez[1:-1] = Ez[1:-1] + (dt / (eps_0 * dx)) * (Hy[1:-1] - Hy[:-2])
        
        # Mur's ABC at x = 0
        Ez[0] = Ez[1] + (dt - dx)/(dt + dx)*(Ez[1] - Ez_old[0]) 
        Ez_old[0] = Ez[0] 
        
        # Mur's ABC at x = L
        Ez[-1] = Ez[-2] + (dt - dx)/(dt + dx)*(Ez[-2] - Ez_old[1])
        Ez_old[1] = Ez[-1] 
        
    return Ez
    
def test_fdtd_damping():
    # Perform the simulation for 4000 steps
    Ez = fdtd_simulation(nt=4000)
    
    # Assert that the maximum Ez is less than 0.01
    assert np.max(Ez) < 0.01, "Test failed: Maximum Ez after 4000 steps is greater than or equal to 0.01"
