import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D, gaussian_pulse, C0, EPS0, EPS1, R, T, C1


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

def test_fdtd_pmc_conditions():
    nx = 101
    xE = np.linspace(-5, 5, nx)
    x0 = 0.0
    sigma = 0.5
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf =np.max(xE) - np.min(xE)/ C0
    

    initial_e = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE, bounds=('pmc', 'pmc'))
    solver.set_initial_condition(initial_e)
    solved_e = solver.run_until(Tf, dt)

    expected_condition = initial_e

    assert np.corrcoef(solved_e, expected_condition)[0,1] >= 0.99

def test_fdtd_periodic_conditions():
    nx = 101
    xE = np.linspace(-5, 5, nx)
    x0 = 2.0
    sigma = 0.5
    
    dx = xE[1] - xE[0]
    dt = 0.5 * dx / C0
    Tf =((np.max(xE) - np.min(xE))/2-x0)/C0
    

    initial_e = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE, bounds=('periodic', 'periodic'))
    solver.set_initial_condition(initial_e)
    solved_e = solver.run_until(Tf, dt)

    assert (solved_e[-1]) != 0.0 and (solved_e[-1]) <0.75, "Test failed"




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

    rms = np.sqrt(np.mean(solved_e**2))
    assert rms < 0.01


def test_fdtd_1d_solver_permittivity():
    """Test FDTD solver with different permittivity regions."""
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

def test_fdtd_1d_solver_energy():
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

    tolerance = 0.01

    for i in range(len(solver.energyE)):
        assert ( solver.energy[0] - solver.energy[i] ) < tolerance
    

    plt.plot(solver.energyE, label='Energy E')
    plt.plot(solver.energyH, label='Energy H')
    plt.xlabel('Time step')
    plt.ylabel('Energy')
    plt.title('Energy in FDTD Simulation')
    plt.show()


def test_fdtd_1d_solver_energy():
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

    tolerance = 0.01

    for i in range(len(solver.energyE)):
        assert ( solver.energy[0] - solver.energy[i] ) < tolerance
    

    #plt.plot(solver.energyE, label='Energy E')
    #plt.plot(solver.energyH, label='Energy H')
    #plt.xlabel('Time step')
    #plt.ylabel('Energy')
    #plt.title('Energy in FDTD Simulation')
    #plt.show()


def test_fdtd_1d_solver_conductivity():
    """Test FDTD solver with a unique conductivity region."""
    nx = 1001
    L = 10
    d = 2
    xE = np.linspace(-L/2, L/2, nx)
    x0 = 0
    sigma = 0.25
    cond1 = 0.5
    
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
    solver.set_conductivity_regions([(-L/2, L/2, cond1)])
    
    solver.run_until(Tf, dt)

    for i in range(len(solver.energyE)):
        assert solver.energy[0] >= solver.energy[i]


if __name__ == "__main__":
    pytest.main([__file__]) 
