import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D, gaussian_pulse, sigmoid_grid, C0, EPS0, EPS1, R, T, C1


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
    final_condition = solver.run_until(Tf=Tf, dt=dt)

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
    final_condition = solver.run_until(Tf=Tf, dt=dt)

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
    solved_e = solver.run_until(Tf=Tf, dt=dt)

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
    solved_e = solver.run_until(Tf,dt)

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

    final_condition = solver.run_until(Tf=Tf, dt=dt)

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
    final_condition = solver.run_until(Tf=Tf, dt=dt)

    tolerance = 0.01

    for i in range(len(solver.energyE)):
        assert ( solver.energy[0] - solver.energy[i] ) < tolerance


    plt.plot(solver.energyE, label='Energy E')
    plt.plot(solver.energyH, label='Energy H')
    plt.xlabel('Time step')
    plt.ylabel('Energy')
    plt.title('Energy in FDTD Simulation')
    plt.show()

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

def test_fdtd_1d_solver_PML_ENERGY():
    """Test FDTD solver with a PML boundary conditions (for the energy)"""
    nx = 101
    L = 101
    xE = np.linspace(-L/2, L/2, nx)
    x0 = 0
    sigma = 3

    dx = xE[1] - xE[0]
    dt = 0.25 * dx / C0
    Tf = L

    #PML variables
    R0=1e-6
    m=2 # Steepness of the grading
    thicknessPML=10
    sigmaMax=(-np.log(R0)*(m+1))/(2*thicknessPML*dx)
    # Set initial conditions to a gaussian pulse
    initial_condition = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE, bounds=('pec','pec'))

    solver.set_initial_condition(initial_condition)
    # Set PML
    solver.set_PML(thicknessPML, m,sigmaMax)

    # Run the simulation
    solver.run_until(Tf, dt)
    # Check that the energy after the wave have passed through the PML is almost 0 in the whole grid
    assert solver.energy[-1]/solver.energy[0] < 0.05

def test_fdtd_1d_solver_PML_R_COEFFICIENT():
    """Test FDTD solver with a PML boundary conditions (for the R coefficient)"""
    nx = 101
    L = 101
    xE = np.linspace(-L/2, L/2, nx)
    x0 = 0
    sigma = 3

    dx = xE[1] - xE[0]
    dt = 0.25 * dx / C0
    Tf = L

    #PML variables
    R0=1e-6
    m=2 # Steepness of the grading
    thicknessPML=10
    sigmaMax=(-np.log(R0)*(m+1))/(2*thicknessPML*dx)
    # Set initial conditions to a gaussian pulse
    initial_condition = gaussian_pulse(xE, x0, sigma)
    solver = FDTD1D(xE, bounds=('pec','pec'))

    solver.set_initial_condition(initial_condition)
    Ei=np.max(np.abs(solver.e))
    # Set PML
    solver.set_PML(thicknessPML, m,sigmaMax)

    # Run the simulation
    solver.run_until(Tf, dt)
    Er=np.max(np.abs(solver.e))
    # Check that the energy after the wave have passed through the PML is almost 0 in the whole grid
    assert np.abs(Er/Ei)**2<0.01

def test_fdtd_1d_total_scattered_field():
    x = np.linspace(0, 300, 500)
    sim = FDTD1D(x, bounds=('pec', 'mur'))
    DT = 0.5 * sim.avgdxE / C0

    def source_test(x,t):
        return 0.5*np.exp(-(x-C0*t)**2 / (20)**2 )

    sim.add_totalfield(
    100,
    source_test)
    sim.set_initial_condition(np.zeros_like(x))

    final_condition = sim.run_until(Tf=1200*DT, dt=DT)
    expected_condition = np.zeros_like(final_condition)

    # plt.plot(sim.xE,final_condition)
    # plt.plot(sim.xE,expected_condition)
    # plt.show()
    assert np.allclose(final_condition, expected_condition,atol = 1e-2), "Total field not properly injected"

def test_fdtd_1d_solver_probe():
    '''Test FDTD solver with a probe in void and mur conditions'''
    x = np.linspace(0, 300, 500)
    sim = FDTD1D(x, bounds=('mur', 'mur'))
    DT = 0.8 * sim.avgdxE / C0
    TF = (200)/C0
    t = np.linspace(0,TF,round(TF/DT)-1)
    x_probe = t*C0

    def initial_condition(x,x0):
        return 0.5*np.exp(-(x-x0)**2 / (20)**2 )
    
    sim.set_initial_condition(initial_condition(x,50))
    sim.add_probe([150])
    sim.run_until(Tf = TF,dt = DT)

    measured_e = sim.e_measure[0]
    expected_e = 0.5*initial_condition(x_probe,100) # Expected shape. It takes into account that that the initial gaussian is splitted in two
    
    '''
    plt.clf()
    plt.plot(t,measured_e, label = f"Probe")
    plt.plot(x_probe,expected_e,label = "Initial Condition")
    plt.legend()
    plt.show()
    '''

    assert np.corrcoef(measured_e, expected_e)[0,1] >= 0.99, "Field not correctly measured"

def test_fdtd_1d_nonuniform_grid():
    nx = 101
    xE_nonuniform = sigmoid_grid(xmin=-1,xmax=1,npoints=nx)
    x0 = 0.0
    sigma = 0.1
    Tf = 2

    initial_condition = gaussian_pulse(xE_nonuniform, x0, sigma)

    # Two tests, one for mur and one for pec

    mursolver = FDTD1D(xE_nonuniform, bounds=('mur', 'mur'))
    mursolver.set_initial_condition(initial_condition)
    final_mur_condition = mursolver.run_until(Tf=Tf)

    pecsolver = FDTD1D(xE_nonuniform, bounds=('pec', 'pec'))
    pecsolver.set_initial_condition(initial_condition)
    final_pec_condition = pecsolver.run_until(Tf=Tf)

    # Test if mur and pec behave well
    tolerance = 1e-3
    expected_pec_condition = -initial_condition
    assert np.max(np.abs(final_mur_condition)) < tolerance and np.corrcoef(final_pec_condition, expected_pec_condition)[0,1] >= 0.95

def test_fdtd_1d_tfsf():
    xE = np.linspace(-5,5,501)

    tfsf_start = -1
    tfsf_end = 1
    tfsf_x0 = -3.0  # So that it takes some time to get to the tf region
    tfsf_sigma = 0.1
    tfsf_function = gaussian_pulse(xE, tfsf_x0, tfsf_sigma)

    T_before = 1
    T_after = 10 # Two times so that the field is tested to be null both before the arrival of the tfsf function and after

    initial_condition = np.zeros_like(xE)
    solver = FDTD1D(xE, bounds=('mur', 'mur'))
    solver.set_initial_condition(initial_condition)
    solver.set_tfsf_conditions(tfsf_start, tfsf_end, tfsf_function)

    final_condition_before_arrival = solver.run_until(Tf=T_before)
    final_condition_after_arrival = solver.run_until(Tf=T_after)

    tolerance = 0.01
    assert np.max(np.abs(final_condition_before_arrival)) < tolerance and np.max(np.abs(final_condition_after_arrival)) < tolerance


if __name__ == "__main__":
    pytest.main([__file__])
