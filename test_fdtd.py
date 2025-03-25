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

        # For debugging
        plt.plot(xE, e, '.-')
        plt.ylim(-1, 1)
        plt.pause(0.01)
        plt.cla()

    return e


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

    assert np.corrcoef(final_condition, expected_condition)[0,1] >= 0.99

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

    assert np.corrcoef(final_condition, expected_condition)[0,1] >= 0.99


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
    assert np.max(solved_e) < 0.01, "Test failed: Maximum e after 4000 steps is greater than or equal to 0.01"


# MODO TMz (representamos Ez frente a Hx, Hy)
def fdtd_2d_solver(initial_condition, xE, yE, dt, Tf, bounds=('pec', 'pec')):
    nx = len(xE)
    ny = len(yE)
    dx = xE[1] - xE[0]
    dy = yE[1] - yE[0]
    
    Ez = np.zeros((nx, ny))
    Hx = np.zeros((nx, ny-1))
    Hy = np.zeros((nx-1, ny))

    Ez[:] = initial_condition[:, :]

    Ez_old_left = np.zeros(ny)
    Ez_old_right = np.zeros(ny)
    Ez_old_bottom = np.zeros(nx)
    Ez_old_top = np.zeros(nx)

    Ez_old_left[:] = Ez[0, :]
    Ez_old_right[:] = Ez[-1, :]
    Ez_old_bottom[:] = Ez[:, 0]
    Ez_old_top[:] = Ez[:, -1]

    n_steps = int(Tf / dt)
    for n in range(n_steps):
        Hx[:, :] = Hx[:, :] - dt / MU0 / dy * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:, :] = Hy[:, :] + dt / MU0 / dx * (Ez[1:, :] - Ez[:-1, :])

        Ez[1:-1, 1:-1] = Ez[1:-1, 1:-1] + dt / EPS0 * (
            (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        )

        if bounds[0] == 'pec':
            Ez[0, :] = 0.0
        elif bounds[0] == 'mur':
            Ez[0, :] = Ez[1, :] + (dt - dx)/(dt + dx)*(Ez[1, :] - Ez_old_left)
            Ez_old_left[:] = Ez[0, :]

        if bounds[1] == 'pec':
            Ez[-1, :] = 0.0
        elif bounds[1] == 'mur':
            Ez[-1, :] = Ez[-2, :] + (dt - dx)/(dt + dx)*(Ez[-2, :] - Ez_old_right)
            Ez_old_right[:] = Ez[-1, :]

        if bounds[0] == 'pec':
            Ez[:, 0] = 0.0
        elif bounds[0] == 'mur':
            Ez[:, 0] = Ez[:, 1] + (dt - dy)/(dt + dy)*(Ez[:, 1] - Ez_old_bottom)
            Ez_old_bottom[:] = Ez[:, 0]

        if bounds[1] == 'pec':
            Ez[:, -1] = 0.0
        elif bounds[1] == 'mur':
            Ez[:, -1] = Ez[:, -2] + (dt - dy)/(dt + dy)*(Ez[:, -2] - Ez_old_top)
            Ez_old_top[:] = Ez[:, -1]

    return Ez


def test_fdtd_2d_solver_basic_propagation():
    nx, ny = 101, 101
    xE = np.linspace(-5, 5, nx)
    yE = np.linspace(-5, 5, ny)

    x0, y0 = 0.0, 0.0
    sigma = 0.25

    dx = xE[1] - xE[0]
    dy = yE[1] - yE[0]
    dt = 0.5 * min(dx, dy) / C0
    Tf = 2

    # --- Prueba 1: Banda gaussiana en X (constante en Y) ---
    initial_condition_x = np.zeros((nx, ny))
    for i in range(nx):
        gaussian_value = gaussian_pulse(xE[i], x0, sigma)
        for j in range(ny):
            initial_condition_x[i, j] = gaussian_value

    # Resolver en 2D
    Ez_final_x = fdtd_2d_solver(initial_condition_x, xE, yE, dt, Tf, bounds=('mur', 'mur'))

    # Extraer sección en X (y = 0)
    mid_y = ny // 2
    section_x = Ez_final_x[:, mid_y]

    # Solución esperada en 1D para X
    expected_x = 0.5 * gaussian_pulse(xE, x0 - C0 * Tf, sigma) + 0.5 * gaussian_pulse(xE, x0 + C0 * Tf, sigma)

    # Graficar para debugging
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xE, section_x, label='Sección en X (2D)')
    ax.plot(xE, expected_x, '--', label='Esperado en X (1D)')
    ax.set_title('Comparación en el eje X (banda en X)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Validar coherencia en X
    assert np.corrcoef(section_x, expected_x)[0, 1] >= 0.99, "Perfil en X no coincide con la solución 1D"
    print("Test pasado: la solución 2D (banda en X) es coherente con la solución 1D en X.")

    # --- Prueba 2: Banda gaussiana en Y (constante en X) ---
    initial_condition_y = np.zeros((nx, ny))
    for j in range(ny):
        gaussian_value = gaussian_pulse(yE[j], y0, sigma)
        for i in range(nx):
            initial_condition_y[i, j] = gaussian_value

    # Resolver en 2D
    Ez_final_y = fdtd_2d_solver(initial_condition_y, xE, yE, dt, Tf, bounds=('mur', 'mur'))

    # Extraer sección en Y (x = 0)
    mid_x = nx // 2
    section_y = Ez_final_y[mid_x, :]

    # Solución esperada en 1D para Y
    expected_y = 0.5 * gaussian_pulse(yE, y0 - C0 * Tf, sigma) + 0.5 * gaussian_pulse(yE, y0 + C0 * Tf, sigma)

    # Graficar para debugging
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yE, section_y, label='Sección en Y (2D)')
    ax.plot(yE, expected_y, '--', label='Esperado en Y (1D)')
    ax.set_title('Comparación en el eje Y (banda en Y)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Validar coherencia en Y
    assert np.corrcoef(section_y, expected_y)[0, 1] >= 0.99, "Perfil en Y no coincide con la solución 1D"
    print("Test pasado: la solución 2D (banda en Y) es coherente con la solución 1D en Y.")

# Invocación de los tests (debugging)
#test_fdtd_1d_solver_basic_propagation()
test_fdtd_2d_solver_basic_propagation()