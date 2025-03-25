import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

# MODO TMz (representamos Ez frente a Hx, Hy)
# ANIMACION CON DEBUGGING
def fdtd_2d_solver_animated(initial_condition, nx, ny, dt, Tf, bounds=('pec', 'pec')):
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

    # Almacenar Ez en cada paso para la animación
    Ez_history = []
    n_steps = int(Tf / dt)
    
    for n in range(n_steps):
        Hx[:, :] = Hx[:, :] - dt / MU0 / dx * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:, :] = Hy[:, :] + dt / MU0 / dy * (Ez[1:, :] - Ez[:-1, :])

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

        # Guardar una copia de Ez cada 10 pasos para no saturar la memoria
        if n % 10 == 0:
            Ez_history.append(Ez.copy())

    return Ez_history

# Configuración
nx, ny = 100, 100
xE = np.linspace(0, 1, nx)
yE = np.linspace(0, 1, ny)
initial_condition = np.zeros((nx, ny))

# Fuente inicial más amplia (gaussiana) para ver propagación clara
# x0, y0 = nx//2, ny//2
# for i in range(nx):
#     for j in range(ny):
#         initial_condition[i, j] = np.exp(-((i-x0)**2 + (j-y0)**2) / 100)
# dt = 0.001  # Asegurarse de que cumpla Courant: dt < dx/c
# Tf = 100
# Ez_history = fdtd_2d_solver_animated(initial_condition, nx, ny, dt, Tf)

# Fuente inicial banda gaussiana
initial_condition = np.zeros((nx, ny))
x0, y0 = nx//2, ny//2
for i in range(nx):
    for j in range(ny):
        initial_condition[i, j] = np.exp(-((i-x0)**2) / 100)
dt = 0.001  # Asegurarse de que cumpla Courant: dt < dx/c
Tf = 100
Ez_history = fdtd_2d_solver_animated(initial_condition, nx, ny, dt, Tf)

# Animación
fig, ax = plt.subplots()
im = ax.imshow(Ez_history[0].T, extent=[xE[0], xE[-1], yE[0], yE[-1]], cmap='RdBu', origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, label='Ez')
im.set_clim(-1, 1)

def update(frame):
    im.set_data(Ez_history[frame].T)
    return im,

ani = FuncAnimation(fig, update, frames=len(Ez_history), interval=50, blit=True)
plt.show()