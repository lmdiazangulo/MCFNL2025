import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0 * EPS0)


def analytic_chiral_RT(wy, lambda0, kappa, sigma, eps_r=1.0, mu_r=1.0):
    """
    Coeficientes de transmisión (T_R, T_L) y reflexión (R_R, R_L) 
    para polarizaciones RCP y LCP en un slab quiral de grosor wy,
    chirality kappa y conductividad sigma.
    """
    # Constantes
    epsilon0 = 1
    k0       = 2 * np.pi / lambda0
    omega    = C0 * k0

    # Permitividad compleja y parámetros relativos
    eps_c = eps_r - 1j * sigma / (omega * epsilon0)
    n     = np.sqrt(eps_c * mu_r)        # índice relativo complejo
    eta   = np.sqrt(mu_r / eps_c)        # impedancia relativa compleja

    # Argumento complejo
    phi  = n * k0 * wy
    cosφ = np.cos(phi)
    sinφ = np.sin(phi)

    # Denominador común (ecuaciones 15,16)
    denom = 2*eta * cosφ + 1j*(eta**2 + 1)*sinφ

    # Transmisión RCP / LCP (con fase de quiralidad)
    phaseR = np.exp(-1j * kappa * phi)
    phaseL = np.exp(+1j * kappa * phi)
    T_R = 2*eta * phaseR / denom
    T_L = 2*eta * phaseL / denom

    # Reflexión (idéntica para RCP y LCP)
    # R = -j/2 * sin(n k0 wy) * (1 - η^2) / denom
    R_coeff = -0.5j * sinφ * (1 - eta**2) / denom
    R_R = R_coeff
    R_L = R_coeff

    return T_R, T_L, R_R, R_L

import numpy as np
from fdtd2d import analytic_chiral_RT, C0

def transmission_from_gaussian_pulse(
    wy,             # grosor del slab
    kappa,          # parámetro quiral
    sigma_cond,     # conductividad del slab
    eps_r,          # permitividad relativa
    mu_r,           # permeabilidad relativa
    sigma_t,        # ancho temporal (σ) de la gaussiana de entrada
    dt,             # paso temporal para muestreo
    N               # número de puntos de la ventana temporal
):
    """
    Calcula la transmitancia de un slab quiral para un pulso gaussiano T(t)=exp[-(t-t0)^2/(2σ^2)]
    mediante FFT:
      1) genera pulso en t
      2) FFT → S(ω)
      3) aplica T(ω) frecuencia a frecuencia
      4) IFFT → s_out(t)
      5) transmitancia = max|s_out| / max|s_in|
    """

    # 1) tiempo y pulso
    t = np.arange(N) * dt
    t0 = 4 * sigma_t           # atraso para centrar el pulso
    s_in = np.exp(- (t - t0)**2 / (2*sigma_t**2))

    # 2) FFT real
    S_in = np.fft.rfft(s_in)
    freqs = np.fft.rfftfreq(N, d=dt)       # frecuencia en Hz
    omegas = 2*np.pi * freqs               # rad/s

    # 3) construimos T_eff(ω) aplicable al pulso lineal
    T_eff = np.zeros_like(omegas, dtype=complex)
    for i, omega in enumerate(omegas):
        if freqs[i] == 0:
            T_eff[i] = 1.0   # DC pasa sin atenuación
        else:
            lambda0 = C0 / freqs[i]
            T_R, T_L, _, _ = analytic_chiral_RT(
                wy, lambda0, kappa, sigma_cond, eps_r, mu_r
            )
            # para polarización lineal simplemente promediamos RCP y LCP:
            T_eff[i] = 0.5*(T_R + T_L)

    # 4) señal transmitida en frecuencia y vuelta al tiempo
    S_out = S_in * T_eff
    s_out = np.fft.irfft(S_out, n=N)

    # 5) cálculo de transmitancia
    T_num = np.max(np.abs(s_out))
    T_den = np.max(np.abs(s_in))
    transmission = T_num / T_den

    return transmission



class FDTD2D:
    def __init__(self, xE, yE):
        self.xE = np.array(xE)
        self.yE = np.array(yE)
        self.nx = len(self.xE)
        self.ny = len(self.yE)
        self.dx = self.xE[1] - self.xE[0]
        self.dy = self.yE[1] - self.yE[0]
        self.E_trans = 0.0

        self.epsEx   = np.ones((self.nx,     self.ny - 1))
        self.condEx  = np.zeros((self.nx,     self.ny - 1))
        self.kappaEx = np.zeros((self.nx,     self.ny - 1))

        self.epsEy   = np.ones((self.nx - 1, self.ny    ))
        self.condEy  = np.zeros((self.nx - 1, self.ny    ))
        self.kappaEy = np.zeros((self.nx - 1, self.ny    ))

        self.condy = np.zeros_like(self.epsEx)  # usada en Ex-update: (nx,ny-1)
        self.condx = np.zeros_like(self.epsEy)  # usada en Ey-update: (nx-1,ny)

        # Conductividad PML en nodos Hz
        self.condPMLx = np.zeros((self.nx, self.ny))
        self.condPMLy = np.zeros((self.nx, self.ny))

        # TE mode fields: Hz (magnetic field in z), Ex (electric field in x), Ey (electric field in y)
        self.Hz  = np.zeros((self.nx,     self.ny    ))
        self.Hzx = np.zeros_like(self.Hz)
        self.Hzy = np.zeros_like(self.Hz)
        self.Ex  = np.zeros_like(self.epsEx)
        self.Ey  = np.zeros_like(self.epsEy)

        # Datos para fuente y panel
        self.source = None
        self.panel  = None
        self.time   = 0.0

        self.aux = 0.0

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
            self.condEx[i, :] += sigmai
            self.condEx[right_index, :] += sigmai

            # Apply PML to the Y-axis boundaries
            self.condPMLy[:, i] += sigmai
            self.condPMLy[:, right_index] += sigmai
            self.condEy[:, i] += sigmai
            self.condEy[:, right_index] += sigmai

  
    def set_chiral_panel(self, x0, y0, wx, wy, eps, sigma, kappa):
        """
        Define a rectangular chiral panel with center (x0,y0), width wx, wy,
        permittivity eps, conductivity sigma, and chiral parameter kappa.
        """
        self.panel = (x0, y0, wx, wy)
        # Masks for Ex grid
        X_ex = self.xE[:,None]
        Y_ex = (self.yE[:-1] + self.yE[1:])[None,:] / 2
        mask_ex = (np.abs(X_ex-x0)<=wx/2) & (np.abs(Y_ex-y0)<=wy/2)
        self.epsEx[mask_ex]   = eps
        self.condEx[mask_ex]  = sigma
        self.kappaEx[mask_ex] = kappa
        # Masks for Ey grid
        X_ey = (self.xE[:-1] + self.xE[1:])[:,None] / 2
        Y_ey = self.yE[None,:]
        mask_ey = (np.abs(X_ey-x0)<=wx/2) & (np.abs(Y_ey-y0)<=wy/2)
        self.epsEy[mask_ey]   = eps
        self.condEy[mask_ey]  = sigma
        self.kappaEy[mask_ey] = kappa

    def step(self, dt, pos):
        if not self.initialized:
            raise RuntimeError("Call set_initial_condition first.")
      
       
        # Update electric field
        # --- Update Ex ---
        Ex_new = (1/(2*self.epsEx + self.condEx*dt)) * (
            (2*self.epsEx - self.condEx*dt)*self.Ex
          + (2*dt/self.dy)*(self.Hz[:,1:]-self.Hz[:,:-1])
         )
        # chiral term: + (dt/eps) * (xi * H_z at Ex location)
        Ex_new += (dt/ (self.epsEx)) * (self.kappaEx * self.Hz[:,1:])
        self.Ex = Ex_new

        # --- Update Ey ---
        Ey_new = (1/(2*self.epsEy + self.condEy*dt)) * (
            (2*self.epsEy - self.condEy*dt)*self.Ey
          - (2*dt/self.dx)*(self.Hz[1:,:]-self.Hz[:-1,:])
         )
        Ey_new -= (dt/ (self.epsEy)) * (self.kappaEy * self.Hz[1:,:])
        self.Ey = Ey_new

      
        # Update magnetic field
        self.Hzx[1:-1, 1:-1] = ( 1 / (2 * MU0 + self.condPMLx[1:-1, 1:-1] * dt) ) * ( (2 * MU0 - self.condPMLx[1:-1, 1:-1] * dt) * self.Hzx[1:-1, 1:-1] - ( (2*dt) / self.dx ) * 
            (self.Ey[1:, 1:-1] - self.Ey[:-1, 1:-1]) )
        self.Hzy[1:-1, 1:-1] = ( 1 / (2 * MU0 + self.condPMLy[1:-1, 1:-1] * dt) ) * ( (2 * MU0 - self.condPMLy[1:-1, 1:-1] * dt) * self.Hzy[1:-1, 1:-1] + ( (2*dt) / self.dy ) *
            (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1]) )

        # Combine magnetic fields
        Hz_new = self.Hzx + self.Hzy
        curlE = np.zeros_like(self.Hz)
        curlE[1:-1,1:-1] = ((self.Ey[1:,1:-1] - self.Ey[:-1,1:-1]) / self.dx
                            - (self.Ex[1:-1,1:] - self.Ex[1:-1,:-1]) / self.dy)
        # full chiral term: - (dt/mu) * (xi interpolated * curlE)
        # interpolate xi to Hz grid
        xi_H = np.zeros_like(self.Hz)
        xi_H[1:-1,1:-1] = 0.25*(self.kappaEx[1:-1,1:] + self.kappaEx[1:-1,:-1]
                               + self.kappaEy[1:,1:-1] + self.kappaEy[:-1,1:-1])
        Hz_new[1:-1,1:-1] -= (dt/MU0) * (xi_H[1:-1,1:-1] * curlE[1:-1,1:-1])

        # PEC boundary conditions
        Hz_new[0,:] = 0; 
        Hz_new[-1,:] = 0; 
        Hz_new[:,0] = 0; 
        Hz_new[:,-1] = 0
        self.Hz = Hz_new
        # Update time
        self.time += dt

        # Returneamos el maximo del campo electrico en el punto x = 5 e y = pos
        self.E_trans = abs(self.Hz[int(5/self.dx), int(pos/self.dy)])

        # print(self.aux)
        if self.E_trans > self.aux:
            self.aux = self.E_trans

    def run_until(self, Tf, dt):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        n_steps = int(Tf / dt)
        for _ in range(n_steps):
            self.step(dt)

        return self.Hz 

    def simulate_and_plot(self, Tf, dt, pos, simulate, interval=10):
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

        if simulate:
            fig, ax = plt.subplots()
            cax = ax.imshow(self.Hz.T, cmap='viridis', origin='lower', extent=[self.xE[0], self.xE[-1], self.yE[0], self.yE[-1]])
            if self.panel:
                x0,y0,wx,wy = self.panel
                ax.add_patch(Rectangle((x0-wx/2,y0-wy/2),wx,wy,fill=False,edgecolor='red',linewidth=2))
 
            # mostramos un punto en el punto x = 5 e y = pos
            ax.plot(5, pos, 'ro', markersize=5, label='Punto de interés')

            fig.colorbar(cax, ax=ax)
            ax.set_title("Hz Field Evolution")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            
            def update(frame):
                for _ in range(interval):
                    self.step(dt, pos)
            
                cax.set_array(self.Hz.T)
                return cax,

            ani = animation.FuncAnimation(fig, update, frames=n_steps // interval, blit=False, repeat=False)
            plt.show()

        else:
            for _ in range(n_steps):
                    self.step(dt, pos)
        