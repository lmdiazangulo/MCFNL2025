import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

# Constants for permittivity regions test
EPS1 = 2.0
C1 = 1 / np.sqrt(MU0*EPS1)
R = (np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
T = 2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def sigmoid_grid(xmin=-1, xmax=1, npoints=101, steepness=7, midpoint=0): # midpoint=0 for centered sigmoid in interval
  """
  Generates a non-uniform grid using a sigmoid function.

  """
  x = np.linspace(-steepness, steepness, npoints)
  grid = xmin + (xmax - xmin) / (1 + np.exp(-x + midpoint*steepness)) #Sigmoid function
  grid = xmin + (grid - np.min(grid)) / (np.max(grid) - np.min(grid)) * (xmax - xmin) #Rescale it so that it matches the endpoints
  return grid

def RT_coeffs_scikit(initial_condition, dt, eps0, cond0, eps1, cond1, wPanel):
    import numpy as np
    from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
    from scipy.interpolate import interp1d
    import skrf as rf

    # === Espectro de frecuencias normalizadas ===
    frequencies = fftfreq(len(initial_condition), dt)  # en unidades naturalizadas
    freq = fftshift(frequencies)
    omega = 2 * np.pi * freq

    # FFT de la señal incidente
    G_incident = fftshift(fft(initial_condition))

    # === Eliminar la frecuencia cero para evitar división por cero en scikit-rf ===
    mid_idx = len(freq) // 2
    freq_no0 = np.delete(freq, mid_idx)
    omega_no0 = 2 * np.pi * freq_no0
    omega_safe = np.where(omega_no0 == 0, 1e-12, omega_no0)  # por seguridad extra

    # === Propiedades del material (sin unidades físicas) ===
    er = eps1 / eps0
    sigma = cond1
    thickness = wPanel
    mu = 1.0  # unidades normalizadas

    # === Cálculo de parámetros complejos ===
    epsilon_complex = er - 1j * sigma / omega_safe
    gamma = 1j * omega_no0 * np.sqrt(mu * epsilon_complex)
    Z0 = np.sqrt(mu / epsilon_complex)

    # === Crear frecuencia para skrf (GHz ficticios) ===
    freq_GHz = freq_no0 * 1e-3
    frequency = rf.Frequency.from_f(freq_GHz, unit='ghz')

    # === Medio y línea de transmisión ===
    medium = rf.media.DefinedGammaZ0(frequency=frequency, gamma=gamma, Z0=Z0)
    layer = medium.line(d=thickness, unit='m')

    # === Parámetros S ===
    s11 = layer.s[:, 0, 0]
    s21 = layer.s[:, 1, 0]

    # === Interpolación a las frecuencias completas (incluyendo ω = 0) ===
    interp_s11 = interp1d(freq_no0 * 1e3, s11, kind='cubic', fill_value='extrapolate', bounds_error=False)

    interp_s21 = interp1d(freq_no0 * 1e3, s21, kind='cubic', fill_value='extrapolate', bounds_error=False)

    S11 = interp_s11(freq * 1e3)
    S21 = interp_s21(freq * 1e3)

    print("Max |S11|:", np.max(np.abs(S11)))
    print("Mean |S11|:", np.mean(np.abs(S11)))


    # === Campos reflejado y transmitido ===
    G_reflected = G_incident * S11
    G_transmitted = G_incident * S21

    E_reflected = np.real(ifft(ifftshift(G_reflected)))
    E_transmitted = np.real(ifft(ifftshift(G_transmitted)))

    # === Coeficientes R y T por energía ===
    R = np.sum(E_reflected**2) / np.sum(initial_condition**2)
    T = np.sum(E_transmitted**2) / np.sum(initial_condition**2)

    return R, T





class FDTD1D:
    def __init__(self, xE, bounds=('pec', 'pec')):
        self.xE = np.array(xE)
        self.xH = (self.xE[:-1] + self.xE[1:]) / 2.0
        self.dxE = np.diff(self.xE)
        self.avgdxE = np.mean(self.dxE) # For energy calculations
        self.dxH = np.diff(self.xH)
        self.dt = 0.9 * np.min(np.concatenate([self.dxE, self.dxH])) / C0 # Choose a safe dt by default for stability
        self.bounds = bounds
        self.e = np.zeros_like(self.xE)
        self.h = np.zeros_like(self.xH)
        self.h_old = np.zeros_like(self.h)
        self.eps = np.ones_like(self.xE)  # Default permittivity is 1 everywhere
        self.cond = np.zeros_like(self.xE)  # Default conductivity is 0 everywheree
        self.condPML=np.zeros_like(self.xH) # Fake PML magnetic conductivty
        self.initialized = False
        self.tfsf = False
        self.energyE = []
        self.energyH = []
        self.energy = []
        self.time = 0
        self.total_field = []
        self.indexProbe = []
        self.e_measure = []
        self.h_measure = []
        self.wPanel = 0
        self.xPanel = 0

    def set_initial_condition(self, initial_condition, initial_h_condition=None):
        self.e[:] = initial_condition[:]
        if initial_h_condition is not None:
            self.h[:] = initial_h_condition[:]
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

        '''max_eps = np.max(self.eps)
        c_max = 1 / np.sqrt(MU0*max_eps)
        self.dt = 0.9 * np.min(np.concatenate([self.dxE, self.dxH])) / c_max  # Redefine the safe dt according to permittivities'''

    def set_conductivity_regions(self, regions):
        """Set different conductivity regions in the grid.

        Args:
            regions: List of tuples (start_x, end_x, cond_a_value) defining regions
                    with different conductivity values.
        """
        for start_x, end_x, cond_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)

            self.cond[start_idx:end_idx] = cond_value


    def set_layer_panel(self, L, wPanel, xPanel, eps_value=1, cond_value=0, eps_layer = 1, cond_layer = 0):
        '''
        Set a layer panel in the grid with given permittivity and conductivity values.
        '''
        self.set_permittivity_regions([
            (-L/2, -wPanel/2+xPanel, eps_value),  # First region with EPS0
            (-wPanel/2+xPanel, wPanel/2+xPanel, eps_layer),    # Second region with EPS1
            (wPanel/2+xPanel, L/2, eps_value) 
        ])

        self.set_conductivity_regions([
            (-L/2, -wPanel/2+xPanel, cond_value),  # First region with EPS0
            (-wPanel/2+xPanel, wPanel/2+xPanel, cond_layer),   # Second region with EPS1
            (wPanel/2+xPanel, L/2, cond_value) 
        ])
        self.wPanel = wPanel
        self.xPanel = xPanel

    def set_PML(self,thicknessPML,m,sigmaMax):

        '''
        Setting the PML region and value
        (it means changing both self.cond and self.condPML)
        on the region in which it is implemented.
        That is at both sides of the positions array with a thickness
        of thicknessPML cells.
        '''

        for i in range(0,thicknessPML):
            sigmai=sigmaMax*((thicknessPML-i)/thicknessPML)**m
            right_index= int(len(self.condPML)-1-i)
            self.condPML[i]=sigmai
            self.condPML[right_index]=sigmai
            self.cond[i]=sigmai
            self.cond[right_index]=sigmai

        #plt.plot(self.xH,self.condPML) #to plot the PML profile
        #plt.show()

    def add_totalfield(self,xs,sourceFunction):
        '''
        Add a field source at a given location, both in the electric and magnetic domain.
        Args:
            xs: Location in space of the source

            sourceFunction: Function F(x,t) that gives the shape of the field injected in xs

        '''
        isource = np.where(self.xE > xs)[0][0] # Index in xE and xH of the location of the source
        self.total_field = self.total_field + [isource, sourceFunction]

    def add_probe(self,x_probe, regions):
        '''
        Add probes that measure and record the magnetic and electric field at their location
        at each time step during the simulation.
        Args:
            x_probe: list of positions in space where the probes will be located
        '''
        for location in x_probe:
            index = np.where(self.xE > location)[0][0]
            self.indexProbe += [index]
            self.e_measure += [[]]
            self.h_measure += [[]]

    def set_tfsf_conditions(self, x_start, x_end, function):
        self.x_start = x_start
        self.x_end = x_end
        self.tfsolver = FDTD1D(self.xE, bounds=('mur', 'mur'))
        self.tfsolver.set_initial_condition(function)
        self.tfsf = True

    




        
    def step(self, regions=None):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        self.e_old_left = self.e[1]
        self.e_old_right = self.e[-2]

        # Field calculation

        C1h = (MU0 / self.dt + self.condPML[:] / 2)
        C2h = (MU0 / self.dt - self.condPML[:] / 2)

        self.h[:] = (C2h / C1h * self.h[:])  - MU0 * self.dt / self.dxE[:] * (self.e[1:] - self.e[:-1])

        # self.h[:] = ( 1 / ((MU0 / self.dt) + (self.condPML[:] / 2)) ) * ( ( (MU0/self.dt) - (self.condPML[:]/2) ) * self.h[:] - self.dt * MU0 / self.dxE[:] * (self.e[1:] - self.e[:-1]) )
        
        #self.h[:] = self.h[:] - dt / self.dx / MU0 * (self.e[1:] - self.e[:-1])
        if self.total_field: # Injection of total field in h field
            isource = self.total_field[0]
            sourcefunction = self.total_field[1]
            self.h[isource] += sourcefunction(self.xH[isource],self.time)/2
            self.h[isource-1] += sourcefunction(self.xH[isource-1],self.time)/2
        if self.indexProbe: # Measure of magnetic field
            for i in range(len(self.indexProbe)):
                self.h_measure[i] += [self.h[self.indexProbe[i]]]
        self.time += self.dt/2 # Half time step upload

        C1e = (self.eps[1:-1] + self.cond[1:-1] * self.dt / 2)
        C2e = (self.eps[1:-1] - self.cond[1:-1] * self.dt / 2)

        self.e[1:-1] = (C2e / C1e * self.e[1:-1]) - self.dt / (self.dxH  * C1e) * (self.h[1:] - self.h[:-1])


        # self.e[1:-1] = ( 1 / ((self.eps[1:-1] / self.dt) + (self.cond[1:-1] / 2)) ) * ( ( (self.eps[1:-1]/self.dt) - (self.cond[1:-1]/2) ) * self.e[1:-1] - 1 / self.dxH[:] * ( 1 / ((self.eps[1:-1] / self.dt) + (self.cond[1:-1] / 2)) ) * (self.h[1:] - self.h[:-1]) )

        if self.total_field: # Injection of total field in e field
            self.e[isource] += sourcefunction(self.xE[isource],self.time)
        if self.indexProbe: # Measure of electric field
            for i in range(len(self.indexProbe)):
                self.e_measure[i] += [self.e[self.indexProbe[i]]]
        self.time += self.dt/2# Half time step upload

        # Bound calculation

        if self.bounds[0] == 'pec':
            self.e[0] = 0.0
        elif self.bounds[0] == 'mur':
            self.e[0] = self.e_old_left + (C0*self.dt - self.dxE[0]) / \
                (C0*self.dt + self.dxE[0])*(self.e[1] - self.e[0])
        elif self.bounds[0] == 'pmc':
            self.e[0] = self.e[0] - 2 * self.dt/ self.dxE[0]/ EPS0*(self.h[0])
        elif self.bounds[0] == 'periodic':
            self.e[0] = self.e[-2]
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[0]}")

        if self.bounds[1] == 'pec':
            self.e[-1] = 0.0
        elif self.bounds[1] == 'mur':
            self.e[-1] = self.e_old_right + (C0*self.dt - self.dxE[-1]) / \
                (C0*self.dt + self.dxE[-1])*(self.e[-2] - self.e[-1])
        elif self.bounds[1] == 'pmc':
            self.e[-1] = self.e[-1] + 2 * self.dt/self.dxE[-1] / EPS0*(self.h[-1])
        elif self.bounds[1] == 'periodic':
            self.e[-1] = self.e[1]
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[1]}")

        # Energy calculation
        self.energyE.append(0.5 * np.dot(self.e, self.avgdxE * self.eps * self.e))
        self.energyH.append(0.5 * np.dot(self.h_old, self.avgdxE * MU0 * self.h))
        self.energy.append(0.5 * np.dot(self.e, self.avgdxE * self.eps * self.e) + 0.5 * np.dot(self.h_old, self.avgdxE * MU0 * self.h))
        self.h_old[:] = self.h[:]


        # For debugging and visualization
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0  # Initialize step counter if it doesn't exist

        self.step_counter += 1

        # For marking some special areas
        # plt.axvspan(self.xE[0], self.xE[len(self.xE)-1], color='skyblue', alpha=0.05, label='zona destacada')
        # plt.axvspan(self.xE[-(len(self.xE)-1)], self.xE[-1], color='skyblue', alpha=0.05, label='zona destacada')

        # Plot only every 100 steps (you can adjust the interval as needed)
        # Plot only every 10 steps (you can adjust the interval as needed)
        
        # Plot every 10 steps
        if self.step_counter % 10 == 0:
            plt.plot(self.xE, self.e, '-', label='Electric Field')
            plt.plot(self.xH, self.h, '-', label='Magnetic Field')
            ax = plt.gca()
            if(self.wPanel>0):
                ax.axvline(x=self.xPanel-self.wPanel/2, color='r', linestyle='--', label='Region Boundary')
                ax.axvline(x=self.xPanel+self.wPanel/2, color='r', linestyle='--')
            plt.ylim(-1, 1)
            plt.legend()
            plt.grid()
            plt.pause(0.01)
            plt.cla()




    def run_until(self, Tf=None, dt=None, n_steps=100):
        print("Running simulation...")
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        if dt is not None: # Define a specific dt for the solver
            if dt > self.dt:
              raise RuntimeError(
                  "Too high dt value. Method is not stable. Set it to a lower value")
            else:
              self.dt = dt

        if Tf is not None: # If a final time is defined, let it run until then. If not, use the steps.
          used_n_steps = int(Tf / self.dt)
          for n in range(used_n_steps):
            self.step()

            #TFSF calculation
            if self.tfsf:
              self.tfsolver.step()
              self.e[(np.abs(self.xE - self.x_start)).argmin()] += self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_start)).argmin()]
              self.e[(np.abs(self.xE - self.x_end)).argmin()] -= self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_end)).argmin()]
              self.h[(np.abs(self.xH - self.x_start)).argmin()] += self.tfsolver.h[(np.abs(self.tfsolver.xH - self.x_start)).argmin()]
              self.h[(np.abs(self.xH - self.x_end)).argmin()] -= self.tfsolver.h[(np.abs(self.tfsolver.xH - self.x_end)).argmin()]

        else:
          for n in range(n_steps):
            self.step()

            # TFSF calculation
            if self.tfsf:
              self.tfsolver.step()
              self.e[(np.abs(self.xE - self.x_start)).argmin()] += self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_start)).argmin()]
              self.e[(np.abs(self.xE - self.x_end)).argmin()] -= self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_end)).argmin()]

        return self.e
