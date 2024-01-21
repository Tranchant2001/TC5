# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from field import Field
from velocity_field import VelocityField
from pressure_field import PressureField
from species import Species, Dioxygen, Dinitrogen, Methane, Water, CarbonDioxide
from temperature_field import TemperatureField
import misc



class CounterFlowCombustion():

    #__slots__ = "L", "physN", "L_slot", "L_coflow", "D", "rho", "max_t_comput", "show_save", "register_period", "ell_crit", "div_crit", "conv_crit", "dx", "dt", "omega", "y", "ghost_thick", "N"
    
    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, nu:float, D:float, a:float, rho:float, c_p:float, Ta:float, time_ignite:float, max_t_comput:datetime.timedelta, show_save:bool, register_period:int, i_reactor:int, j_reactor:int, ell_crit:float, div_crit:float, conv_crit:float):

        self.L = L
        self.physN = physN
        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.nu = nu
        self.D = D
        self.a = a
        self.rho = rho
        self.c_p = c_p
        self.Ta = Ta
        self.time_ignite = time_ignite
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.register_period = register_period
        self.ell_crit = ell_crit
        self.div_crit = div_crit
        self.conv_crit = conv_crit

        self.dx = L/(physN-1)
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.4*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step
        self.dtchem_list = self.set_dtchem_list(self.dt)
        #self.dt = 4e-6
        
        self.omega = 2*(1 - math.pi/physN - (math.pi/physN)**2)
        #self.omega = 1.5    # omega parameter evaluation of the SOR method

        self.y = np.linspace(0, L, physN, endpoint=True, dtype=np.float32)
        # Create mesh grid
        #self.X, self.Y = np.meshgrid(np.linspace(0, L, physN, endpoint=True) , np.linspace(0, L, N, endpoint=True))
        #self.I, self.J = np.meshgrid(np.linspace(0, N, N, endpoint=True, dtype=int) , np.linspace(0, N, N, endpoint=True, dtype=int))

        self.ghost_thick = 2
        self.N = physN + 2*self.ghost_thick
        self.i_reactor = i_reactor + self.ghost_thick 
        self.j_reactor = j_reactor + self.ghost_thick

        # I want the variable counting the nb of frames realized to be a class attribute.
        self.frame = 0 
        self.chem_frame = 0     


    def set_dtchem_list(self, dt):
        
        number = 12
        l0 = np.linspace(-4.4, 0, number, endpoint=True)
        l1 = 10**l0
        l3 = l1*dt
        l4 = [l3[0]]
        acc_time = l3[0]
        for k in range(1, number):
            dttemp = min(l3[k] - l3[k-1], dt*0.1)
            l4.append(dttemp)
            acc_time += dttemp

        n_r = int((dt - acc_time)/(dt*0.1))
        l4 = l4 + [dt*0.1]*n_r
        acc_time += n_r*dt*0.1
        l4.append(dt-acc_time)

        l5 = np.array(l4, dtype=float)
        l6 = np.zeros(l5.shape[0], dtype=float)
        l6[0] = l5[0]
        for k in range(1, l5.shape[0]):
            l6[k] = l5[k] + l6[k-1]

        l4 = np.full(800, dt/800)

        #fig1 = plt.figure()
        #plt.plot(l6, label="Time piled up")
        #plt.plot(l5, label="$\Delta t$")
        #plt.xlabel("Index")
        #plt.ylabel("Time s")
        #plt.legend()
        #plt.show()

        return l4
        

    def get_beta(self, uvet:VelocityField):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.dx
        u = uvet.u.values
        v = uvet.v.values

        forward_x =     0.5*(-3*u     + 4*np.roll(u, -1, axis=1)    - np.roll(u, -2, axis=1))/dx
        backward_x =    0.5*( 3*u     - 4*np.roll(u, 1, axis=1)     + np.roll(u, 2, axis=1))/dx

        forward_y =     0.5*(-3*v   + 4*np.roll(v, -1, axis=0) - np.roll(v, -2, axis=0))/dx
        backward_y =    0.5*( 3*v   - 4*np.roll(v, 1, axis=0)  + np.roll(v, 2, axis=0))/dx

        u_x = np.where(u < 0, forward_x, backward_x)

        v_y = np.where(v < 0, forward_y, backward_y)

        return (dx**2*self.rho/self.dt)*(u_x + v_y)


    def ell_solver(self, uvet:VelocityField, P:PressureField):

        # omega parameter evaluation of the SOR method
        omega = self.omega
        dx =self.dx
        dt = self.dt
        rho = self.rho

        # beta Laplcaian P converges to beta_arr
        beta_arr =self.get_beta(uvet)
    
        eps = 1.
        kiter = 1
        maxiter = 999
        while eps > self.ell_crit and kiter < maxiter:
            if kiter%1002==1000:
                print(f"\tElliptic Solver\ti={kiter}")
                misc.plot_field(P, title=f"Elliptic solver\nIter={kiter}\nresidual err={eps:.3f}\n $\omega={omega}$", saveaspng=f"{kiter}_pressure_convergance.png", pause=2)
            # Realizes a backward and then a forward SOR to symmetrizes error.
            P.update_forward_SOR(beta_arr, omega)
            P.update_backward_SOR(beta_arr, omega)

            eps = P.residual_error(beta_arr)

            kiter += 1
        
        P.last_nb_iter = kiter-1

        return P


    def _f_velocities(self, uval, vval,  phi):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.dx

        # get the translation arrays to calculate further the derivatives estimates.
        # Translation along x axis, indexed by i.
        phi_ip1 = np.roll(phi, -1, axis=1)
        phi_ip2 = np.roll(phi, -2, axis=1)
        phi_im1 = np.roll(phi, 1, axis=1)
        phi_im2 = np.roll(phi, 2, axis=1)
        
        # Translation along y axis, indexed by j.
        phi_jp1 = np.roll(phi, -1, axis=0)
        phi_jp2 = np.roll(phi, -2, axis=0)
        phi_jm1 = np.roll(phi, 1, axis=0)
        phi_jm2 = np.roll(phi, 2, axis=0)

        forward_x =     0.5*(-3*phi     + 4*phi_ip1 - phi_ip2)/dx
        backward_x =    0.5*( 3*phi     - 4*phi_im1 + phi_im2)/dx

        forward_y =     0.5*(-3*phi     + 4*phi_jp1 - phi_jp2)/dx
        backward_y =    0.5*( 3*phi     - 4*phi_jm1 + phi_jm2)/dx

        phi_x = np.where(uval < 0, forward_x, backward_x) #### bon sens, test !

        phi_y = np.where(vval < 0, forward_y, backward_y) ####

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return self.nu*laplacian_phi - uval*phi_x - vval*phi_y
    

    def _update_RK2_velocity(self, uv:VelocityField, uvet:VelocityField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """
        L = self.L
        N = self.N
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        thick = uv.ghost_thick

        u0 = np.copy(uv.u.values)
        v0 = np.copy(uv.v.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        uvet.u.values = u0 + 0.5*self.dt*self._f_velocities(u0, v0, u0)
        uvet.v.values = v0 + 0.5*self.dt*self._f_velocities(u0, v0, v0)
        uvet._fillGhosts()
        misc.plot_field(uvet.v, True, title=f'V* Field',saveaspng="_v-et00_field.png")
        misc.plot_field(uvet.u, True, title=f'U* Field',saveaspng="_u-et00_field.png")            

 
        u12 = np.copy(uvet.u.values)
        v12 = np.copy(uvet.v.values)
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        uvet.u.values = u0 + self.dt*self._f_velocities(u12, v12, u12)
        uvet.v.values = v0 + self.dt*self._f_velocities(u12, v12, v12)
        uvet._fillGhosts()
        misc.plot_field(uvet.v, True, title=f'V* Field',saveaspng="_v-et01_field.png")
        misc.plot_field(uvet.u, True, title=f'U* Field',saveaspng="_u-et01_field.png")            


    def _f_species(self, uval, vval, phi):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.dx

        # get the translation arrays to calculate further the derivatives estimates.
        # Translation along x axis, indexed by i.
        phi_ip1 = np.roll(phi, -1, axis=1)
        phi_ip2 = np.roll(phi, -2, axis=1)
        phi_im1 = np.roll(phi, 1, axis=1)
        phi_im2 = np.roll(phi, 2, axis=1)
        
        # Translation along y axis, indexed by j.
        phi_jp1 = np.roll(phi, -1, axis=0)
        phi_jp2 = np.roll(phi, -2, axis=0)
        phi_jm1 = np.roll(phi, 1, axis=0)
        phi_jm2 = np.roll(phi, 2, axis=0)

        forward_x =     0.5*(-3*phi     + 4*phi_ip1 - phi_ip2)/dx
        backward_x =    0.5*( 3*phi     - 4*phi_im1 + phi_im2)/dx

        forward_y =     0.5*(-3*phi     + 4*phi_jp1 - phi_jp2)/dx
        backward_y =    0.5*( 3*phi     - 4*phi_jm1 + phi_jm2)/dx

        phi_x = np.where(uval < 0, forward_x, backward_x)

        phi_y = np.where(vval < 0, forward_y, backward_y)

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        #return self.D*laplacian_phi - uval*phi_x - vval*phi_y + rr_arr/self.rho
        return self.D*laplacian_phi - uval*phi_x - vval*phi_y


    def _update_RK2_species(self, uv:VelocityField, species_field:Species):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """

        u_arr = uv.u.values
        v_arr = uv.v.values

        y0 = np.copy(species_field.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        #Q = self.progress_rate(ch4_field, o2_field, T_field)
        #species_field.update_reaction_rate(Q)
        species_field.values = y0 + 0.5*self.dt*self._f_species(u_arr, v_arr, species_field.values)
        species_field.fillGhosts()

        species_field.values = y0 + self.dt*self._f_species(u_arr, v_arr, species_field.values)
        species_field.fillGhosts()


    def _f_temperature(self, uval, vval, phi):
        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.dx

        # get the translation arrays to calculate further the derivatives estimates.
        # Translation along x axis, indexed by i.
        phi_ip1 = np.roll(phi, -1, axis=1)
        phi_ip2 = np.roll(phi, -2, axis=1)
        phi_im1 = np.roll(phi, 1, axis=1)
        phi_im2 = np.roll(phi, 2, axis=1)
        
        # Translation along y axis, indexed by j.
        phi_jp1 = np.roll(phi, -1, axis=0)
        phi_jp2 = np.roll(phi, -2, axis=0)
        phi_jm1 = np.roll(phi, 1, axis=0)
        phi_jm2 = np.roll(phi, 2, axis=0)

        forward_x =     0.5*(-3*phi     + 4*phi_ip1 - phi_ip2)/dx
        backward_x =    0.5*( 3*phi     - 4*phi_im1 + phi_im2)/dx

        forward_y =     0.5*(-3*phi     + 4*phi_jp1 - phi_jp2)/dx
        backward_y =    0.5*( 3*phi     - 4*phi_jm1 + phi_jm2)/dx

        phi_x = np.where(uval < 0, forward_x, backward_x)

        phi_y = np.where(vval < 0, forward_y, backward_y)

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        #return self.D*laplacian_phi - uval*phi_x - vval*phi_y + rr_arr/self.rho
        return self.a*laplacian_phi - uval*phi_x - vval*phi_y


    def _update_RK2_temperature(self, uv:VelocityField, temp_field:TemperatureField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """

        u_arr = uv.u.values
        v_arr = uv.v.values

        T0 = np.copy(temp_field.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        temp_field.values = T0 + 0.5*self.dt*self._f_temperature(u_arr, v_arr, temp_field.values)
        temp_field.fillGhosts()

        temp_field.values = T0 + self.dt*self._f_temperature(u_arr, v_arr, temp_field.values)
        temp_field.fillGhosts()


    def _converge_all_species(self, o2:Dioxygen, n2:Dinitrogen, ch4:Methane, h2o:Water, co2:CarbonDioxide, T_field:TemperatureField, dtchem:float): #, conv_crit_chem):

        rho = self.rho
        c_p = self.c_p
        #thick = self.ghost_thick

        #all_converg = np.full(5, 1.)
        #while np.sum(np.where(all_converg > conv_crit_chem, False, True)) < 5:
        o20 = np.copy(o2.values)
        ch40 = np.copy(ch4.values)
        T0 = np.copy(T_field.values)

        ## First step of RK2
        QoverA = self.progress_rate(o2, ch4, T_field)
        #misc.register_array_csv(f"{self.frame}_{self.chem_frame:02d}_step1_QoverA.csv", QoverA)
        #misc.plot_array(QoverA, title=f"Reaction rate at the beginning of k={self.frame}", saveaspng=f"{self.frame}_{self.chem_frame:02d}_step1_QoverA.png")
        A = 1.1e8
        o2.values = o20 + (0.5*dtchem*o2.stoech*o2.W*A/rho)*QoverA
        ch4.values = ch40 + (0.5*dtchem*ch4.stoech*ch4.W*A/rho)*QoverA
        T_field.values = T0 - (0.5*dtchem*A/(rho*c_p))*(ch4.stoech*ch4.Dhf + co2.stoech*co2.Dhf + h2o.stoech*h2o.Dhf)*QoverA
        #sys.exit()

        ## Second step of RK2
        QoverA = self.progress_rate(o2, ch4, T_field)
        #misc.register_array_csv(f"{self.frame}_{self.chem_frame:02d}_step2_QoverA.csv", QoverA)
        #misc.plot_array(QoverA, title=f"Reaction rate at the 2nd step of k={self.frame}", saveaspng=f"{self.frame}_{self.chem_frame:02d}_step2_QoverA.png")
        o2.values = o20 + (dtchem*o2.stoech*o2.W*A/rho)*QoverA
        #n2.values = n2.values + (dtchem*n2.stoech*n2.W*A/rho)*QoverA
        ch4.values = ch40 + (dtchem*ch4.stoech*ch4.W*A/rho)*QoverA
        h2o.values = h2o.values + (dtchem*h2o.stoech*h2o.W*A/rho)*QoverA
        co2.values = co2.values + (dtchem*co2.stoech*co2.W*A/rho)*QoverA
        T_field.values = T0 - (dtchem*A/(rho*c_p))*(ch4.stoech*ch4.Dhf + co2.stoech*co2.Dhf + h2o.stoech*h2o.Dhf)*QoverA
        #sys.exit()

        #omega_T = -(o2.Dhf*o2.stoech + n2.Dhf*n2.stoech + ch4.Dhf*ch4.stoech + h2o.Dhf*h2o.stoech + co2.Dhf*co2.stoech)*Q
        #T_field.values = T_field.values + (dtchem/(rho*c_p))*omega_T


    def progress_rate(self, o2:Dioxygen, ch4:Methane, T_field:TemperatureField):
        
        N = self.N
        rho = self.rho
        thick = self.ghost_thick
        Ta = self.Ta

        ch4_conc = (rho/ch4.W)*ch4.values[thick:-thick, thick:-thick]
        o2_conc = (rho/o2.W)*o2.values[thick:-thick, thick:-thick]

        #ch4_max = np.max(ch4_conc)
        ch4_investigate = np.argwhere(np.isnan(ch4_conc))
        #misc.plot_array(ch4_conc, title="CH4_conccentration", pause=3)
        #misc.register_array_csv(f"{self.frame}_{self.chem_frame:02d}_ch4_conc.csv", ch4_conc)
        #o2_max = np.max(o2_conc)
        o2_investigate = np.argwhere(np.isnan(o2_conc))
        #misc.plot_array(o2_conc, title="CH4_conccentration", pause=3)
        #misc.register_array_csv(f"{self.frame}_{self.chem_frame:02d}_o2_conc.csv", o2_conc)

        T_arr = T_field.values[thick:-thick, thick:-thick]
        exp_array = np.where(T_arr >= 100, np.exp(-Ta/T_arr), 0.) # Caution to counter from RuntimeWarning: overflow encountered in exp
        #misc.register_array_csv(f"{self.frame}_{self.chem_frame:02d}_exp_array.csv", exp_array)

        #exp_max = np.max(exp_array)
        exp_investigate = np.argwhere(np.isnan(exp_array))
        #misc.plot_array(exp_array, title="Exp array", pause=3)

        QoverA = np.zeros((N,N), dtype=float)
        QoverA[thick:-thick, thick:-thick] = np.multiply(o2_conc, o2_conc)*ch4_conc*exp_array
        
        #q_max = np.max(QoverA)
        q_investigate = np.argwhere(np.isnan(QoverA))

        return QoverA
    

    def compute(self):
        
        # Bring attributes closer as variables.
        N = self.N
        physN = self.physN
        dt = self.dt
        rho = self.rho
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        thick = self.ghost_thick
        frame_period = self.register_period

        # Initialize Velocity field
        uv = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uv.update_metric()

        # Initalize uvcopy
        uvcopy = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uv_consecutive_diff = 1.0

        # Initializing the V* field
        uvet = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uvet.update_metric()

        # Initializing the P field
        Press = PressureField(np.full((physN, physN), 1.0), dx, False, thick)

        # Initializing the species
        o2 = Dioxygen(np.full((physN, physN), 0.2, dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        n2 = Dinitrogen(np.full((physN, physN), 0.8 ,dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        ch4 = Methane(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        h2o = Water(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        co2 = CarbonDioxide(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)

        # Initializing the temperature
        Temp = TemperatureField(np.full((physN, physN), 300), dx, False, thick)

        #Initializing the thickness of the diffusive zone.
        diffz_thick = 0.

        # Set time to zero
        time = 0.
        
        start_datetime = datetime.datetime.now()
        stop_datetime = datetime.datetime.now()


        while uv_consecutive_diff >= self.conv_crit and uv.metric < self.div_crit and stop_datetime - start_datetime < self.max_t_comput:
            
            time = self.frame * dt

            if self.frame%frame_period == 0:
                
                if self.frame != 0:
                    uv_consecutive_diff = misc.velocity_derivative_norm(uvcopy, uv, dt)

                print(f"Frame=\t{self.frame:06}\t ; \tVirtual time={time:.2e} s\t;\tLast SOR nb iter={Press.last_nb_iter}\t;\tVelocity Residual={uv_consecutive_diff:.2e}")
                stop_datetime = datetime.datetime.now()

                if self.show_save:
                    misc.register_array_csv(f"{self.frame}_ch4.csv", ch4.values)
                    misc.plot_field(uv.v, True, title=f'V Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v_field.png")
                    misc.plot_field(uv.u, True, title=f'U Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u_field.png")
                    misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et_field.png")
                    misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et_field.png")
                    misc.plot_strain_rate(uv, self.y, title="Spatial representation of th strain rate along the slipping wall", saveaspng=str(self.frame)+"_strain_rate.png")
                    misc.plot_field(Press, True, title=f'Pressure Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_press_field.png")
                    misc.plot_field(o2, True, title=f'O2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_O2.png")
                    misc.plot_field(n2, True, title=f'N2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_N2.png")
                    misc.plot_field(ch4, True, title=f'CH4 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CH4.png")
                    misc.plot_field(h2o, True, title=f'H2O Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_H2O.png")
                    misc.plot_field(co2, True, title=f'CO2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CO2.png")
                    misc.plot_field(Temp, True, title=f'T° Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_temp_field.png")
                    misc.plot_diffusive_zone(n2, self.y, self.frame, dt, time)

            elif self.frame%frame_period == frame_period-1 :
                uvcopy = misc.uv_copy(uv)

            if time >= 2e-5:
                # This function ignites the middle of the box to 1000 K or more.
                Temp.ignite_or_not()

            # Temperature transport
            self._update_RK2_temperature(uv, Temp)
                
            # Population transport
            self._update_RK2_species(uv, o2)
            self._update_RK2_species(uv, n2)
            self._update_RK2_species(uv, ch4)
            self._update_RK2_species(uv, h2o)
            self._update_RK2_species(uv, co2)

            #ch4_investigate = np.argwhere(np.isnan(ch4.values))

            # Chemistry, impacting all species populations and temperature.
            i_reactor = self.i_reactor
            j_reactor = self.j_reactor
            suivitime = [0.]
            suivio2 = [o2.values[i_reactor, j_reactor]]
            suivin2 = [n2.values[i_reactor, j_reactor]]
            suivich4 = [ch4.values[i_reactor, j_reactor]]
            suivih2o = [h2o.values[i_reactor, j_reactor]]
            suivico2 = [co2.values[i_reactor, j_reactor]]
            suiviT = [Temp.values[i_reactor, j_reactor]]
            
            self.chem_frame = 0
            nchem = self.dtchem_list.shape[0]

            isworth = True
            while self.chem_frame < nchem and isworth:
                o2_values_0 = np.copy(o2.values)
                T_values_0 = np.copy(Temp.values)
                dtchem = self.dtchem_list[self.chem_frame]
                #print(f"\tChem. frame = {self.chem_frame}\t ; dtchem={dtchem:.3e}")
                self._converge_all_species(o2, n2, ch4, h2o, co2, Temp, dtchem)
                isworth = misc.worth_continue_chemistry(o2_values_0, o2.values, T_values_0, Temp.values, self.ghost_thick, dtchem)
                if self.frame%frame_period == 0:
                    suivitime.append(suivitime[-1]+dtchem)
                    suivio2.append(o2.values[i_reactor, j_reactor])
                    suivin2.append(n2.values[i_reactor, j_reactor])
                    suivich4.append(ch4.values[i_reactor, j_reactor])
                    suivih2o.append(h2o.values[i_reactor, j_reactor])
                    suivico2.append(co2.values[i_reactor, j_reactor])
                    suiviT.append(Temp.values[i_reactor, j_reactor])
                self.chem_frame += 1

            if self.frame%frame_period == 0:
                misc.plot_chemistry(suivitime, suivio2, suivich4, suivih2o, suivico2, suiviT, i_reactor, j_reactor, self.frame)

            o2.fillGhosts()
            n2.fillGhosts()
            ch4.fillGhosts()
            h2o.fillGhosts()
            co2.fillGhosts()
            Temp.fillGhosts()

            #misc.register_array_csv(f"{self.frame}_ch4_endofframe.csv", ch4.values)

            #sys.exit()

            # Fluid Flow
            self._update_RK2_velocity(uv, uvet)
            uvet.update_metric()
            #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et0_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et0_field.png")            
            self.ell_solver(uvet, Press)
            #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et1_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et1_field.png")            
            uv.u.values = uvet.u.values - (dt/rho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/rho)*Press.derivative_y()
            #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et2_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et2_field.png")            
            uv._fillGhosts()
            uv.update_metric()
            
            self.frame += 1

        # Plots of the final frame
        misc.plot_field(uv.v, True, title=f'V Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(self.frame)+"_v_field.png")
        misc.plot_field(uv.u, True, title=f'U Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(self.frame)+"_u_field.png")
        misc.plot_strain_rate(uv, self.y, title=f"Spatial representation of th strain rate along the slipping wall\nMax is {uv.max_strain_rate:.4e} Hz", saveaspng=str(self.frame)+"_strain_rate.png")
        misc.plot_field(Press, True, title=f'Pressure Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_press_field.png")
        misc.plot_field(o2, True, title=f'O2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_O2.png")
        misc.plot_field(n2, True, title=f'N2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_N2.png")
        misc.plot_field(ch4, True, title=f'CH4 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CH4.png")
        misc.plot_field(h2o, True, title=f'H2O Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_H2O.png")
        misc.plot_field(co2, True, title=f'CO2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CO2.png")
        misc.plot_field(Temp, True, title=f'T° Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_temp_field.png")
        misc.plot_diffusive_zone(n2, self.y, self.frame, dt, time)

        # About to print and write in final file the report of the simulation.
        if uv.metric >= self.div_crit :
            code = "divergence"        
        elif stop_datetime - start_datetime >= self.max_t_comput:
            code = "timeout"
        else:
            code = "success"

        misc.print_write_end_message(code, self.div_crit, self.max_t_comput, self.conv_crit, self.L, self.D, N, dt, stop_datetime-start_datetime, time, self.frame, uv_consecutive_diff, uv.max_strain_rate, n2.diff_zone_thick)

