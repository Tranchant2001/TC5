import math
import pickle
import datetime

class SimuParameters:

    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, nu:float, D:float, a:float, rho:float, c_p:float, Ta:float, time_ignite:float, register_period:int, i_reactor:int, j_reactor:int, ell_crit:float, div_crit:float, uv_conv_crit:float, temp_conv_crit:float, max_t_comput:datetime.timedelta, show_save:bool, datapath:str):
        
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
        self.register_period = register_period
        self.i_reactor = i_reactor
        self.j_reactor = j_reactor
        self.ell_crit = ell_crit
        self.div_crit = div_crit
        self.uv_conv_crit = uv_conv_crit
        self.temp_conv_crit = temp_conv_crit
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.datapath = datapath

        self.dx = L/(physN-1)
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.4*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step
        #self.dt = 4e-6

        self.Temp_maxT_flame = None
        self.uv_max_strain_rate = None
        self.n2_diff_zone_thick = None
        
        self.omega = 2*(1 - math.pi/physN - (math.pi/physN)**2)

        self.ghost_thick = 2
        self.N = physN + 2*self.ghost_thick

    
    def save_as_pickle(self):
        with open(self.datapath+"\\simu_params.pickle", "wb") as filehandler:
            pickle.dump(self, filehandler)