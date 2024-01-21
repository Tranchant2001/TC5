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

def RK2_method(o2:Dioxygen, ch4:Methane, h2o:Water,co2:CarbonDioxide, Temp:TemperatureField, dtchemlist, rho, c_p):
    
    thick = o2.ghost_thick
    physN = o2.N - 2*thick

    nchem = dtchemlist.shape[0]

    for i in range(physN):
        for j in range(physN):
            isworth = True
            frame_chem = 0
            o20 = o2.values[i,j]
            ch40 = ch4.values[i,j]
            h2o0 = h2o.values[i,j]
            co20 = co2.values[i,j]
            Temp0 = Temp.values[i,j]
            while frame_chem < nchem and isworth:
                
                dtchem = dtchemlist[frame_chem]

                Q = progress_rate(o2.values[i,j], ch4.values[i,j], Temp.values[i,j])
                o2.values[i,j] = o20 + (0.5*dtchem*o2.stoech*o2.W/rho)*Q
                ch4.values[i,j] = ch40 + (0.5*dtchem*ch4.stoech*ch4.W/rho)*Q
                Temp.values[i,j] = Temp0 - (0.5*dtchem/(rho*c_p))*(ch4.stoech*ch4.Dhf + co2.stoech*co2.Dhf + h2o.stoech*h2o.Dhf)*Q

                Q = progress_rate(o2.values[i,j], ch4.values[i,j], Temp.values[i,j])
                o2.values[i,j] = o20 + (dtchem*o2.stoech*o2.W/rho)*Q
                ch4.values[i,j] = ch40 + (dtchem*ch4.stoech*ch4.W/rho)*Q
                h2o.values[i,j] = h2o0 + (dtchem*h2o.stoech*h2o.W/rho)*Q
                co2.values[i,j] = co20 + (dtchem*co2.stoech*co2.W/rho)*Q
                Temp.values[i,j] = Temp0 - (dtchem/(rho*c_p))*(ch4.stoech*ch4.Dhf + co2.stoech*co2.Dhf + h2o.stoech*h2o.Dhf)*Q

                isworth = is_worth_continue(o20, o2.values[i,j], Temp0, Temp.values[i,j], dtchem)

                o20 = o2.values[i,j]
                ch40 = ch4.values[i,j]
                h2o0 = h2o.values[i,j]
                co20 = co2.values[i,j]
                Temp0 = Temp.values[i,j]                


def progress_rate(o2_val, ch4_val, T_val, Ta, rho):

        A = 1.1e8
        W_o2 = 31.999e-3
        W_ch4 = 16.04e-3

        ch4_conc = (rho/W_ch4)*ch4_val
        o2_conc = (rho/W_o2)*o2_val

        Q = A*(o2_conc**2)*ch4_conc*np.exp(-Ta/T_val)

        return Q

def is_worth_continue(o2_previous, o2_current, Temp_previous, Temp_current, dtchem):
    max_derive_o2 = abs((o2_current - o2_previous)/dtchem)
    max_deriv_T = abs((Temp_current - Temp_previous)/dtchem)

    return (max_derive_o2 > 50. or max_deriv_T > 5e5)    