import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
import h5py



#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
projectpath = os.path.dirname(fullpath)
# Chemin des data
datapath = projectpath + "\\outputs\\Data"
# Chemin des figures
figpath = projectpath + "\\outputs\\Figures"
# Animation path
animpath = projectpath+ "\\outputs\\Animations"



# Collecting the parameters object
with open(datapath+"\\simu_params.pickle", "rb") as filehandler:
    theparams = pickle.load(filehandler)

# Create a figure and a set of axes
fig, ax = plt.subplots(1,2, figsize=(16,8), gridspec_kw={'width_ratios':[1, 1.2]}, dpi=96)

# Create some data
x = np.linspace(0, theparams.L, theparams.physN, endpoint=True, dtype=np.float32)
y = np.linspace(0, theparams.L, theparams.physN, endpoint=True, dtype=np.float32)
X, Y = np.meshgrid(x, y)
init_array = np.zeros((theparams.physN, theparams.physN), dtype=float)

# Create a pcolor plot
pc0 = ax[0].pcolor(X, Y, init_array, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
ax[0].set_xlabel("X (m)")
ax[0].set_ylabel("Y (m)")
ax[0].set_title(f"$v_x$")
ax[0].ticklabel_format(style="sci", scilimits=(-1,1), useMathText=True)
pc1 = ax[1].pcolor(X, Y, init_array, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
ax[1].set_xlabel("X (m)")
ax[1].set_ylabel("Y (m)")
ax[1].set_title(f"$v_y$")
ax[1].ticklabel_format(style="sci", scilimits=(-1,1), useMathText=True)
fig.colorbar(pc1)

def init():
    init_array = np.zeros((theparams.physN, theparams.physN), dtype=float)
    pc0.set_array(init_array.ravel())
    pc1.set_array(init_array.ravel())
    return pc0, pc1


def animate(i):
    # Read the data:
    filename_i = f"{i}_all_fields.h5"
    with h5py.File(datapath+"\\"+filename_i, "r") as h5file:
        datagroup = h5file.get('Fields')
        dataset0 = datagroup.get("uv_u_values")
        dataset1 = datagroup.get("uv_v_values")
        thick = datagroup.attrs["ghost_thick"]
        Z0 = np.array(dataset0)
        Z1 = np.array(dataset1)
        del datagroup
        del dataset0
        del dataset1
    # Strip the array of its ghost cells
    Z0 = Z0[thick:-thick , thick:-thick]
    Z1 = Z1[thick:-thick , thick:-thick]    
    # Update the data
    pc0.set_array(Z0.ravel())
    pc1.set_array(Z1.ravel())

    return pc0, pc1

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=True)

# Save the animation
ani.save(animpath+'\\velocity_field.mp4', fps=20, extra_args=['-vcodec', 'libx264'])