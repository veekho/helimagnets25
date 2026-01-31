#Produce a gif of the folded band structure while varying a value
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, animation
from scipy import linalg
import hamiltonian1d as hm

#Constants
exchange = 0.4 #eV, lambda
chirality = +1
moment_mag = 1 # M
hopping = 1
angle = np.pi/4 #radians, alpha
lattice_const = 1 #angstrom
#rashba_const = 2 #eV angstrom, lambda_R

chain_length = 4
hm.set_chainlength(chain_length)

n_datapoints = 500
fig = plt.figure()
ax = fig.add_subplot(111)

norm = colors.Normalize(vmin=-1, vmax=+1)
cmap=cm.bwr
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\langle S_z \rangle$")
cbar.set_ticks(ticks=[-1, 0, 1], labels=['-1', '0', '+1'])

frame_rate = 15 #fps
no_frames = 250
frame_count = 0

def update(rashba_const):
    global frame_count, no_frames

    print(f"Plotting frame {frame_count}/{no_frames}")
    frame_count+=1

    data = np.zeros((0,3)) #[[wavenumber, eigenvalue, spin],]
    for k in np.linspace(-np.pi/(4*lattice_const), +np.pi/(4*lattice_const), n_datapoints):
        eig_val, eig_vec = linalg.eigh(hm.hamiltonian(k, exchange, hopping, chirality, moment_mag, angle, rashba_const, lattice_const))

        spin_polarisation = np.sum(np.conj(eig_vec)*(hm.spin_z@eig_vec), 0).real
        data = np.vstack((data, np.transpose([k*np.ones(2*chain_length), eig_val, spin_polarisation])))

    ax.clear()
    ax.grid(True, "major")
    ax.scatter(data[:,0], data[:,1], marker='.', s=2, c = data[:,2], cmap=cmap, norm=norm)
    ax.set_xticks([-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4], labels=[r'$-\pi/4a$', '', '0', '', r'$+\pi/4a$'])
    ax.set_xlabel(u"k (\u212B"+r' ^-1)')
    ax.set_ylabel("E(k) (eV)")
    ax.set_title(rf"$\lambda_R =$ {rashba_const:.2f} eV$\cdot$"+u'\u212B')

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2.5, no_frames))
ani.save("./1D/plots/rashba_const_scan.gif", fps=frame_rate)