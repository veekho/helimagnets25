#Plots folded band structure for 1D helimagnet
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import hamiltonian1d as hm

rashba_const = 2 #eV/angstrom, lambda_R
lattice_const = 1 #angstrom, a
exchange_coupling = 0.4 #eV, lambda
hopping_amplitude = 1 #eV, t
chirality = +1 # , gamma
moment_magnitude = 1 # , M
angle = np.pi/4 #radians, alpha

min_kx = -np.pi/(4*lattice_const)
max_kx = +np.pi/(4*lattice_const)
no_datapoints = 1000

data = np.zeros((0,5)) #kx, E(kx), spin_x, spin_y, spin_z

for kx in np.linspace(min_kx, max_kx, no_datapoints):
    eig_val, eig_vec = linalg.eigh(hm.hamiltonian(kx, exchange_coupling, hopping_amplitude, chirality, moment_magnitude, angle, rashba_const, lattice_const, rsoc_type='linear'))

    spin_polarisation_x = np.sum(np.conj(eig_vec)*(hm.spin_x@eig_vec), 0).real
    spin_polarisation_y = np.sum(np.conj(eig_vec)*(hm.spin_y@eig_vec), 0).real
    spin_polarisation_z = np.sum(np.conj(eig_vec)*(hm.spin_z@eig_vec), 0).real

    data = np.vstack((data, np.transpose([kx*np.ones(8), eig_val, spin_polarisation_x, spin_polarisation_y, spin_polarisation_z])))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True, 'major')
cmap = cm.bwr
norm = colors.Normalize(vmin=-1, vmax=+1)
ax.scatter(data[:,0], data[:,1], marker='.', s=2, c=data[:,4], cmap=cmap, norm=norm)
ax.set_xticks([min_kx, min_kx/2, 0, max_kx/2, max_kx], labels=[r'$-\pi/4a$', '', '0', '', r'$+\pi/4a$'])
ax.set_ylabel(r'$E(k_x)$ (eV)')
ax.set_xlabel(r'$k_x$ ('+u'\u212B ^-1)')
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\langle S_z \rangle$")
cbar.set_ticks(ticks=[-1, 0, 1], labels=['-1', '0', '+1'])
plt.show()