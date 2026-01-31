#Plot unfolded band structure for 1D helimagnet
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import hamiltonian1d as hm

#Constants
exchange = 0.4 #eV, lambda
hopping = 1 #eV, t
moment_mag = 1 # M
angle = 0*np.pi/4 #radians, alpha
rashba_const = 0 #eV, lambda_R
lattice_const = 1 #angstrom

#For a range of wavenumbers
n_datapoints = 1000
data = np.zeros((0,3)) #[[wavenumber, eigenvalue, spin],]
unfold_weight = np.zeros((0,1))

for chirality in [+1]:
#for chirality in [-1, +1]: #Include both chiralities only if in helical phase of non spin-orbit coupled because band structure is degenerate
    for k in np.linspace(-np.pi/lattice_const, np.pi/lattice_const, n_datapoints):
        eig_val, eig_vec = linalg.eigh(hm.hamiltonian(k, exchange, hopping, chirality, moment_mag, angle, rashba_const))
        spin_polarisation = np.sum(np.conj(eig_vec)*(hm.spin_z@eig_vec), 0).real

        unfold_sum = np.sum(eig_vec, 0)
        unfold_weight = np.append(unfold_weight, (unfold_sum.conj()*unfold_sum).real/len(eig_val))
        data = np.vstack((data, np.transpose([k*np.ones(len(eig_val)), eig_val, spin_polarisation])))

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

'''
#Show spin polarisation
cmap_spin = cm.bwr
norm_spin = colors.Normalize(vmin=-1, vmax=+1)
ax.scatter(data[:,0], data[:,1], marker='.', s=2, c=data[:,2], cmap=cmap_spin, norm=norm_spin, alpha=unfold_weight)
fig.colorbar(cm.ScalarMappable(norm=norm_spin, cmap=cmap_spin), ax=ax, label="Sz")
'''
#Show unfold weight
cmap_unfold=colors.LinearSegmentedColormap.from_list("white-blue", ["#ffffff", "#0000ff"])
norm_unfold=colors.Normalize(vmin=0, vmax=+1)
ax.scatter(data[:,0], data[:,1], marker='.', s=2, c=unfold_weight, cmap=cmap_unfold, norm=norm_unfold)
fig.colorbar(cm.ScalarMappable(norm=norm_unfold, cmap=cmap_unfold), ax=ax, label=r"w_k")
#'''

ax.set_xticks([-np.pi/lattice_const, -np.pi/(4*lattice_const), 0, +np.pi/(4*lattice_const), np.pi/lattice_const],
             labels=[r'$-\frac{\pi}{a}$', r'$-\frac{\pi}{4a}$', '0', r'$+\frac{\pi}{4a}$', r'$+\frac{\pi}{a}$'])
ax.grid(True, "major")

ylim_lower, ylim_upper = ax.get_ylim()
ax.plot(np.pi*np.ones(10)/4,np.linspace(ylim_lower, ylim_upper, 10), 'k--', alpha=0.5, label='Edge of supercell BZ')
ax.plot(-np.pi*np.ones(10)/4,np.linspace(ylim_lower, ylim_upper, 10), 'k--', alpha=0.5)
ax.plot(np.pi*np.ones(10),np.linspace(ylim_lower, ylim_upper, 10), 'k:', alpha=0.5, label='Edge of primitive BZ')
ax.plot(-np.pi*np.ones(10),np.linspace(ylim_lower, ylim_upper, 10), 'k:', alpha=0.5)

ax.set_ylabel("E(k)")
ax.set_xlabel(r"$k_x$")
fig.legend(bbox_to_anchor=(0.84, 0.98))
plt.show()