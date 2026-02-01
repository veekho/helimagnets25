#Calculate the localaisation of the Wannier function corresponding each Bloch state at a range of spin-orbit coupling strengths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import linalg
import hamiltonian1d as hm

def d_dk(data, dk):
    return (np.roll(data, -1, 0) - np.roll(data, +1, 0))/(2*dk)

lattice_const = 1 #angstrom, a
exchange_coupling = 0.4 #eV, lambda
hopping_amplitude = 1 #eV, t
chirality = +1 # , gamma
moment_magnitude = 1 # , M
angle = 0#np.pi/4 #radians, alpha

no_kx_points = 1000
max_kx = np.pi/(4*lattice_const)
min_kx = -np.pi/(4*lattice_const)
d_kx = (max_kx-min_kx)/no_kx_points
kx_range = np.linspace(min_kx, max_kx, no_kx_points)

'''
#Single calculation
rashba_const = 2
eig_vecs = np.zeros((no_kx_points,8,8), dtype=np.complex128)
for i, kx in enumerate(kx_range):
    eig_vec, eig_val = linalg.eigh(hm.hamiltonian(kx, exchange_coupling, hopping_amplitude, chirality, moment_magnitude, angle, rashba_const))
    eig_vecs[i] = eig_vec

d_eigvecs = d_dk(eig_vecs, d_kx)
d2_eigvecs = d_dk(d_eigvecs, d_kx)

avg_r = 1j*4*lattice_const/(2*np.pi) * np.sum(np.sum(np.conj(eig_vecs)*d_eigvecs, 1),0)*d_kx #Imaginary parts for interband hopping
avg_r2 = -4*lattice_const/(2*np.pi) * np.sum(np.sum(np.conj(eig_vecs)*d2_eigvecs, 1),0)*d_kx #This is real and dominates
var = avg_r2 - avg_r**2

for n in range(len(avg_r)):
    print(f"<r> = {avg_r[n]:3e}, <r2> = {avg_r2[n]:3e}, var = {var[n]:3e}")

#'''
#Range of spin-orbit coupling strengths
no_datapoints = 50
rashba_const_list = np.linspace(0, 3, no_datapoints)
localisations = np.zeros((0, 8))

for rashba_const in rashba_const_list:

    eig_vecs = np.zeros((no_kx_points,8,8), dtype=np.complex128)
    
    for i, kx in enumerate(kx_range):
        eig_vec, eig_val = linalg.eigh(hm.hamiltonian(kx, exchange_coupling, hopping_amplitude, chirality, moment_magnitude, angle, rashba_const))
        eig_vecs[i] = eig_vec

    d_eigvecs = d_dk(eig_vecs, d_kx)
    d2_eigvecs = d_dk(d_eigvecs, d_kx)

    avg_r = 1j*4*lattice_const/(2*np.pi) * np.sum(np.sum(np.conj(eig_vecs)*d_eigvecs, 1),0)*d_kx #Imaginary parts for interband hopping
    avg_r2 = -4*lattice_const/(2*np.pi) * np.sum(np.sum(np.conj(eig_vecs)*d2_eigvecs, 1),0)*d_kx #This is real and dominates
    var = avg_r2 - avg_r**2

    localisations = np.vstack((localisations, var.real))


cmap = cm.hot.reversed()
norm = colors.Normalize(vmin=0, vmax=np.max(localisations))

localisations = np.transpose(localisations)
X, Y = np.meshgrid(rashba_const_list, np.arange(1, 9))

fig = plt.figure(figsize=(9, 2))
ax = fig.add_subplot(111)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\sigma_r ^2 (a^2)$')
ax.pcolormesh(X, Y, localisations, cmap=cmap, norm=norm)
ax.set_ylabel("Band")
ax.set_xlabel(r'$\lambda_R$ (eV$\cdot$'+u'\u212B)')

print(np.max(localisations))
plt.show()

#'''