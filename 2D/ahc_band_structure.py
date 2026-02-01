#Copy of own script: ahc_band_structure.py
#Calculate AHC from saved data and plot alongside band structure for a single file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import linalg, constants
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons
import hamiltonian2d as hm

rashba_const = 2 #eV/angstrom, lambda_R
rashba_type = "linear"
lattice_const = 1 #angstrom, a
exchange_coupling = 0.4 #eV, lambda
hopping_amplitude = 1 #eV, t
chirality = +1 # , gamma
moment_magnitude = 1 # , M
angle = np.pi/4 #radians, alpha
interchain=0

t_x = hopping_amplitude
t_y = hopping_amplitude
a_x = lattice_const
a_y = lattice_const
chain_length = 4

fermi_levels = 500
axis_len_half = 100
axis_len_x = 2*axis_len_half
axis_len_y = 2*chain_length*axis_len_half
min_kx = -np.pi/(chain_length*a_x)
max_kx = np.pi/(chain_length*a_x)
min_ky = -np.pi/a_y
max_ky = np.pi/a_y

pauli_x = np.array([[0, 1],
                    [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -1j],
                    [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0],
                    [0, -1]], dtype=np.complex128)

spin_operator = np.zeros((4*chain_length, 4*chain_length), dtype=np.complex128)
for i in range(2*chain_length):
    spin_operator+=np.pad(pauli_z, (2*i, 2*(2*chain_length-1-i)))

def fermi_dirac(energy, temp, fermi_energy):
    return 1/(np.exp((energy-fermi_energy)*constants.electron_volt/(temp*constants.Boltzmann)) + 1)

def d_dKx(data, interval):
    return (np.roll(data, -1, 1) - np.roll(data, +1, 1))/(2*interval)
def d_dKy(data, interval):
    return (np.roll(data, -1, 0) - np.roll(data, +1, 0))/(2*interval)

delta_kx = (max_kx - min_kx)/axis_len_x
delta_ky = (max_ky - min_ky)/axis_len_y
Kx, Ky = np.meshgrid(np.linspace(min_kx, max_kx, axis_len_x), np.linspace(min_ky, max_ky, axis_len_y))

hams = np.zeros(np.shape(Kx)+(4*chain_length, 4*chain_length), dtype=np.complex128)
results_eigvecs = np.zeros(np.shape(Kx)+(4*chain_length, 4*chain_length), dtype=np.complex128)
results_eigvals = np.zeros(np.shape(Kx)+(4*chain_length,))
spin_projections = np.zeros(np.shape(Kx)+(4*chain_length,))

print("Calculating eigensolutions")
for i in range(axis_len_x):
    for j in range(axis_len_y):
        hams[j,i] = hm.hamiltonian(Kx[j, i], Ky[j, i], exchange_coupling, t_x, t_y, chirality, moment_magnitude, angle, rashba_const, rashba_type, a_x, a_y, interchain)
        eig_val, eig_vec = linalg.eigh(hams[j,i]) #Must be in order of ascending eigenvalue
        results_eigvecs[j,i,:,:] = eig_vec
        results_eigvals[j,i,:] = eig_val
        spin_projections[j,i,:] = np.sum(np.conj(eig_vec)*spin_operator@eig_vec, 0).real

#eig_product = np.reshape(results_eigvals, np.shape(Kx)+(1,4*chain_length))*results_eigvecs
eig_product = hams@results_eigvecs
berry_curvature_contributions = np.zeros(np.shape(Kx)+(4*chain_length,)) #[kx,ky,bands]

print("Calculating Berry curvature contributions")
'''
#Velocity operator AB and A methods
for eig_index in range(4*chain_length):        
    eig_vec_local = np.reshape(results_eigvecs[:,:,:,eig_index], np.shape(Kx)+(4*chain_length,1)) #[kx,ky,16,1]
    eig_vec_other = np.delete(results_eigvecs, eig_index, 3) #[kx,ky,16,15]

    #eig_prod_local = np.reshape(eig_product[:,:,:,eig_index], np.shape(Kx)+(4*chain_length,1))
    #eig_prod_other = np.delete(eig_product, eig_index, 3)
    eig_prod_local = hams@eig_vec_local
    eig_prod_other = hams@eig_vec_other
        
    vx_part_a = np.sum(np.conj(eig_vec_local)*d_dKx(eig_prod_other, delta_kx), 2)
    vy_part_a = np.sum(np.conj(eig_vec_other)*d_dKy(eig_prod_local, delta_ky), 2)
    vx_part_b = np.sum(np.conj(eig_vec_other)*d_dKx(eig_prod_local, delta_kx), 2)
    vy_part_b = np.sum(np.conj(eig_vec_local)*d_dKy(eig_prod_other, delta_ky), 2)

    denominator = np.square( np.delete(results_eigvals, eig_index, 2) - np.reshape( results_eigvals[:,:,eig_index], (np.shape(Kx)+(1,)) ) ) #[100,100,15]
    
#AB Method
    #berry_curvature_contributions[:,:,eig_index] = -2*np.sum(np.imag(vx_part_a*vy_part_a - vx_part_b*vy_part_b)/denominator, 2)
#A Method
    berry_curvature_contributions[:,:,eig_index] = -2*np.sum(np.imag(vx_part_a*vy_part_a)/denominator, 2)
#'''
'''
#Basic Berry connection method
x_berry_connection = 1j*np.sum(np.conj(results_eigvecs)*d_dKx(results_eigvecs, delta_kx), 2)
y_berry_connection = 1j*np.sum(np.conj(results_eigvecs)*d_dKy(results_eigvecs, delta_ky), 2)
berry_curvature_contributions = ( d_dKx(y_berry_connection, delta_kx) - d_dKy(x_berry_connection, delta_ky) ).real
#'''
#'''
#Full velocity operator method
for eig_index in range(4*chain_length):        
    eig_vec_local = np.reshape(results_eigvecs[:,:,:,eig_index], np.shape(Kx)+(4*chain_length,1)) #[kx,ky,16,1]
    eig_vec_other = np.delete(results_eigvecs, eig_index, 3) #[kx,ky,16,15]

    vx_part_a = np.sum(np.conj(eig_vec_local)*d_dKx(hams@eig_vec_other, delta_kx), 2)
    vx_part_b = np.sum(np.conj(eig_vec_local)*hams@d_dKx(eig_vec_other, delta_kx), 2)
    vy_part_a = np.sum(np.conj(eig_vec_other)*d_dKy(hams@eig_vec_local, delta_ky), 2)
    vy_part_b = np.sum(hams@d_dKy(eig_vec_local, delta_ky)*np.conj(eig_vec_other), 2)

    denominator = np.square( np.delete(results_eigvals, eig_index, 2) - np.reshape( results_eigvals[:,:,eig_index], (np.shape(Kx)+(1,)) ) ) #[100,100,15]
    
    berry_curvature_contributions[:,:,eig_index] = -2*np.sum(np.imag((vx_part_a-vx_part_b)*(vy_part_a-vy_part_b))/denominator, 2)

#'''

np.nan_to_num(berry_curvature_contributions, copy=False, posinf=1e40, neginf=-1e40)
ahc = np.zeros((fermi_levels))
fermi_energies = np.linspace(np.min(results_eigvals), np.max(results_eigvals), fermi_levels)

print("Calculating anomalous Hall conductivities")
#for fermi_level, fermi_energy in enumerate(fermi_energies):
#    berry_curvature = np.sum(np.heaviside(fermi_energy-results_eigvals,0.5)*berry_curvature_contributions, 2)
#    #berry_curvature = np.sum(fermi_dirac(results_eigvals, plot_temp, fermi_energy)*berry_curvature_contributions, 2)
#    ahc[fermi_level] = - constants.elementary_charge**2/constants.hbar * np.sum(berry_curvature)/(2*np.pi)**2 * delta_kx*delta_ky / 100 #in e^2/hbar
berry_curvature = np.zeros(np.shape(Kx)+(fermi_levels,))
for fermi_level, fermi_energy in enumerate(fermi_energies):
    berry_curvature[:,:,fermi_level] = np.sum(np.heaviside(fermi_energy-results_eigvals,0.5)*berry_curvature_contributions, 2)
    #ahc[fermi_level] = - constants.elementary_charge**2/constants.hbar * np.sum(berry_curvature[:,:,fermi_level])/(2*np.pi)**2 * delta_kx*delta_ky / 100 #in ohms-1 cm-2
    ahc[fermi_level] = - np.sum(berry_curvature[:,:,fermi_level])/(2*np.pi)**2 * delta_kx*delta_ky #in e^2/hbar

#ahc = - constants.elementary_charge**2/constants.hbar * np.sum(np.sum(berry_curvature[:,:,fermi_level], 0),0)/(2*np.pi)**2 * delta_kx*delta_ky / 100

fig = plt.figure(figsize=(12,5))
gs = fig.add_gridspec(1,20)
#fig_ahc = plt.figure(figsize=(6,7))
#ax_ahc = fig_ahc.add_subplot(111)
ax_ahc = fig.add_subplot(gs[0,:6])
ax_ahc.grid(True)
ax_ahc.plot(ahc, fermi_energies, 'k-')
ax_ahc.set_ylabel("Energy (eV)")
#ax_ahc.set_xlabel(r"$\sigma_{xy} (\Omega^-1 cm^-1)$")
ax_ahc.set_xlabel(r"$\sigma_{xy} (e^2/\hbar)$")



# K4--M4--K1
# |\  |  /|
# | \ | / |
# |  \|/  |
# M3--G---M1
# |  /|\  |
# | / | \ |
# |/  |  \|
# K3--M2--K2

print("Sampling band structure")
plot_data = np.zeros((0,3)) #momentum, energy, spin
plot_axis_labels = np.zeros((0,2))
mid_axis_y = int(axis_len_y/2)
mid_axis_x = int(axis_len_x/2)
axis_ratio = int(axis_len_y/axis_len_x)

plot_k = 0
plot_axis_labels = []
plot_axis_ticks = []

print("\tM1 -> Gamma")
plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$M1$')
delta_k = delta_kx
for index in reversed(range(axis_len_half, 2*axis_len_half)):
    #spins = np.sum(results_eigvecs[chain_length*axis_len_half, index]*(spin_operator@results_eigvecs[chain_length*axis_len_half, index]), 0).real
    spins = spin_projections[chain_length*axis_len_half, index]
    energies = results_eigvals[chain_length*axis_len_half, index]
    momenta = np.ones(4*chain_length)*plot_k
    plot_k+=delta_k
    plot_data = np.vstack((plot_data, np.transpose([momenta, energies, spins])))
    
print("\tGamma -> M3")
plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$\Gamma$')
delta_k = delta_kx
for index in reversed(range(0, axis_len_half)):
    #spins = np.sum(results_eigvecs[chain_length*axis_len_half, index]*(spin_operator@results_eigvecs[chain_length*axis_len_half, index]), 0).real
    spins = spin_projections[chain_length*axis_len_half, index]
    energies = results_eigvals[chain_length*axis_len_half, index]
    momenta = np.ones(4*chain_length)*plot_k
    plot_k+=delta_k
    plot_data = np.vstack((plot_data, np.transpose([momenta, energies, spins])))

print("\tM3 -> K4")
plot_axis_ticks.append(plot_k)
plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$M3$')
delta_k = delta_ky
for index in reversed(range(0, axis_len_half)):
    #spins = np.sum(results_eigvecs[chain_length*index, 0]*(spin_operator@results_eigvecs[chain_length*index, 0]), 0).real
    spins = spin_projections[chain_length*index, 0]
    energies = results_eigvals[chain_length*index, 0]
    momenta = np.ones(4*chain_length)*plot_k
    plot_k+=delta_k
    plot_data = np.vstack((plot_data, np.transpose([momenta, energies, spins])))

print("\tK4 -> G")
plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$K4$')
delta_k = np.sqrt(delta_kx**2 + delta_ky**2)
for index in range(0, axis_len_half):
    #spins = np.sum(results_eigvecs[chain_length*index, index]*(spin_operator@results_eigvecs[chain_length*index, index]), 0).real
    spins = spin_projections[chain_length*index, index]
    energies = results_eigvals[chain_length*index, index]
    momenta = np.ones(4*chain_length)*plot_k
    plot_k+=delta_k
    plot_data = np.vstack((plot_data, np.transpose([momenta, energies, spins])))
    
print("\tG -> K2")
plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$\Gamma$')
delta_k = np.sqrt(delta_kx**2 + delta_ky**2)
for index in range(axis_len_half, 2*axis_len_half):
    #spins = np.sum(results_eigvecs[chain_length*index, index]*(spin_operator@results_eigvecs[chain_length*index, index]), 0).real
    spins = spin_projections[chain_length*index, index]
    energies = results_eigvals[chain_length*index, index]
    momenta = np.ones(4*chain_length)*plot_k
    plot_k+=delta_k
    plot_data = np.vstack((plot_data, np.transpose([momenta, energies, spins])))
    
print("\tK2 -> M1")
plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$K2$')
delta_k = delta_ky
for index in reversed(range(axis_len_half, 2*axis_len_half)):
    #spins = np.sum(results_eigvecs[chain_length*index, 2*axis_len_half-1]*(spin_operator@results_eigvecs[chain_length*index, 2*axis_len_half-1]), 0).real
    spins = spin_projections[chain_length*index, 2*axis_len_half-1]
    energies = results_eigvals[chain_length*index, 2*axis_len_half-1]
    momenta = np.ones(4*chain_length)*plot_k
    plot_k+=delta_k
    plot_data = np.vstack((plot_data, np.transpose([momenta, energies, spins])))

plot_axis_ticks.append(plot_k)
plot_axis_labels.append(r'$M1$')

cmap=colors.LinearSegmentedColormap.from_list("bgr", ["#0000ff", "#00ff00", "#ff0000"])
norm = colors.Normalize(vmin=-1, vmax=+1)
#color_scale = cm.ScalarMappable(norm=norm, cmap=cmap)
color_scale = cm.ScalarMappable(norm=norm, cmap=cm.bwr)


#fig_bands = plt.figure(figsize=(12,7))
#ax_bands = fig_bands.add_subplot(111)
ax_bands = fig.add_subplot(gs[0,7:], sharey=ax_ahc)
ax_bands.scatter(plot_data[:,0], plot_data[:,1], marker='.', s=2, c=plot_data[:,2], cmap=cmap, norm=norm)
#for i in range(8):
#    ax_bands.plot(plot_data[i::8,0], plot_data[i::8,1], c=plot_data[i::8,2], linestyle='-', cmap=cmap, norm=norm)
ax_bands.set_xticks(plot_axis_ticks, labels=plot_axis_labels)
cbar = fig.colorbar(color_scale, ax=ax_bands, label=r"$\langle S_x \rangle$")
cbar.set_ticks(ticks=[-1, 0, 1], labels=['-1', '0', '+1'])
ax_bands.grid(True, which='major')

plot_fermi_level = int(fermi_levels/2)
plot_minKx = min_kx
plot_maxKx = max_kx
plot_minKy = min_ky
plot_maxKy = max_ky

fig_bc = plt.figure()
ax_bc = fig_bc.add_subplot(111)
cmap_bc = cm.bwr
norm_bc = colors.Normalize(vmin=np.min(-abs(berry_curvature)), vmax=np.max(abs(berry_curvature)))
fig_bc.subplots_adjust(bottom=0.3, right=0.75)
ax_fermi_slider = fig_bc.add_subplot([0.12, 0.15, 0.78, 0.03])
fermi_slider = Slider(ax_fermi_slider, 'Fermi level', 0, fermi_levels-1, valstep=1, valinit=plot_fermi_level)
ax_text_minKx = fig_bc.add_axes([0.1, 0.05, 0.1, 0.1])
ax_text_maxKx = fig_bc.add_axes([0.3, 0.05, 0.1, 0.1])
ax_text_minKy = fig_bc.add_axes([0.5, 0.05, 0.1, 0.1])
ax_text_maxKy = fig_bc.add_axes([0.7, 0.05, 0.1, 0.1])
text_minKx = TextBox(ax_text_minKx, "Kx min", textalignment="left")
text_maxKx = TextBox(ax_text_maxKx, "Kx max", textalignment="left")
text_minKy = TextBox(ax_text_minKy, "Ky min", textalignment="left")
text_maxKy = TextBox(ax_text_maxKy, "Ky max", textalignment="left")
ax_plot_button = fig_bc.add_axes([0.85, 0.05, 0.1, 0.1])
plot_button = Button(ax_plot_button, "PLOT")

fig_surface = plt.figure()
ax_surface = fig_surface.add_subplot(projection='3d')
#fig_surface.subplots_adjust(right=0.75)
surface_visibility = {}
for i in range(4*chain_length):
    surface_visibility[str(i+1)] = True
ax_checkbuttons = fig_bc.add_subplot([0.8, 0.2, 0.1, 0.6])
check = CheckButtons(ax_checkbuttons, list(surface_visibility.keys()), list(surface_visibility.values()))
for band in range(4*chain_length):
    ax_surface.plot_surface(Kx, Ky, results_eigvals[:,:,band], facecolors=color_scale.to_rgba(spin_projections[:-1, :-1, band]), linewidth=0.1, alpha=0.5)
ax_surface.set_xlabel("Kx")
ax_surface.set_ylabel("Ky")
ax_surface.set_zlabel("Energy (eV)")

def reset_visibility(label):
    global surface_visibility
    surface_visibility[label] = not surface_visibility[label]

def plot_surface(idk):
    global plot_minKx, plot_maxKx, plot_minKy, plot_maxKy, surface_visibility
    print(f"Beginning replot: kx=[{plot_minKx:.3f},{plot_maxKx:.3f}], ky=[{plot_minKy:.3f},{plot_maxKy:.3f}]")

    if min_kx<plot_minKx<plot_maxKx<max_kx and min_ky<plot_minKy<plot_maxKy<max_ky:
        ax_surface.clear()
        index_minKx = np.argmin(abs(plot_minKx-Kx[0,:]))
        index_maxKx = np.argmin(abs(plot_maxKx-Kx[0,:]))
        index_minKy = np.argmin(abs(plot_minKy-Ky[:,0]))
        index_maxKy = np.argmin(abs(plot_maxKy-Ky[:,0]))
        for band in range(4*chain_length):
            if surface_visibility[str(band+1)]:
                ax_surface.plot_surface(Kx[index_minKy:index_maxKy, index_minKx:index_maxKx], Ky[index_minKy:index_maxKy, index_minKx:index_maxKx], results_eigvals[index_minKy:index_maxKy, index_minKx:index_maxKx, band], facecolors=color_scale.to_rgba(spin_projections[index_minKy:index_maxKy-1, index_minKx:index_maxKx-1, band]), linewidth=0.1, alpha=0.5)
        ax_surface.set_xlabel("Kx")
        ax_surface.set_ylabel("Ky")
        ax_surface.set_zlabel("Energy (eV)")
        print("Replot completed :)")
    else:
        print("Check Kx/Ky bounds")


def submit_minKx(text_input):
    global plot_minKx
    try:
        plot_minKx = float(text_input)
    except:
        print("Suck a fat one!")

def submit_maxKx(text_input):
    global plot_maxKx
    try:
        plot_maxKx = float(text_input)
    except:
        print("Suck a fat one!")

def submit_minKy(text_input):
    global plot_minKy
    try:
        plot_minKy = float(text_input)
    except:
        print("Suck a fat one!")

def submit_maxKy(text_input):
    global plot_maxKy
    try:
        plot_maxKy = float(text_input)
    except:
        print("Suck a fat one!")

def update_fermi_lvl(fermi_lvl):
    global plot_fermi_level
    plot_fermi_level = fermi_lvl

    ax_bc.clear()
    ax_bc.contourf(Kx, Ky, berry_curvature[:,:,plot_fermi_level], cmap=cmap_bc, norm=norm_bc)
    
    for band in range(4*chain_length):
        if np.max(results_eigvals[:,:,band])>fermi_energies[plot_fermi_level] and np.min(results_eigvals[:,:,band])<=fermi_energies[plot_fermi_level]:
            ax_bc.contour(Kx, Ky, results_eigvals[:,:,band], levels=[fermi_energies[plot_fermi_level]], colors="black")

    
    ax_bc.set_title(f'Berry curvature at E_F = {fermi_energies[plot_fermi_level]:.2f}eV')
    ax_bc.set_xlabel('Kx')
    ax_bc.set_ylabel('Ky')

fermi_slider.on_changed(update_fermi_lvl)
update_fermi_lvl(plot_fermi_level)
text_minKx.on_submit(submit_minKx)
text_maxKx.on_submit(submit_maxKx)
text_minKy.on_submit(submit_minKy)
text_maxKy.on_submit(submit_maxKy)
plot_button.on_clicked(plot_surface)
check.on_clicked(reset_visibility)


plt.show()
