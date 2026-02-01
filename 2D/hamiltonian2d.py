#Copy of own script: helimagnet_hamiltonian.py
#General library for Hamiltonian matrix, differential Hamiltonian matrix, spin projection operation

import numpy as np

chain_length = 4
pauli_x = np.array([[0, 1],
                    [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -1j],
                    [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0],
                    [0, -1]], dtype=np.complex128)

spin_x = spin_y = spin_z = hop_fwd = hop_back = np.zeros((16, 16), dtype=np.complex128)

def set_chainlength(new_length):
    global chain_length, spin_x, spin_y, spin_z, hop_fwd, hop_back
    
    chain_length=new_length
    spin_x = spin_y = spin_z = np.zeros((4*n, 4*n), dtype=np.complex128)
    
    for i in range(2*n):
        spin_x+=np.pad(pauli_x, (2*i, 2*(2*n-1-i)))
        spin_y+=np.pad(pauli_y, (2*i, 2*(2*n-1-i)))
        spin_z+=np.pad(pauli_z, (2*i, 2*(2*n-1-i)))
        
    hop_back = np.eye(2*chain_length, k=2)+np.eye(2*chain_length, k=-2*(chain_length-1))
    hop_fwd = np.eye(2*chain_length, k=-2)+np.eye(2*chain_length, k=2*(chain_length-1))

set_chainlength(chain_length)

def hamiltonian(k_x, k_y, exchange, hopping_x, hopping_y, chirality, moment_mag, angle, rashba_const, rsoc_type='linear', lattice_const_x=1, lattice_const_y=1, interchain_coupling=0):
    """ Calculate Hamiltonian matrix given above parameters
    Args:
        k_x, k_y (float) = wavevectors/angstrom^-1
        hopping_x, hopping_y (float) = hopping amplitudes/between -1 and 1
        chirality (int) = chirality/either +1 or -1
        moment_mag (float)
        rashba_const (float) = magnitude of RSOC/eV angstrom
        rashba_type (str) = type of RSOC/ 'constant', 'linear' or 'rotating'
        lattice_const_x, lattice_const_y (float) = lattice constant/angstrom
        interchain_coupling (int) = interchain coupling/-1 (AFM), 0 (none), +1(FM)
    Returns:
        np.ndarray of shape (n, n) and dtype np.complex128
    """
    #Hopping
    longitudinal = hopping_x*np.exp(+1j*k_x*lattice_const_x)*hop_fwd + hopping_x*np.exp(-1j*k_x*lattice_const_x)*hop_back
    transverse_xy = hopping_y*np.exp(-1j*k_y*lattice_const_y)*np.eye(2*chain_length)
    transverse_yx = hopping_y*np.exp(+1j*k_y*lattice_const_y)*np.eye(2*chain_length)

    for i in range(chain_length):
        #Exchange interaction
        exch = moment_mag*exchange*(np.sin(angle)*pauli_x + chirality*np.sin(2*i*np.pi/chain_length)*np.cos(angle)*pauli_y + np.cos(2*i*np.pi/chain_length)*np.cos(angle)*pauli_z)
        longitudinal+=np.pad(exch, (2*i, 2*(chain_length-1-i)))

        #Interchain exchange
        transverse_xy+=np.pad(interchain_coupling*exch, (2*i, 2*(chain_length-1-i)))
        transverse_yx+=np.pad(interchain_coupling*exch, (2*i, 2*(chain_length-1-i)))

        #Rashba SOC
        if rsoc_type=='constant':
            longitudinal+=np.pad(-rashba_const*pauli_z, (2*i, 2*(chain_length-1-i)))
        elif rsoc_type=='linear':
            longitudinal+=np.pad(rashba_const*(k_x*pauli_y - k_y*pauli_x), (2*i, 2*(chain_length-1-i)))
        elif rsoc_type=='periodic':
            longitudinal+=np.pad(rashba_const*np.sin(2*k_x*lattice_const_x)*pauli_y-rashba_const*np.sin(k_y*lattice_const_y/2)*pauli_x, (2*i, 2*(chain_length-1-i)))
        elif rsoc_type=='rotating':
            longitudinal+=np.pad(rashba_const*(-k_y*np.sin(2*i*np.pi/chain_length)*pauli_x + k_x*np.sin(2*i*np.pi/chain_length)*pauli_y + chirality*k_x*np.cos(2*i*np.pi/chain_length)*pauli_z), (2*i, 2*(chain_length-1-i)))

    return np.vstack(( np.hstack((longitudinal, transverse_xy)), np.hstack((transverse_yx, longitudinal)) ))