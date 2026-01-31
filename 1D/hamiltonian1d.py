#Formulates hamiltonian matrix in 1D
import numpy as np

pauli_x = np.array([[0,1],
                    [1,0]], np.complex128)
pauli_y = np.array([[0,-1j],
                    [1j,0]], np.complex128)
pauli_z = np.array([[1,0],
                    [0,-1]], np.complex128)
#Default setups
chain_length = 4
spin_x = spin_y = spin_z = np.zeros((8,8), np.complex128)
for i in range(4):
    spin_x+=np.pad(pauli_x, (2*i, 2*(3-i)))
    spin_y+=np.pad(pauli_y, (2*i, 2*(3-i)))
    spin_z+=np.pad(pauli_z, (2*i, 2*(3-i)))

hop_back = np.eye(8, k=2)+np.eye(8, k=-6)
hop_fwd = np.eye(8, k=-2)+np.eye(8, k=6)

def set_chainlength(new_length):
    '''
    Sets the size of the supercell to new_length, updates dimensions of spin operators and future hamiltonian matrices

    Args:
        new_length (int) : New supercell length
    '''
    global spin_x, spin_y, spin_z, hop_back, hop_fwd, chain_length

    chain_length = new_length
    hop_back = np.eye(2*chain_length, k=2)+np.eye(2*chain_length, k=-2*chain_length+2)
    hop_fwd = np.eye(2*chain_length, k=2)+np.eye(2*chain_length, k=-2*chain_length+2)

    spin_x = spin_y = spin_z = np.zeros((8,8), np.complex128)
    for i in range(chain_length):
        spin_x+=np.pad(pauli_x, (2*i, 2*(chain_length-1-i)))
        spin_y+=np.pad(pauli_y, (2*i, 2*(chain_length-1-i)))
        spin_z+=np.pad(pauli_z, (2*i, 2*(chain_length-1-i)))

def hamiltonian(k_x, exchange, hopping, chirality, moment_mag, angle, rashba_const, lattice_const=1, rsoc_type="linear"):
    '''
    Calculates hamiltonian matrix according to 

    Args:
        k_x (float) : Wavenumber, in angstrom^-1
        exchange (float) : Exchange coupling constant (lambda), in eV
        hopping (float) : Hopping amplitude (t), 0 <= t <= 1 eV
        chirality (int) : Chirality (gamma), gamma = +/- 1
        moment_mag (float) : Moment magnitude (M)
        angle (float) : tilt angle (alpha), in radians
        rashba_const (float) : Rashba coupling constant (lambda_R), in eV angstrom
        lattice_const (float) : Lattice constant/spacing of spins (a), in angstroms
        rsoc_type (str) : Type of Rashba spin-orbit couping (linear, rotating or periodic)

    Returns:
        hamiltonian_matrix (np.ndarray, dtype=np.complex128)
    '''

    hamiltonian_matrix = hopping*np.exp(+1j*k_x*lattice_const)*hop_fwd + hopping*np.exp(-1j*k_x*lattice_const)*hop_back

    for i in range(chain_length):
        #Helical magnetisation, Exchange coupling
        exchange_x = moment_mag*exchange*np.sin(angle)*pauli_x
        exchange_y = moment_mag*exchange*chirality*np.sin(i*2*np.pi/chain_length)*np.cos(angle)*pauli_y
        exchange_z = moment_mag*exchange*np.cos(i*2*np.pi/chain_length)*np.cos(angle)*pauli_z

        #Rashba spin-orbit coupling
        rashba_soc = np.zeros((2,2), dtype=np.complex128)
        if rsoc_type=="linear":
            rashba_soc += -rashba_const*k_x*pauli_z
        elif rsoc_type=="rotating":
            rashba_soc += rashba_const*k_x*(chirality*np.cos(i*np.pi/2)*pauli_z + np.sin(i*np.pi/2)*pauli_y)
        elif rsoc_type=="periodic":
            rashba_soc += -rashba_const*np.sin(2*k_x*lattice_const)*pauli_z

        hamiltonian_matrix+=np.pad(exchange_x+exchange_y+exchange_z+rashba_soc, (2*i, 2*(chain_length-1-i)))

    return hamiltonian_matrix