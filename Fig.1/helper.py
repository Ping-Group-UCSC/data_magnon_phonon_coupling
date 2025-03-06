import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class model:
    def __init__(self, lattice, J_list, z_list, S) -> None:
        """
        Initializes the model with lattice vectors, J1, z1, and S parameters.
        Args:
            lattice (list of numpy arrays): Lattice vectors.
            J1_list (list of floats): List of J1 parameters.
            z1_list (list of floats): List of z1 parameters.
            S (float): Parameter S.
        """
        self.S = S
        self.a1, self.a2, self.a3 = lattice
        self.b1, self.b2, self.b3 = self.get_reciprocal()
        #if len(J_list) != len(z_list):
        #    raise ValueError('J_list and z_list must have the same length.')
        self.J_list = J_list
        self.z_list = z_list
        self.num_neighbors = len(J_list)

    def get_reciprocal(self):
        """
        Calculates the reciprocal lattice vectors.
        
        Returns:
            tuple of numpy arrays: Reciprocal lattice vectors (b1, b2, b3).
        """
        V = np.dot(self.a1, np.cross(self.a2, self.a3))
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / V
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / V
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / V
        return b1, b2, b3

    def _k_C(self, k):
        """
        convert k point in direct to Cartesian
        """
        b1, b2, b3 = self.get_reciprocal()
        k1, k2, k3 = k[0], k[1], k[2]
        k_C = k1 * b1 + k2 * b2 + k3 * b3
        return k_C

    def _band_helper(self, H_k, k1, k2, Nb, Nk):
        """
        Computes the energy bands for a segment between two k-points.

        Args:
            H_k (function): Hamiltonian function taking k and parameters J1, z1, S.
            k1 (numpy array): Start k-point.
            k2 (numpy array): End k-point.
            Nb (int): Number of bands.
            Nk (int): Number of k-points.

        Returns:
            tuple: X (distance along the segment), E (energy bands), d (distance between k1 and k2).
        """
        E = np.zeros((Nk, Nb))
        for nk in range(Nk):
            k = (k2 - k1) * nk / (Nk - 1) + k1
            E[nk, :] = H_k(k, self.J_list, self.z_list, self.S)
        # distance between k1 and k2
        dist = np.linalg.norm(k2 - k1)
        X = np.linspace(0, dist, Nk)
        return X, E, dist

    def band(self, kpoints, H_k, Nb=1, Nk=51):
        """
        Computes the energy bands along specified k-points using a provided Hamiltonian H_k.

        Args:
            kpoints (numpy array): Array of high-symmetry points. Each row has the coordinates in Cartesian.
            H_k (function): Hamiltonian function taking k and parameters J1, z1, S.
            Nb (int): Number of bands.
            Nk (int): Number of k-points between two high-symmetry points.

        Returns:
            tuple: X (distance along the path), E (energy bands), d (distances between k-points).
        """
        # start band calculation
        # number of high-symmetry points
        n = kpoints.shape[0]
        X = np.zeros(Nk * (n - 1))
        E = np.zeros((Nk * (n - 1), Nb))
        d = np.zeros(n)
        d[0] = 0
        # iterate each segment
        for i in range(n - 1):
            k1 = kpoints[i, :]
            k2 = kpoints[i + 1, :]
            # get the bands and distance between k1 and k2
            X1, E1, d1 = self._band_helper(H_k, k1, k2, Nb, Nk)
            start = i * Nk
            end = (i + 1) * Nk
            # this accumulate the distance to be plotted later            
            X[start:end] = X1 + d[i]
            E[start:end, :] = E1
            d[i + 1] = d[i] + d1
        return X, E, d


def phonon_interpolate(k, ib):
    """
    get the phonon energy at any k interpolated from DFT calculations
    Input:
        k: a list of (kx, ky)
        ib: band index
    Required files (run read_phonopy.py first): 
        q_points.npy
        bands.npy
    """
    # DFT calculation
    q_points = np.load('q_points.npy')
    bands = np.load('bands.npy')
    # select the band
    band = bands[:, ib]
    kx = q_points[:, 0]
    ky = q_points[:, 1]
    # interpolate
    e = griddata((kx, ky), band, (k[0], k[1]), method='nearest')
    return e


def plot_phonon(file_name, nq, nband):
    """"
    plot the phonon band from DFT calculations with phonopy
    Input:
    file_name: file path of band.dat
    nq: number of q points
    nband: number of bands
    """
    # unit from THz to meV
    unit = 4.136
    kpath = np.loadtxt(file_name, comments='#')[:, 0].reshape((nband, nq))[0, :]
    band = np.loadtxt(file_name, comments='#')[:, 1].reshape((nband, nq))
    xticks = [0.00000000, 0.09521360, 0.14282050, 0.22527870]
    xlabels = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']
    plt.figure()
    for i in range(nband):
        plt.plot(kpath, band[i, :] * unit, c='blue')
    ymax = 32
    plt.xlim(xticks[0], xticks[-1])
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.axvline(xticks[1], lw=0.5, c='black')
    plt.axvline(xticks[2], lw=0.5, c='black')
    plt.ylim(0, ymax)
    plt.ylabel('Energy (mev)')


def get_derivative(force_file, num_atom):
    """
    The force file contains the force on each atom in (Fx, Fy, Fz)
    There are 4 spin configurations.
    Line number = num_atom * 4
    """
    data = np.loadtxt(force_file)
    data_parts = np.split(data, 4)
    data0, data1, data2, data3 = data_parts
    S = 1.5

    def get_x(atom, comp):
        f0 = data0[atom, comp]
        f1 = data1[atom, comp]
        f2 = data2[atom, comp]
        f3 = data3[atom, comp]
        return (f0 - f1 - f2 + f3) * 1000 / 4 / S ** 2

    derivatives = []
    for i in range(num_atom):
        x = get_x(i, 0)
        y = get_x(i, 1)
        z = get_x(i, 2)
        derivatives.append([x, y, z])
    derivatives = np.array(derivatives)
    return derivatives


def find_shortest(vector, Lvectors, thres):
    """
    For a vector R, try to find RL in Lvectors so that
    R + RL is the shortest.
    Store the multiplicity.
    Eq.(47) in https://iopscience.iop.org/article/10.1088/1361-648X/acd831
    """
    distances = []
    shortest_distance = float('inf')
    # Calculate all distances and find the shortest one
    for RL in Lvectors:
        R_plus_RL = vector + RL
        distance = np.linalg.norm(R_plus_RL)
        distances.append((R_plus_RL, distance))
        if distance < shortest_distance:
            shortest_distance = distance

    # There could be multiple RL, search again
    closest_vectors = []
    for vec, dist in distances:
        if abs(dist - shortest_distance) <= thres:
            closest_vectors.append(vec)

    return closest_vectors

def get_reciprocal(lattice):
    """
    Calculates the reciprocal lattice vectors.
    Input: lattice a list of (a1, a2, a3)
    Returns:
        tuple of numpy arrays: Reciprocal lattice vectors (b1, b2, b3).
    """
    a1, a2, a3 = lattice[0], lattice[1], lattice[2]
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / V
    b2 = 2 * np.pi * np.cross(a3, a1) / V
    b3 = 2 * np.pi * np.cross(a1, a2) / V
    return b1, b2, b3

def delta(E, width):
    """
    Use a Gaussian function to model the delta function
    E and width is in meV
    """
    f = 1 / np.sqrt(2 * np.pi * width**2) * np.exp(-E**2 / (2 * width**2))

    return f

def occupation(E, T):
    """
    Bose-Einstein distribution
    E: meV
    T: K
    """
    kB = 8.6173324 * 10**(-2) # meV/K
    kBT = kB * T
    return 1.0 / (np.exp(E/kBT) - 1)

def read_poscar(file_name):
    """Read POSCAR file and return coordinates and number of elements."""
    with open(file_name) as file_input:
        data = file_input.readlines()
    
    va = np.fromstring(data[2], sep=' ')
    vb = np.fromstring(data[3], sep=' ')
    vc = np.fromstring(data[4], sep=' ')

    # Store the type and number of each element
    elements_label = data[5].split()
    num_elements = np.fromstring(data[6], dtype=int, sep=' ')
    total_num = np.sum(num_elements)
    
    coordinates = []
    for i in range(total_num):
        c = np.fromstring(data[8 + i], dtype=float, sep=' ')
        coordinates.append(c)
    
    coordinates = np.array(coordinates)
    
    return va, vb, vc, coordinates