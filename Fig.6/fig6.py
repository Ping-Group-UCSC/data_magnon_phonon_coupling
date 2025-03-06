import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as patches

# Helper function to load data
def load_data(kpath, label, magnon_folder, phonon_folder, lambda_folder):
    phonon_energies = np.load(f'{phonon_folder}/eigenvalues_{kpath}.npy')
    magnon_energies = np.load(f'{magnon_folder}/eigenvalues_{kpath}_{label}.npy')
    lambda_q = np.load(f'{lambda_folder}/lambda_{kpath}_{label}.npy')
    q_points = np.load(f'{phonon_folder}/q_points_{kpath}.npy')
    return phonon_energies, magnon_energies, lambda_q, q_points

# Helper function to compute eigenvalues and weights
def compute_eigenvalues_weights(phonon_energies, magnon_energies, lambda_q):
    nq, nb_ph = phonon_energies.shape
    nb_mag = magnon_energies.shape[1]
    nb = nb_ph + nb_mag
    eigenvalues = np.zeros((nq, nb))
    magnon_weight = np.zeros((nq, nb))
    
    for iq in range(nq):
        H_k = np.zeros((nb, nb), dtype=complex)
        H_k[:nb_mag, :nb_mag] = np.diag(magnon_energies[iq])
        H_k[nb_mag:, nb_mag:] = np.diag(phonon_energies[iq])
        
        # Coupling terms
        for i in range(nb_mag):
            for j in range(nb_ph):
                H_k[i, j + nb_mag] = lambda_q[iq, i, j]
                H_k[j + nb_mag, i] = lambda_q[iq, i, j].conj()
                
        eig, eigv = np.linalg.eig(H_k)
        eigenvalues[iq] = np.real(eig)
        magnon_weight[iq] = np.linalg.norm(eigv[:nb_mag], axis=0)
    
    return eigenvalues, magnon_weight

# Helper function for plotting
def plot_band_structure(ax, xx, eigenvalues, magnon_weight, xticks, xticklabels, cmap, norm, ylabel, ylim):
    for ib in range(eigenvalues.shape[1]):
        ax.scatter(xx, eigenvalues[:, ib], s=0.5, c=magnon_weight[:, ib], cmap=cmap, norm=norm)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    for xtick in xticks:
        ax.axvline(x=xtick, color='black', linewidth=0.5)

# Define colormap and normalization
colors = [(0, 0, 1), (1, 0, 0)]  # Blue to Red
cmap = LinearSegmentedColormap.from_list('blue_red', colors, N=100)
norm = Normalize(vmin=0, vmax=1)

# Plotting setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Data for CrI3
kpath, label = 'GKM', 'exp'
phonon_energies, magnon_energies, lambda_q, _ = load_data(
    kpath, label, 'CrI3', 'CrI3', 'CrI3'
)
eigenvalues, magnon_weight = compute_eigenvalues_weights(phonon_energies, magnon_energies, lambda_q)
xx = np.arange(eigenvalues.shape[0])
index_K = 501
plot_band_structure(ax1, xx, eigenvalues, magnon_weight, [0, index_K, 2*index_K], [r'$\Gamma$', 'K', 'M'], cmap, norm, 'Energy (meV)', (8, 16))
ax1.text(-0.15, 0.97, '(a)', transform=ax1.transAxes, fontsize=14, va='top', ha='left')
ax1.set_xlim(0, index_K*2)
width = 60
height = 2.1
rect = patches.Rectangle((index_K-width/2, 11.7), width, height, linewidth=0.5, edgecolor='green', facecolor='none')
ax1.add_patch(rect)

# Data for CrTe2
kpath, label = 'GKMG', 'DFT_3.91'
phonon_energies, magnon_energies, lambda_q, _ = load_data(
    kpath, label, 'CrTe2', 'CrTe2', 'CrTe2'
)
eigenvalues, magnon_weight = compute_eigenvalues_weights(phonon_energies, magnon_energies, lambda_q)
xx = np.arange(eigenvalues.shape[0])
index_K = 1001
plot_band_structure(ax2, xx, eigenvalues, magnon_weight, [0, index_K, 2*index_K, 3*index_K], [r'$\Gamma$', 'K', 'M', r'$\Gamma$'], cmap, norm, 'Energy (meV)', (10, 40))
ax2.text(-0.20, 0.97, '(b)', transform=ax2.transAxes, fontsize=14, va='top', ha='left')
ax2.set_xlim(0, index_K*3)

# Add colorbar
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax2, orientation="vertical")

plt.tight_layout()
plt.show()
