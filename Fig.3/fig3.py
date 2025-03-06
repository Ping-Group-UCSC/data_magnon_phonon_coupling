import numpy as np
import matplotlib.pyplot as plt
from helper import get_reciprocal

np.set_printoptions(precision=6)

# Common parameters
system = 'q31x31'
phonon_folder = '../data_phonon'
a = 7.0017278277323989
a1 = np.array([a, 0.0, 0.0])
a2 = np.array([-a / 2.0, np.sqrt(3) * a / 2.0, 0.0])
a3 = np.array([0.0, 0.0, 23.0])
b1, b2, b3 = get_reciprocal([a1, a2, a3])

# magnon and phonon mode
i_mag = 0
i_ph = 15

# Load q points
q_27 = np.load(f'{phonon_folder}/q_points_{system}.npy')

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
fs = 16

# Plot 1: DFT_LDA
label1 = 'DFT_LDA'
lambda_q_27_lda = np.load(f'./lambda_{system}_{label1}.npy')
cvalues_lda = np.abs(lambda_q_27_lda[:, i_mag, i_ph])

for iq in range(q_27.shape[0]):
    q1, q2, q3 = q_27[iq]
    q = q1 * b1 + q2 * b2 + q3 * b3
    scatter1 = axs[0].scatter(q[0], q[1], c=cvalues_lda[iq], cmap='Blues', s=125, 
                               alpha=1.0, edgecolors='none', vmin=0.0, vmax=max(cvalues_lda))

axs[0].set_title('LDA', fontsize=fs)
axs[0].set_aspect('equal')  # Set equal aspect ratio
cbar1 = fig.colorbar(scatter1, ax=axs[0], shrink=0.7)
cbar1.ax.tick_params(labelsize=fs-2)
#cbar1.set_label('|$\lambda$| (mev)', rotation=270, labelpad=25, fontsize=fs)
axs[0].text(0.02, 0.97, '(a)', transform=axs[0].transAxes, fontsize=fs+2, va='top', ha='left')
axs[0].set_xlabel('$q_x$', fontsize=fs)
axs[0].set_ylabel('$q_y$', fontsize=fs)
axs[0].tick_params(labelsize=fs)

# Plot 2: DFT
label2 = 'DFT'
lambda_q_27_dft = np.load(f'./lambda_{system}_{label2}.npy')
cvalues_dft = np.abs(lambda_q_27_dft[:, i_mag, i_ph])

for iq in range(q_27.shape[0]):
    q1, q2, q3 = q_27[iq]
    q = q1 * b1 + q2 * b2 + q3 * b3
    scatter2 = axs[1].scatter(q[0], q[1], c=cvalues_dft[iq], cmap='Blues', s=125, 
                               alpha=1.0, edgecolors='none', vmin=0.0, vmax=max(cvalues_dft))

axs[1].set_title('PBE', fontsize=fs)
axs[1].set_aspect('equal')  # Set equal aspect ratio
cbar2 = fig.colorbar(scatter2, ax=axs[1], shrink=0.7)
cbar2.ax.tick_params(labelsize=fs-2)
#cbar2.set_label('|$\lambda$| (mev)', rotation=270, labelpad=25, fontsize=fs)
axs[1].text(0.02, 0.97, '(b)', transform=axs[1].transAxes, fontsize=fs+2, va='top', ha='left')
axs[1].set_xlabel('$q_x$', fontsize=fs)
axs[1].set_ylabel('$q_y$', fontsize=fs)
axs[1].tick_params(labelsize=fs)

# Show plot
plt.show()

