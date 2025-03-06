import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

default_font = plt.rcParams['font.family']
print("Default font family:", default_font)

# grid 3x3: exact for Fourier transform in a 3x3 supercell
system = 'q100x1'
label = 'DFT'
lambda_q_3 = np.load(f'lambda_{system}_{label}.npy')
print('shape of lambda_q: ', lambda_q_3.shape)
phonon_folder = '../data_phonon'
q_3 = np.load(f'{phonon_folder}/q_points_{system}.npy')
print('shape of q_points: ', q_3.shape)

# plotting
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
iq = 1  # plot the Gamma point
print(q_3[iq])

# Set bar width and offset
fs = 15
bar_width = 0.35
x = np.arange(len(lambda_q_3[iq, 0, :])) + 1  # x positions for bars

# Plot the bars with offset to prevent overlap
axs.bar(x - bar_width/2, np.abs(lambda_q_3[iq, 0, :]), width=bar_width, label='Acoustic magnon')
axs.bar(x + bar_width/2, np.abs(lambda_q_3[iq, 1, :]), width=bar_width, label='Optical magnon')

# Set xticks and yticks
xticks = np.arange(1, 25, 1)
yticks = [float(f'{y:.2f}') for y in np.arange(0.00, 0.60, 0.10)]

axs.set_xticks(xticks)
axs.set_xlim(0.5, 24.5)
axs.set_yticks(yticks)
axs.set_yticklabels(yticks)
axs.set_xlabel('Phonon mode #', fontsize=fs)
axs.set_ylabel(r'$|\lambda|$ (meV)', fontsize=fs)
axs.tick_params(labelsize=fs-2)

# Add legend and save
axs.legend(frameon=False, fontsize=fs)
plt.show()
