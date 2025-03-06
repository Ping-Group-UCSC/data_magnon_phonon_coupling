import numpy as np
import matplotlib.pyplot as plt

# This is to plot the magnon-phonon coupling constant from tilting angle.
# The tilting angle is induced from phonon displacement.
# thetas for different amplitude
thetas = np.load('thetas.npy')

# load the data
# grid 3x3: exact for Fourier transform in a 3x3 supercell
system = 'q3x3'
label = 'DFT'
phonon_folder = '../data_phonon'
magnon_folder = '../data_magnon'
# load q points
q_3 = np.load(f'{phonon_folder}/q_points_{system}.npy')
# load phonon
phonon_energies = np.load(f'{phonon_folder}/eigenvalues_{system}.npy')
# load magnon
magnon_energies = np.load(f'{magnon_folder}/eigenvalues_{system}_{label}.npy')
# select K
iq = 4
# select phonon modes from 9 to 16
mode_index = np.arange(8,16)
e_ph = phonon_energies[iq][mode_index]
# magnon modes are degenerate at K
e_mag = magnon_energies[iq][0]
print('phonon energy', e_ph)
print('magnon energy', e_mag)
# load coupling constant
lambda_q_3 = np.load(f'lambda_{system}_{label}.npy')
print('shape of lambda_q: ', lambda_q_3.shape)
# select K and phonon modes
lambda_K = lambda_q_3[iq,0,:][mode_index]
print('coupling constant', np.abs(lambda_K))

# select the last column, which A=50
theta_degrees = thetas[:,2]
print('theta degrees', theta_degrees)
theta = np.deg2rad(theta_degrees)
# unit conversion from theta to meV
AMU = 1.66e-27 # kg
h = 6.626e-34 # J.s
hbar = 1.05e-34 # J.s
e = 1.602e-19
Na = 9 * 8
A = 20 * 1e-10 * np.sqrt(AMU) / np.sqrt(Na)
S = 1.5
eta = np.sin(theta) * np.sqrt(S)
omega_ph = e_ph * 1e-3 * e / hbar # mev -> Hz
lambda_K2 = e_mag * eta / (A/2) / np.sqrt(omega_ph / hbar)
print('coupling constant', np.abs(lambda_K2))

fs = 12
# plot coulping between acoustic magnon and 9-16 phonon modes from two methods
x = np.arange(len(theta_degrees))
xticks = np.arange(9,17)

width = 0.35  # Width of the bars

plt.figure(figsize=(4,3))
plt.bar(x - width/2, np.abs(lambda_K), width=width, color='blue')
plt.bar(x + width/2, np.abs(lambda_K2), width=width, color='red')

plt.legend(fontsize=fs, facecolor='none', edgecolor='none',)
plt.xticks(x, xticks, fontsize=fs)
plt.xlabel('Phonon mode #', fontsize=fs)
plt.ylabel(r'$|\lambda|$ (meV)', fontsize=fs)
plt.show()
