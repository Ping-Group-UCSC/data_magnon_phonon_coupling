import numpy as np
import matplotlib.pyplot as plt
from helper import model

# CrTe2
a1 = np.array([1.0, 0.0, 0.0])
a2 = np.array([-1/2.0, np.sqrt(3)/2.0, 0.0])
a3 = np.array([0.0, 0.0, 1.0])

S = 3/2
J_CrTe2 = np.array([5.4, 3.4, 1.7, 2.5])
J_CrTe2_scaled = J_CrTe2 * 2 / S**2
z1, z2, z3 = (6, 6, 6)

cri3_CrTe2 = model([a1, a2, a3], J_CrTe2_scaled, [z1, z2, z3], S)
b1, b2, b3 = cri3_CrTe2.get_reciprocal()

G = [0.0, 0.0, 0.0]
M = [0.5, 0.0, 0.0]
kpoints_CrTe2 = np.array([G, M])

def H_k_CrTe2(k, J_list, z_list, S):
    k1, k2 = k[0], k[1]
    k_C = k1 * b1 + k2 * b2
    kx, ky = k_C[0], k_C[1]
    J1, J2, J3, Az = J_list
    z1, z2, z3 = z_list
    Ak = (Az + z1 * J1 + z2 * J2 + z3 * J3 
          - J1 * 2 * (np.cos(kx) + 2 * np.cos(kx/2) * np.cos(np.sqrt(3)*ky/2))
          - J2 * 2 * (np.cos(np.sqrt(3)*ky) + 2 * np.cos(3*kx/2) * np.cos(np.sqrt(3)*ky/2))
          - J3 * 2 * (np.cos(2*kx) + 2 * np.cos(kx) * np.cos(np.sqrt(3)*ky)))
    return S * Ak

X_CrTe2, E_CrTe2, d_CrTe2 = cri3_CrTe2.band(kpoints_CrTe2, H_k_CrTe2, Nb=1, Nk=51)
xx_CrTe2 = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * d_CrTe2[-1]
e_CrTe2 = np.array([8.3, 13.4, 23.6, 44.4, 65.0, 80.4, 91, 103, 103, 97, 78])

# CrI3
J_CrI3 = np.array([3.26, 0.71, -0.18, 1.97/2])
J_CrI3_scaled = J_CrI3 * 2 / S**2
J_CrI3_LDA = np.array([2.16, 0.62, -0.25, 0.38*2])
J_CrI3_exp = np.array([2.01, 0.16, -0.08, 0.49*2])
J_CrI3_HSE = np.array([3.59, 0.41, -0.04, 0.49*2])
z1, z2, z3 = (3, 6, 3)


cri3_DFT_CrI3 = model([a1, a2, a3], J_CrI3_scaled, [z1, z2, z3], S)
cri3_exp_CrI3 = model([a1, a2, a3], J_CrI3_exp, [z1, z2, z3], S)
cri3_DFT_LDA = model([a1, a2, a3], J_CrI3_LDA, [z1, z2, z3], S)
cri3_DFT_HSE = model([a1, a2, a3], J_CrI3_HSE, [z1, z2, z3], S)
b1, b2, b3 = cri3_DFT_CrI3.get_reciprocal()
G = 0.0 * b1 + 0.0 * b2
K = 0.333333 * b1 + 0.333333 * b2
M = 0.5 * b1 + 0.0 * b2
kpoints_CrI3 = np.array([G, K, M, G])

def H_k_CrI3(k, J_list, z_list, S):
    kx, ky = k[0], k[1]
    nb = 2
    H_k = np.zeros((nb, nb), dtype=complex)
    J1, J2, J3, A = J_list
    z1, z2, z3 = z_list
    Ak = A + z1 * J1 + z2 * J2 + z3 * J3 - 2 * J2 * (np.cos(kx) + 2*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2))
    Bk = -J1 * (np.exp(1j*np.sqrt(3)*ky/3) + 2*np.cos(kx/2)*np.exp(-1j*np.sqrt(3)*ky/6)) \
         -J3 * (np.exp(-1j*2*np.sqrt(3)*ky/3) + 2*np.cos(kx)*np.exp(1j*np.sqrt(3)*ky/3)) 
    H_k[0,0] = Ak
    H_k[0,1] = Bk
    H_k[1,0] = np.conjugate(Bk)
    H_k[1,1] = Ak
    eigenvalues, _ = np.linalg.eig(H_k)
    sorted_eigenvalues = np.sort(eigenvalues)
    return S * sorted_eigenvalues

X_CrI3_DFT, E_CrI3_DFT, d_CrI3 = cri3_DFT_CrI3.band(kpoints_CrI3, H_k_CrI3, Nb=2, Nk=51)
X_CrI3_exp, E_CrI3_exp, _ = cri3_exp_CrI3.band(kpoints_CrI3, H_k_CrI3, Nb=2, Nk=51)
X_CrI3_LDA, E_CrI3_LDA, _ = cri3_DFT_LDA.band(kpoints_CrI3, H_k_CrI3, Nb=2, Nk=51)
X_CrI3_HSE, E_CrI3_HSE, _ = cri3_DFT_HSE.band(kpoints_CrI3, H_k_CrI3, Nb=2, Nk=51)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fs=18

# CrTe2 Plot (a)
ax1.plot(X_CrTe2, E_CrTe2, lw=1, color='red', label='LSWT')
ax1.scatter(xx_CrTe2, e_CrTe2, label='TDDFT')
ax1.set_xlim(d_CrTe2[0], d_CrTe2[-1])
ax1.set_ylim(0,)
ax1.set_xticks(d_CrTe2, [r'$\Gamma$', 'M'])
ax1.set_ylabel('Energy (meV)', fontsize=fs)
ax1.legend(facecolor='none', edgecolor='none', fontsize=fs)
ax1.tick_params(labelsize=fs-2)
ax1.text(-0.25, 0.97, '(a)', transform=ax1.transAxes, fontsize=fs+2, va='top', ha='left')


# for the extracted data, experimental data
print('GK', d_CrI3[1]-d_CrI3[0])
print('KM', d_CrI3[2]-d_CrI3[1])
print('MG', d_CrI3[3]-d_CrI3[2])
G = -2
K = -1.335
M = -1.0
X_exp = (np.loadtxt('./G-K.dat', delimiter=',')[:, 0] - G) * (d_CrI3[1] - d_CrI3[0]) / (K-G)
E_exp = np.loadtxt('./G-K.dat', delimiter=',')[:, 1]
# G-K
ax2.scatter(X_exp, E_exp, color='black', s=20)
# K-M
X_exp = d_CrI3[1] + (np.loadtxt('./K-M.dat', delimiter=',')[:, 0] - K) * (d_CrI3[2] - d_CrI3[1]) / (M-K)
E_exp = np.loadtxt('./K-M.dat', delimiter=',')[:, 1]
ax2.scatter(X_exp, E_exp, color='black', s=20)
# M-G
M = 0.5
G = 1.0
X_exp = d_CrI3[2] + (np.loadtxt('./M-G.dat', delimiter=',')[:, 0] - M) * (d_CrI3[3] - d_CrI3[2]) / (G-M)
E_exp = np.loadtxt('./M-G.dat', delimiter=',')[:, 1]
ax2.scatter(X_exp, E_exp, label='Exp', color='black', s=20)

# CrI3 Plot (b)
ax2.plot(X_CrI3_DFT, E_CrI3_DFT[:,0], lw=1, color='red', label='PBE')
ax2.plot(X_CrI3_LDA, E_CrI3_LDA[:,0], lw=1, color='blue', label='LDA')
ax2.plot(X_CrI3_HSE, E_CrI3_HSE[:,0], lw=1, color='green', label='HSE')
ax2.plot(X_CrI3_exp, E_CrI3_exp[:,0], lw=1, color='black')
ax2.plot(X_CrI3_DFT, E_CrI3_DFT[:,1], lw=1, color='red')
ax2.plot(X_CrI3_exp, E_CrI3_exp[:,1], lw=1, color='black')
ax2.plot(X_CrI3_LDA, E_CrI3_LDA[:,1], lw=1, color='blue')
ax2.plot(X_CrI3_HSE, E_CrI3_HSE[:,1], lw=1, color='green')
ax2.set_xlim(d_CrI3[0], d_CrI3[-1])
ax2.set_ylim(0,)
ax2.axvline(d_CrI3[1], c='black', ls='--', lw=1/3)
ax2.axvline(d_CrI3[2], c='black', ls='--', lw=1/3)
ax2.set_xticks(d_CrI3, [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
ax2.set_ylabel('Energy (meV)', fontsize=fs)
ax2.legend(facecolor='none', edgecolor='none', bbox_to_anchor=(0.5, 0.2), fontsize=fs-6, ncol=2)
ax2.tick_params(labelsize=fs-2)
ax2.text(-0.25, 0.97, '(b)', transform=ax2.transAxes, fontsize=fs+2, va='top', ha='left')

plt.tight_layout()
plt.show()
