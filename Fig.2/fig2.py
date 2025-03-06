import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import get_reciprocal
from matplotlib import cm  # For color map handling
from scipy.interpolate import griddata

np.set_printoptions(precision=6)

# Set up system information for data loading
system_3x3 = 'q3x3'
label = 'DFT'
lambda_q_3 = np.load(f'lambda_{system_3x3}_{label}.npy')
q_3 = np.load(f'../data_phonon/q_points_{system_3x3}.npy')

system_31x31 = 'q31x31'
lambda_q_27 = np.load(f'lambda_{system_31x31}_{label}.npy')
q_27 = np.load(f'../data_phonon/q_points_{system_31x31}.npy')

# Reciprocal lattice in Cartesian coordinates
a = 7.0017278277323989
a1 = np.array([a, 0.0, 0.0])
a2 = np.array([-a / 2.0, np.sqrt(3) * a / 2.0, 0.0])
a3 = np.array([0.0, 0.0, 23.0])
b1, b2, b3 = get_reciprocal([a1, a2, a3])

# Magnon and phonon mode indices and coupling values
i_mag = 0
i_ph = 15
c_27 = np.abs(lambda_q_27[:, i_mag, i_ph])
c_3 = np.abs(lambda_q_3[:, i_mag, i_ph])

# Convert q points to Cartesian coordinates for 31x31 grid
q_x = [q1 * b1[0] + q2 * b2[0] for q1, q2, q3 in q_27]
q_y = [q1 * b1[1] + q2 * b2[1] for q1, q2, q3 in q_27]

# Create a grid for interpolation
grid_x, grid_y = np.meshgrid(
    np.linspace(min(q_x), max(q_x), 150),   # 100 points along x
    np.linspace(min(q_y), max(q_y), 150)    # 100 points along y
)

# Interpolate c_27 values on the grid
grid_z = griddata((q_x, q_y), c_27, (grid_x, grid_y), method='cubic')

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fs = 14

# Plot the surface with interpolated data
surf = ax.plot_surface(
    grid_x, grid_y, grid_z, 
    cmap='Blues',            # Colormap for visual clarity
    edgecolor='none',        # Remove edges for smooth look
    alpha=0.7,               # Transparency for better visual clarity
    shade=True,
    rstride=1, cstride=1     # Stride options for a smooth surface
)

# Plot q points on 3x3 grid in red
ax.scatter(
    [q1 * b1[0] + q2 * b2[0] for q1, q2, q3 in q_3],
    [q1 * b1[1] + q2 * b2[1] for q1, q2, q3 in q_3],
    c_3,
    color='red',         # Set color directly for 3x3 grid points
    s=80,                # Size adjustment for smaller grid
    alpha=1.0,           # Transparency for overlap clarity
    edgecolors='k',      # Black edges for better visual separation
    linewidths=0.5
)

# Add color bar for coupling values
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('|$\lambda$| (mev)', fontsize=fs, rotation=270, labelpad=25)

# Labels and display
ax.set_xlabel('$q_x$', fontsize=fs)
ax.set_ylabel('$q_y$', fontsize=fs)
ax.tick_params(labelsize=fs-2)

plt.show()
