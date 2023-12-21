import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifftn

# Parameters
grid_size = 200  # Size of the 3D grid
epsilon = 1.0  # Turbulence dissipation rate
C_K = 1.0  # Kolmogorov constant (typical value)

# Set seed for reproducibility (optional)
np.random.seed(42)

# Generate random complex field in Fourier space
random_field_fourier = np.random.randn(grid_size, grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size, grid_size)

# Compute the magnitude of the wavenumber vector
k_values = np.fft.fftfreq(grid_size)
k_mag = np.sqrt(np.sum(np.meshgrid(k_values, k_values, k_values), axis=0)**2)

# Compute the square root of the Kolmogorov energy spectrum
kolmogorov_sqrt_spectrum = np.zeros_like(k_mag)
nonzero_indices = k_mag != 0
kolmogorov_sqrt_spectrum[nonzero_indices] = np.sqrt(C_K * epsilon**(2.0/3.0) * k_mag[nonzero_indices]**(-5.0/3.0))

# Apply the Kolmogorov spectrum to the random field in Fourier space
velocity_field_fourier = kolmogorov_sqrt_spectrum #* random_field_fourier

# Perform the inverse Fourier transform to obtain the velocity field
velocity_field = np.real(ifftn(velocity_field_fourier))

# Create 1D arrays for x, y, z
x_values = np.linspace(0, 1, grid_size, endpoint=False)
y_values = np.linspace(0, 1, grid_size, endpoint=False)
z_values = np.linspace(0, 1, grid_size, endpoint=False)

# Create a grid of coordinates
x, y, z = np.meshgrid(x_values, y_values, z_values, indexing='ij')

# Reshape arrays for output
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()
vx_flat = velocity_field[:, :, :].flatten()
vy_flat = velocity_field[:, :, :].flatten()
vz_flat = velocity_field[:, :, :].flatten()

# Output to a file
output_data = np.column_stack((x_flat, y_flat, z_flat, vx_flat, vy_flat, vz_flat))
output_file_path = 'analytical_velocity_field.txt'
np.savetxt(output_file_path, output_data)



# Plot a slice of the velocity field
slice_index = grid_size // 2
plt.imshow(velocity_field[:, :, slice_index], cmap='viridis')
plt.colorbar()
plt.title('Analytical Velocity Field Slice')
# Save the plot as a PNG file
plt.savefig('analytical_velocity_field.png')

# Close the plot
plt.close()
