import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def calculate_power_spectrum_radial(k_values, k_1d, power_spectrum, grid_size):
    power_spectrum_radial = np.zeros_like(k_1d)
    count = np.zeros_like(k_1d)

    for i in range(grid_size):
        for j in range(grid_size):
            for l in range(grid_size):
                k = np.sqrt(k_values[i] ** 2 + k_values[j] ** 2 + k_values[l] ** 2)
                k_index = np.argmin(np.abs(k_1d - k))
                count[k_index] += 1.0
                power_spectrum_radial[k_index] += power_spectrum[i, j, l]
    
    return power_spectrum_radial, count


# Load the data
print("Loading...")
#data = np.loadtxt('Artificial_Velocities3.txt', delimiter=',')  # Adjust the filename
#data = np.loadtxt('interpolated_velocities_400_KD.txt')  # Adjust the filename

# Path to the HDF5 file
input_hdf5_file_path = 'interpolated_velocities.h5'
# Reading the data from HDF5 file
with h5py.File(input_hdf5_file_path, 'r') as hdf_file:
    vx = hdf_file['interpolated_vx'][:]
    vy = hdf_file['interpolated_vy'][:]
    vz = hdf_file['interpolated_vz'][:]

print("Loaded")

# Extract columns
#vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]

# Number of grid points in each dimension
grid_size = 800 #150

# Compute the velocity field
vx_field = vx.reshape((grid_size, grid_size, grid_size))
vy_field = vy.reshape((grid_size, grid_size, grid_size))
vz_field = vz.reshape((grid_size, grid_size, grid_size))
print("Calculating means")

# Calculate the mean along each spatial dimension
mean_vx = np.mean(vx_field, axis=(0, 1, 2))
mean_vy = np.mean(vy_field, axis=(0, 1, 2))
mean_vz = np.mean(vz_field, axis=(0, 1, 2))
print(mean_vx)
print(mean_vy)
print(mean_vz)

# Calculate the magnitude of the velocity field
#velocity_magnitude = np.linalg.norm([vx_field, vy_field, vz_field], axis=0)
#mean_v = np.mean(velocity_magnitude)
#print(mean_v)
# Subtract the mean from each component
#velocity_magnitude -= mean_v
# Perform 3D Fourier transform
print("Calculating FFTs")

vx_fft = np.fft.fftn(vx_field)/grid_size**3
vy_fft = np.fft.fftn(vy_field)/grid_size**3
vz_fft = np.fft.fftn(vz_field)/grid_size**3

# Perform 3D Fourier transform of the magnitude of the velocity field
#velocity_magnitude_fft = np.fft.fftn(velocity_magnitude)/grid_size**3
print("Calculating Power Spectra")

# Compute the magnitude of the Fourier transform
power_spectrum = (np.abs(vx_fft) ** 2 + np.abs(vy_fft) ** 2 + np.abs(vz_fft) ** 2)

# Compute the magnitude of the Fourier transform
#power_spectrum = np.abs(velocity_magnitude_fft) ** 2

# Compute the 1D wavenumber array
k_values = np.fft.fftfreq(grid_size, d=1.0 / grid_size)

k_1d = np.abs(k_values)
print("Spherical averaging")

# Perform spherical averaging to get 1D power spectrum
sum_PS                       = np.sum(power_spectrum)
power_spectrum_radial, count = calculate_power_spectrum_radial(k_values, k_1d, power_spectrum, grid_size)

# Normalize the result
print(sum_PS)
for i in range(len(k_1d)):
    power_spectrum_radial[i] = power_spectrum_radial[i] * 4.0 * np.pi * k_1d[i]**2 / count[i]

print(np.sum(power_spectrum_radial[count>0]))
#power_spectrum_radial /= np.sum(power_spectrum_radial)
#power_spectrum_radial *= k_1d
print("Outputing...")

# Save 1D spectra and k to a file
output_file_path = 'power_spectrum_data4.txt'
np.savetxt(output_file_path, np.column_stack((k_1d[k_values>0], power_spectrum_radial[k_values>0], count[k_values>0])))
print("Plotting")

# Plot the 1D power spectrum
plt.plot(k_1d[1:], power_spectrum_radial[1:])
plt.xscale('log')
plt.yscale('log')
# Set x-axis limit to start from 6
plt.xlim(6, k_1d.max())

# Add a line for the expected Kolmogorov slope of -5/3
kolmogorov_line = k_1d[1:]**(-5.0/3.0)
plt.plot(k_1d[1:], kolmogorov_line, label='Kolmogorov slope (-5/3)', linestyle='--')

plt.xlabel('Wavenumber (k)')
plt.ylabel('E_k')
plt.title('Power Spectrum 400^3')

# Add legend
plt.legend()

# Save the plot as a PNG file
plt.savefig('power_spectrum.png')
print("Done!")
