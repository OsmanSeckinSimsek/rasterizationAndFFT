import numpy as np
import h5py
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Record the start time
start_time = time.time()

file_name = 'turb_200_correct.h5'

print("Reading")
with h5py.File(file_name, 'r') as file:
    x = file['Step#22/x'][:]
    y = file['Step#22/y'][:]
    z = file['Step#22/z'][:]
    vx = file['Step#22/vx'][:]
    vy = file['Step#22/vy'][:]
    vz = file['Step#22/vz'][:]

end_time1 = time.time()
elapsed = end_time1 - start_time
print(f"Reading done. Time: {elapsed} s")

# Create a 3D Cartesian mesh
mesh_resolution = len(x.shape[0]) * 2
mesh_x, mesh_y, mesh_z = np.meshgrid(np.linspace(-0.5, 0.5, mesh_resolution),
                                     np.linspace(-0.5, 0.5, mesh_resolution),
                                     np.linspace(-0.5, 0.5, mesh_resolution))

# Convert mesh coordinates to a flat array
target_grid_points = np.column_stack(
    (mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()))
end_time2 = time.time()
elapsed = end_time2 - end_time1
print(f"Creating Cartesian mesh. Time: {elapsed} s")

# Create KD-Tree from scattered points
scattered_points = np.column_stack((x, y, z))
kdtree = cKDTree(scattered_points)
end_time3 = time.time()
elapsed = end_time3 - end_time2
print(f"Creating KD-tree. Time: {elapsed} s")

# Query the KD-Tree for nearest neighbors
distances, indices = kdtree.query(target_grid_points, k=1)
end_time4 = time.time()
elapsed = end_time4 - end_time3
print(f"Querying KD-tree. Time: {elapsed} s")

# Interpolate values based on the nearest neighbors
interpolated_values = np.column_stack((vx[indices], vy[indices], vz[indices]))

# Reshape the interpolated values to match the shape of the target grid
interpolated_values_reshaped = interpolated_values.reshape(
    (mesh_resolution, mesh_resolution, mesh_resolution, 3))
end_time5 = time.time()
elapsed = end_time5 - end_time4
print(f"Interpolating. Time: {elapsed} s")

# Number of grid points in each dimension
grid_size = mesh_resolution

# Compute the velocity field
vx_field = interpolated_values[:, :, :, 0].reshape(
    (grid_size, grid_size, grid_size))
vy_field = interpolated_values[:, :, :, 1].reshape(
    (grid_size, grid_size, grid_size))
vz_field = interpolated_values[:, :, :, 1].reshape(
    (grid_size, grid_size, grid_size))
print("Calculating means")

# Calculate the mean along each spatial dimension
mean_vx = np.mean(vx_field, axis=(0, 1, 2))
mean_vy = np.mean(vy_field, axis=(0, 1, 2))
mean_vz = np.mean(vz_field, axis=(0, 1, 2))
print(mean_vx)
print(mean_vy)
print(mean_vz)

print("Calculating FFTs")
vx_fft = np.fft.fftn(vx_field)
vy_fft = np.fft.fftn(vy_field)
vz_fft = np.fft.fftn(vz_field)

print("Calculating Power Spectra")
power_spectrum = (np.abs(vx_fft) ** 2 + np.abs(vy_fft)
                  ** 2 + np.abs(vz_fft) ** 2)
end_time6 = time.time()
elapsed = end_time6 - end_time5
print(f"Calculating Power Spectra. Time: {elapsed} s")

# Compute the 1D wavenumber array
k_values = np.fft.fftfreq(grid_size, d=1.0 / grid_size)

k_1d = np.abs(k_values)

# Perform spherical averaging to get 1D power spectrum
power_spectrum_radial = np.zeros_like(k_1d)

for i in range(grid_size):
    for j in range(grid_size):
        for l in range(grid_size):
            k = np.sqrt(k_values[i] ** 2 + k_values[j] ** 2 + k_values[l] ** 2)
            k_index = np.argmin(np.abs(k_1d - k))
            power_spectrum_radial[k_index] += power_spectrum[i, j, l]

# Normalize the result
power_spectrum_radial /= np.sum(power_spectrum_radial)
# power_spectrum_radial *= k_1d

end_time7 = time.time()
elapsed = end_time7 - end_time6
print(f"Spherical averaging. Time: {elapsed} s")

print("Outputing...")

# Save 1D spectra and k to a file
output_file_path = 'power_spectrum_data_analytical.txt'
np.savetxt(output_file_path, np.column_stack(
    (k_1d[k_values > 0], power_spectrum_radial[k_values > 0])))

end_time8 = time.time()
elapsed = end_time8 - end_time7
print(f"Outputing. Time: {elapsed} s")

print("Plotting")

# Plot the 1D power spectrum
plt.plot(k_1d[1:], power_spectrum_radial[1:])
plt.xscale('log')
plt.yscale('log')
# Set x-axis limit to start from 6
plt.xlim(6, k_1d.max())

# Add a line for the expected Kolmogorov slope of -5/3
kolmogorov_line = k_1d[1:]**(-5.0/3.0)
plt.plot(k_1d[1:], kolmogorov_line,
         label='Kolmogorov slope (-5/3)', linestyle='--')

plt.xlabel('Wavenumber (k)')
plt.ylabel('E_k')
plt.title('Power Spectrum 200^3')

# Add legend
plt.legend()

# Save the plot as a PNG file
plt.savefig('power_spectrum.png')

print("Done!")
elapsed = end_time8 - start_time
print(f"Total elapsed time: {elapsed} s")
