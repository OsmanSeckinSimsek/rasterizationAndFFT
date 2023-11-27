import numpy as np
import matplotlib.pyplot as plt

# Load data from file1
file1_data = np.loadtxt('power_spectrum_data_100.txt')
k_1d_file1, power_spectrum_file1 = file1_data[:-2, 0], file1_data[:-2, 1]

# Load data from file2
file2_data = np.loadtxt('power_spectrum_data_analytical.txt')
k_1d_file2, power_spectrum_file2 = file2_data[:-2, 0], file2_data[:-2, 1]

# Load data from file3
file3_data = np.loadtxt('power_spectrum_data_analytical_400.txt')
k_1d_file3, power_spectrum_file3 = file3_data[:-2, 0], file3_data[:-2, 1]

# Plot power spectra from file1 and file2
plt.plot(k_1d_file1, power_spectrum_file1, label='100^3')
plt.plot(k_1d_file2, power_spectrum_file2, label='200^3')
plt.plot(k_1d_file3, power_spectrum_file3, label='400^3')

# Plot the Kolmogorov slope
kolmogorov_line = k_1d_file3**(-5.0/3.0)
plt.plot(k_1d_file3, kolmogorov_line, label='Kolmogorov slope (-5/3)', linestyle='--')

# Add legend
plt.legend()

# Set log scale on both axes
plt.xscale('log')
plt.yscale('log')

# Set x-axis starting point
plt.xlim(left=6)

# Set labels
plt.xlabel('k')
plt.ylabel('k * E_k')

# Save the plot as a PNG file
plt.savefig('combined_power_spectra_v4.png')
