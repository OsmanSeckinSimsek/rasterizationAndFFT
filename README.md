# rasterizationAndFFT

# Python scripts in this Repository

1. **`raster.py`**: Reads a specific step from an h5 snapshot of SPH-EXA, creates a 3D Cartesian mesh, and performs a nearest neighbor search for each cell, to find the closest SPH particle and assign its velocity. Generates a file with the Cartesian grid and corresponding velocity components: x, y, z, vx, vy, vz
2. **`kdtree_raster.py`**: Same as above, but implemented a KD-tree over the scattered SPH particles and perform the nearest neighbor search as a query to the KD-tree structure, to speed things up.
3. **`power_spectra.py`**: Reads the data created by the raster, performs a 3D FFT, and calculates the 1D-averaged power spectra. Saves the spectra in a file: k, Ek, and plots it.
4. **`plot_spectra.py`**: Simple script to plot several power spectra on the same plot.
5. **`kolmogorov.py`**: Generates a synthetic velocity field that has a Kolmogorov energy distribution. It is used to validate power_spectra.py.