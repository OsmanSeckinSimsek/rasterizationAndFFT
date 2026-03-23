# Python utilities

This repo includes helper scripts for rasterization and spectra workflows.

- `scripts/raster.py`: Rasterizes HDF5 particle data to a Cartesian mesh via nearest neighbor.
- `scripts/kdtree_raster.py`: Rasterizes using a KD-tree for faster nearest neighbor queries.
- `scripts/power_spectra.py`: Reads mesh data, performs 3D FFT, computes 1D-averaged power spectra.
- `scripts/plot_spectra.py`: Plots multiple power spectra.
- `scripts/kolmogorov.py`: Generates a synthetic velocity field with Kolmogorov spectrum (for validation).

Example:
```bash
python3 scripts/raster.py --input data/turb_50.h5 --step 0 --output mesh.npz
python3 scripts/power_spectra.py --input mesh.npz --output spectrum.txt
python3 scripts/plot_spectra.py --inputs spectrum.txt
```
