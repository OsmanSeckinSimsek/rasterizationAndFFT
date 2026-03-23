# Mesh and Power Spectrum application

The app rasterizes SPH particles onto a 3D mesh and computes a power spectrum via HeFFTe.

## Entry point
See `main/src/power_spectrum.cpp`. CLI options:
- `--checkpoint`: HDF5 file path
- `--stepNo`: integer step index in H5Part file
- `--meshSizeMultiplier`: scale factor for mesh size (currently informative)
- `--numShells`: number of radial shells for averaging

Run (example):
```bash
mpirun -n 4 ./power_spectrum \
  --checkpoint data/turb_50.h5 \
  --stepNo 0 \
  --numShells 128
```

## Mesh class
`main/src/mesh.hpp` provides a templated `Mesh<T>` with:
- Construction from `(rank, numRanks, gridDim, numShells)`
- `rasterize_particles_to_mesh(keys, x, y, z, vx, vy, vz, powerDim)`
- `calculate_power_spectrum()`

Typical usage:
```cpp
Mesh<double> mesh(rank, numRanks, gridDim, numShells);
mesh.rasterize_particles_to_mesh(keys, x, y, z, vx, vy, vz, powerDim);
mesh.calculate_power_spectrum();
```

## Utilities
- `initMpi()`, `exitSuccess()`, and `Timer` from `main/src/utils.hpp`.
