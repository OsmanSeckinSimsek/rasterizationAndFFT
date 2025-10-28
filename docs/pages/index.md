# Project Documentation

Welcome to the API and usage documentation for `rasterizationAndFFT` and the embedded `cstone` Cornerstone octree library.

- Overview
- Getting Started
- Key Components
- Examples

## Overview
This repository combines:
- A high-performance, distributed octree/domain library (`cstone`) for N-body/SPH simulations on CPUs/GPUs.
- An application that rasterizes particles to a mesh and computes power spectra using HeFFTe/FFTW.

## Getting Started
- Build prerequisites: C++20, MPI, OpenMP, HeFFTe, optionally CUDA/HIP and HDF5.
- See the top-level `README.md` and `cstone/README.md` for build and run instructions.

## Key Components
- Domain decomposition and halo exchange: `cstone::Domain<Key, Real, Accelerator>` in `cstone/include/cstone/domain/domain.hpp`.
- Space-filling curves and trees: `cstone/sfc/*`, `cstone/tree/*`.
- Application mesh and pipeline: `main/src/mesh.hpp`, `main/src/power_spectrum.cpp`, `main/src/utils.hpp`.
- IO helpers: files in `extern/io`.
- Python utilities for rasterization/spectra: `scripts/*.py`.

## Examples
- See the dedicated pages in this documentation for hands-on examples.
