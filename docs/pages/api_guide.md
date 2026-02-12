# Public API Guide

This guide lists the primary public classes, functions, and modules exposed to users.

## Namespaces
- `cstone`: Octree, domain decomposition, SFC utilities, traversal, halos, fields, and utilities.
- `sphexa` (from IO helpers): minimal CLI/IO utilities for reading H5Part checkpoints.

## Core classes
- `cstone::Domain<Key, Real, Accelerator>`: Distributed domain management and halo exchange.
- `cstone::FocusedOctree<Key, Real, Accelerator>` (read-only via `Domain::focusTree()`): Focused octree view.
- `cstone::Octree<Key>` (read-only via `Domain::globalTree()`): Global octree.
- `Mesh<T>`: Application-side mesh and power spectrum routines in `main/src/mesh.hpp`.

## Key methods (selection)
- `Domain::sync(...)`: Distributes particles, builds trees, exchanges halos, and sorts by SFC.
- `Domain::reapplySync(...)`: Applies previous exchange patterns to additional fields.
- `Domain::exchangeHalos(...)`: Repeats halo exchange for new arrays.
- `Domain::startIndex()`, `endIndex()`, `nParticles()`, `nParticlesWithHalos()`: Access local ranges and sizes.

## SFC and octree utilities
- `cstone::decodeHilbert(key)`: Map Hilbert key to 3D indices.
- `cstone::computeSfcKeys(...)`: Compute SFC keys from positions (CPU/GPU variants).

## Traversal and halos
- `cstone::findPeersMac(...)`: Determine peer ranks for halo exchange using MAC.
- `cstone::Halos<...>::exchangeHalos(...)`: Low-level halo exchange invoked by `Domain`.

Refer to the generated API reference for full member listings.
