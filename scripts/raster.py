import numpy as np
import h5py
from numba import njit,prange

# Function to assign velocities using Numba JIT compilation
@njit(parallel=True)
def assign_velocities(x, y, z, vx, vy, vz, mesh_x, mesh_y, mesh_z, result_vx, result_vy, result_vz):
    for i in prange(mesh_x.shape[0]):
        print(f'Outer Loop Index i: {i}')
        for j in range(mesh_x.shape[1]):
            for k in range(mesh_x.shape[2]):
                min_distance = np.inf
                min_index = -1

                for p in range(x.shape[0]):
                    distance = (x[p] - mesh_x[i, j, k]) ** 2 + (y[p] - mesh_y[i, j, k]) ** 2 + (z[p] - mesh_z[i, j, k]) ** 2

                    if distance < min_distance:
                        min_distance = distance
                        min_index = p

                result_vx[i, j, k] = vx[min_index]
                result_vy[i, j, k] = vy[min_index]
                result_vz[i, j, k] = vz[min_index]

# Read particle positions and velocities from HDF5 file
file_name = 'turb_200_correct.h5'
with h5py.File(file_name, 'r') as file:
    x = file['Step#22/x'][:]
    y = file['Step#22/y'][:]
    z = file['Step#22/z'][:]
    vx = file['Step#22/vx'][:]
    vy = file['Step#22/vy'][:]
    vz = file['Step#22/vz'][:]

# Create a 3D Cartesian mesh
mesh_resolution = 400
mesh_x, mesh_y, mesh_z = np.meshgrid(np.linspace(-0.5, 0.5, mesh_resolution),
                                     np.linspace(-0.5, 0.5, mesh_resolution),
                                     np.linspace(-0.5, 0.5, mesh_resolution))

# Initialize result arrays
result_vx = np.zeros_like(mesh_x)
result_vy = np.zeros_like(mesh_y)
result_vz = np.zeros_like(mesh_z)

# Call the Numba-compiled function
assign_velocities(x, y, z, vx, vy, vz, mesh_x, mesh_y, mesh_z, result_vx, result_vy, result_vz)

#output the data
output_file_path = 'output_velocities_400.txt'
with open(output_file_path, 'w') as output_file:
    for i in range(mesh_x.shape[0]):
        print(f'Writing outer Loop Index i: {i}')
        for j in range(mesh_x.shape[1]):
            for k in range(mesh_x.shape[2]):
                output_file.write(f"{mesh_x[i, j, k]} {mesh_y[i, j, k]} {mesh_z[i, j, k]} "
                                  f"{result_vx[i, j, k]} {result_vy[i, j, k]} {result_vz[i, j, k]}\n")
