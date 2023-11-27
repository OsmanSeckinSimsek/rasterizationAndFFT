import numpy as np
import h5py
import time
from scipy.spatial import cKDTree

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
elapsed   = end_time1 - start_time
print(f"Reading done. Time: {elapsed} s")

# Create a 3D Cartesian mesh
mesh_resolution = 200
mesh_x, mesh_y, mesh_z = np.meshgrid(np.linspace(-0.5, 0.5, mesh_resolution),
                                     np.linspace(-0.5, 0.5, mesh_resolution),
                                     np.linspace(-0.5, 0.5, mesh_resolution))

# Convert mesh coordinates to a flat array
target_grid_points = np.column_stack((mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()))
end_time2 = time.time()
elapsed  = end_time2 - end_time1
print(f"Creating Cartesian mesh. Time: {elapsed} s")

# Create KD-Tree from scattered points
scattered_points = np.column_stack((x, y, z))
kdtree = cKDTree(scattered_points)
end_time3 = time.time()
elapsed  = end_time3 - end_time2
print(f"Creating KD-tree. Time: {elapsed} s")

# Query the KD-Tree for nearest neighbors
distances, indices = kdtree.query(target_grid_points, k=1)
end_time4 = time.time()
elapsed  = end_time4 - end_time3
print(f"Querying KD-tree. Time: {elapsed} s")

# Interpolate values based on the nearest neighbors
interpolated_values = np.column_stack((vx[indices], vy[indices], vz[indices]))

# Reshape the interpolated values to match the shape of the target grid
interpolated_values_reshaped = interpolated_values.reshape((mesh_resolution, mesh_resolution, mesh_resolution, 3))
end_time5 = time.time()
elapsed  = end_time5 - end_time4
print(f"Interpolating. Time: {elapsed} s")

# Output the resulting velocities to a file
output_file_path = 'interpolated_velocities_200_KD.txt'
with open(output_file_path, 'w') as output_file:
    #output_file.write("x, y, z, interpolated_vx, interpolated_vy, interpolated_vz\n")
    for i in range(mesh_resolution):
        for j in range(mesh_resolution):
            for k in range(mesh_resolution):
                output_file.write(f"{mesh_x[i, j, k]}, {mesh_y[i, j, k]}, {mesh_z[i, j, k]}, "
                                  f"{interpolated_values_reshaped[i, j, k, 0]}, "
                                  f"{interpolated_values_reshaped[i, j, k, 1]}, "
                                  f"{interpolated_values_reshaped[i, j, k, 2]}\n")
end_time6 = time.time()
elapsed  = end_time6 - end_time5
print(f"Outputting. Time: {elapsed} s")
                
print("Done!")
elapsed  = end_time6 - start_time
print(f"Total elapsed time: {elapsed} s")