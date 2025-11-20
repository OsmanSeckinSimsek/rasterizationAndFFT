#include "mesh.hpp"
#include <cuda_runtime.h>


void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}    

// Kernel to process particles and classify them as local or remote
template<typename T>
__global__ void classifyParticlesKernel(KeyType* keys, T* x, T* y, T* z, T* vx, T* vy, T* vz, int numParticles,
                                        int gridDim, T Lmin, T Lmax, int* inboxLow, int* inboxHigh, int* inboxSize,
                                        int* procGrid, int rank,
                                        // Output arrays for local assignments
                                        uint64_t* localIndices, T* localDistances, T* localVx, T* localVy,
                                        T* localVz, int* localCount,
                                        // Output arrays for remote assignments
                                        int* remoteRanks, uint64_t* remoteIndices, T* remoteDistances, T* remoteVx,
                                        T* remoteVy, T* remoteVz, int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Decode key to get mesh indices using cstone
    auto     coords = cstone::decodeHilbert(keys[idx]);
    unsigned coordX = util::get<0>(coords);
    unsigned coordY = util::get<1>(coords);
    unsigned coordZ = util::get<2>(coords);

    unsigned divisor = 1 + (1 << 21) / gridDim;
    int      indexi  = coordX / divisor;
    int      indexj  = coordY / divisor;
    int      indexk  = coordZ / divisor;

    // Calculate distance to mesh cell center
    T cellCenterX = Lmin + (indexi + 0.5) * (Lmax - Lmin) / gridDim;
    T cellCenterY = Lmin + (indexj + 0.5) * (Lmax - Lmin) / gridDim;
    T cellCenterZ = Lmin + (indexk + 0.5) * (Lmax - Lmin) / gridDim;

    T xDist    = x[idx] - cellCenterX;
    T yDist    = y[idx] - cellCenterY;
    T zDist    = z[idx] - cellCenterZ;
    T distance = sqrt(xDist * xDist + yDist * yDist + zDist * zDist);

    // Determine which rank owns this mesh cell
    int xBox       = indexi / inboxSize[0];
    int yBox       = indexj / inboxSize[1];
    int zBox       = indexk / inboxSize[2];
    int targetRank = xBox + yBox * procGrid[0] + zBox * procGrid[0] * procGrid[1];

    // Calculate inbox index
    int      xLocal     = indexi % inboxSize[0];
    int      yLocal     = indexj % inboxSize[1];
    int      zLocal     = indexk % inboxSize[2];
    uint64_t inboxIndex = xLocal + yLocal * inboxSize[0] + zLocal * inboxSize[0] * inboxSize[1];

    if (targetRank == rank)
    {
        // Local assignment
        int pos             = atomicAdd(localCount, 1);
        localIndices[pos]   = inboxIndex;
        localDistances[pos] = distance;
        localVx[pos]        = vx[idx];
        localVy[pos]        = vy[idx];
        localVz[pos]        = vz[idx];
    }
    else
    {
        // Remote assignment
        int pos              = atomicAdd(remoteCount, 1);
        remoteRanks[pos]     = targetRank;
        remoteIndices[pos]   = inboxIndex;
        remoteDistances[pos] = distance;
        remoteVx[pos]        = vx[idx];
        remoteVy[pos]        = vy[idx];
        remoteVz[pos]        = vz[idx];
    }
}

// Kernel to update mesh with local assignments
template<typename T>
__global__ void updateMeshLocalKernel(uint64_t* indices, T* distances, T* vx, T* vy, T* vz, int count, T* meshVelX,
                                    T* meshVelY, T* meshVelZ, T* meshDistance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t meshIndex = indices[idx];
    T        distance  = distances[idx];

    // Atomically update if this particle is closer
    unsigned long long int* distance_as_ull = (unsigned long long int*)&meshDistance[meshIndex];
    unsigned long long int  old             = *distance_as_ull;
    unsigned long long int  assumed;

    do
    {
        assumed       = old;
        T oldDistance = __longlong_as_double(old);
        if (distance >= oldDistance) break;
        old = atomicCAS(distance_as_ull, assumed, __double_as_longlong(distance));
    } while (assumed != old);

    // If we updated the distance, also update velocities
    if (__longlong_as_double(old) > distance)
    {
        meshVelX[meshIndex] = vx[idx];
        meshVelY[meshIndex] = vy[idx];
        meshVelZ[meshIndex] = vz[idx];
    }
}

// Kernel to update mesh with received data
template<typename T>
__global__ void updateMeshRecvKernel(uint64_t* indices, T* distances, T* vx, T* vy, T* vz, int count, T* meshVelX,
                                    T* meshVelY, T* meshVelZ, T* meshDistance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t meshIndex = indices[idx];
    T        distance  = distances[idx];

    // Atomically update if this particle is closer
    unsigned long long int* distance_as_ull = (unsigned long long int*)&meshDistance[meshIndex];
    unsigned long long int  old             = *distance_as_ull;
    unsigned long long int  assumed;

    do
    {
        assumed       = old;
        T oldDistance = __longlong_as_double(old);
        if (distance >= oldDistance) break;
        old = atomicCAS(distance_as_ull, assumed, __double_as_longlong(distance));
    } while (assumed != old);

    // If we updated the distance, also update velocities
    if (__longlong_as_double(old) > distance)
    {
        meshVelX[meshIndex] = vx[idx];
        meshVelY[meshIndex] = vy[idx];
        meshVelZ[meshIndex] = vz[idx];
    }
}

template<typename T>
void rasterize_particles_to_mesh_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                          std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                          int powerDim)
{
    std::cout << "rank" << mesh.rank_ << " rasterize start (CUDA) " << powerDim << std::endl;
    std::cout << "rank" << mesh.rank_ << " keys between " << *keys.begin() << " - " << *keys.end() << std::endl;

    int      numParticles = keys.size();
    uint64_t inboxSize    = static_cast<uint64_t>(mesh.inbox_.size[0]) * mesh.inbox_.size[1] * mesh.inbox_.size[2];

    // ========== Device Memory Allocation ==========
    // Input particle data
    KeyType* d_keys;
    T *      d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;

    // Mesh configuration
    int *d_inboxLow, *d_inboxHigh, *d_inboxSize, *d_procGrid;

    // Local assignment buffers (max size = numParticles)
    uint64_t* d_localIndices;
    T *       d_localDistances, *d_localVx, *d_localVy, *d_localVz;
    int*      d_localCount;

    // Remote assignment buffers (max size = numParticles)
    int*      d_remoteRanks;
    uint64_t* d_remoteIndices;
    T *       d_remoteDistances, *d_remoteVx, *d_remoteVy, *d_remoteVz;
    int*      d_remoteCount;

    // Mesh data
    T *d_meshVelX, *d_meshVelY, *d_meshVelZ, *d_meshDistance;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_keys, numParticles * sizeof(KeyType)), "Allocating d_keys");
    checkCudaError(cudaMalloc(&d_x, numParticles * sizeof(T)), "Allocating d_x");
    checkCudaError(cudaMalloc(&d_y, numParticles * sizeof(T)), "Allocating d_y");
    checkCudaError(cudaMalloc(&d_z, numParticles * sizeof(T)), "Allocating d_z");
    checkCudaError(cudaMalloc(&d_vx, numParticles * sizeof(T)), "Allocating d_vx");
    checkCudaError(cudaMalloc(&d_vy, numParticles * sizeof(T)), "Allocating d_vy");
    checkCudaError(cudaMalloc(&d_vz, numParticles * sizeof(T)), "Allocating d_vz");

    checkCudaError(cudaMalloc(&d_inboxLow, 3 * sizeof(int)), "Allocating d_inboxLow");
    checkCudaError(cudaMalloc(&d_inboxHigh, 3 * sizeof(int)), "Allocating d_inboxHigh");
    checkCudaError(cudaMalloc(&d_inboxSize, 3 * sizeof(int)), "Allocating d_inboxSize");
    checkCudaError(cudaMalloc(&d_procGrid, 3 * sizeof(int)), "Allocating d_procGrid");

    checkCudaError(cudaMalloc(&d_localIndices, numParticles * sizeof(uint64_t)), "Allocating d_localIndices");
    checkCudaError(cudaMalloc(&d_localDistances, numParticles * sizeof(T)), "Allocating d_localDistances");
    checkCudaError(cudaMalloc(&d_localVx, numParticles * sizeof(T)), "Allocating d_localVx");
    checkCudaError(cudaMalloc(&d_localVy, numParticles * sizeof(T)), "Allocating d_localVy");
    checkCudaError(cudaMalloc(&d_localVz, numParticles * sizeof(T)), "Allocating d_localVz");
    checkCudaError(cudaMalloc(&d_localCount, sizeof(int)), "Allocating d_localCount");

    checkCudaError(cudaMalloc(&d_remoteRanks, numParticles * sizeof(int)), "Allocating d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, numParticles * sizeof(uint64_t)), "Allocating d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteDistances, numParticles * sizeof(T)), "Allocating d_remoteDistances");
    checkCudaError(cudaMalloc(&d_remoteVx, numParticles * sizeof(T)), "Allocating d_remoteVx");
    checkCudaError(cudaMalloc(&d_remoteVy, numParticles * sizeof(T)), "Allocating d_remoteVy");
    checkCudaError(cudaMalloc(&d_remoteVz, numParticles * sizeof(T)), "Allocating d_remoteVz");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "Allocating d_remoteCount");

    checkCudaError(cudaMalloc(&d_meshVelX, inboxSize * sizeof(T)), "Allocating d_meshVelX");
    checkCudaError(cudaMalloc(&d_meshVelY, inboxSize * sizeof(T)), "Allocating d_meshVelY");
    checkCudaError(cudaMalloc(&d_meshVelZ, inboxSize * sizeof(T)), "Allocating d_meshVelZ");
    checkCudaError(cudaMalloc(&d_meshDistance, inboxSize * sizeof(T)), "Allocating d_meshDistance");

    // ========== Copy Data to Device ==========
    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice),
                    "Copying keys to device");
    checkCudaError(cudaMemcpy(d_x, x.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying x to device");
    checkCudaError(cudaMemcpy(d_y, y.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying y to device");
    checkCudaError(cudaMemcpy(d_z, z.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying z to device");
    checkCudaError(cudaMemcpy(d_vx, vx.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying vx to device");
    checkCudaError(cudaMemcpy(d_vy, vy.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying vy to device");
    checkCudaError(cudaMemcpy(d_vz, vz.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying vz to device");

    // Copy mesh configuration
    std::array<int, 3> inboxLow     = {mesh.inbox_.low[0], mesh.inbox_.low[1], mesh.inbox_.low[2]};
    std::array<int, 3> inboxHigh    = {mesh.inbox_.high[0], mesh.inbox_.high[1], mesh.inbox_.high[2]};
    std::array<int, 3> inboxSizeArr = {mesh.inbox_.size[0], mesh.inbox_.size[1], mesh.inbox_.size[2]};
    checkCudaError(cudaMemcpy(d_inboxLow, inboxLow.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying inbox low to device");
    checkCudaError(cudaMemcpy(d_inboxHigh, inboxHigh.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying inbox high to device");
    checkCudaError(cudaMemcpy(d_inboxSize, inboxSizeArr.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying inbox size to device");
    checkCudaError(cudaMemcpy(d_procGrid, mesh.proc_grid_.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying proc_grid to device");

    // Initialize counters
    int zeroCount = 0;
    checkCudaError(cudaMemcpy(d_localCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                    "Initializing local count");
    checkCudaError(cudaMemcpy(d_remoteCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                    "Initializing remote count");

    // Copy mesh data
    checkCudaError(cudaMemcpy(d_meshVelX, mesh.velX_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh velX to device");
    checkCudaError(cudaMemcpy(d_meshVelY, mesh.velY_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh velY to device");
    checkCudaError(cudaMemcpy(d_meshVelZ, mesh.velZ_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh velZ to device");
    checkCudaError(cudaMemcpy(d_meshDistance, mesh.distance_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh distance to device");

    // ========== Launch Classification Kernel ==========
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    classifyParticlesKernel<T><<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_x, d_y, d_z, d_vx, d_vy, d_vz, numParticles, mesh.gridDim_, mesh.Lmin_, mesh.Lmax_, d_inboxLow, d_inboxHigh,
        d_inboxSize, d_procGrid, mesh.rank_, d_localIndices, d_localDistances, d_localVx, d_localVy, d_localVz,
        d_localCount, d_remoteRanks, d_remoteIndices, d_remoteDistances, d_remoteVx, d_remoteVy, d_remoteVz,
        d_remoteCount);

    checkCudaError(cudaDeviceSynchronize(), "Classification kernel execution");

    // ========== Get Counts from Device ==========
    int h_localCount, h_remoteCount;
    checkCudaError(cudaMemcpy(&h_localCount, d_localCount, sizeof(int), cudaMemcpyDeviceToHost),
                    "Copying local count to host");
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost),
                    "Copying remote count to host");

    std::cout << "rank = " << mesh.rank_ << " local count = " << h_localCount << " remote count = " << h_remoteCount
                << std::endl;

    // ========== Update Mesh with Local Assignments ==========
    if (h_localCount > 0)
    {
        int localBlocks = (h_localCount + threadsPerBlock - 1) / threadsPerBlock;
        updateMeshLocalKernel<T><<<localBlocks, threadsPerBlock>>>(d_localIndices, d_localDistances, d_localVx,
                                                                d_localVy, d_localVz, h_localCount, d_meshVelX,
                                                                d_meshVelY, d_meshVelZ, d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Local mesh update kernel");
    }

    // ========== Process Remote Assignments ==========
    std::vector<int>      h_remoteRanks(h_remoteCount);
    std::vector<uint64_t> h_remoteIndices(h_remoteCount);
    std::vector<T>        h_remoteDistances(h_remoteCount);
    std::vector<T>        h_remoteVx(h_remoteCount);
    std::vector<T>        h_remoteVy(h_remoteCount);
    std::vector<T>        h_remoteVz(h_remoteCount);

    if (h_remoteCount > 0)
    {
        checkCudaError(
            cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, h_remoteCount * sizeof(int), cudaMemcpyDeviceToHost),
            "Copying remote ranks to host");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, h_remoteCount * sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote indices to host");
        checkCudaError(cudaMemcpy(h_remoteDistances.data(), d_remoteDistances, h_remoteCount * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote distances to host");
        checkCudaError(cudaMemcpy(h_remoteVx.data(), d_remoteVx, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost),
                        "Copying remote vx to host");
        checkCudaError(cudaMemcpy(h_remoteVy.data(), d_remoteVy, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost),
                        "Copying remote vy to host");
        checkCudaError(cudaMemcpy(h_remoteVz.data(), d_remoteVz, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost),
                        "Copying remote vz to host");
    }

    // Organize remote data by rank
    for (int i = 0; i < h_remoteCount; i++)
    {
        int targetRank = h_remoteRanks[i];
        mesh.send_count[targetRank]++;
        mesh.vdataSender[targetRank].send_index.push_back(h_remoteIndices[i]);
        mesh.vdataSender[targetRank].send_distance.push_back(h_remoteDistances[i]);
        mesh.vdataSender[targetRank].send_vx.push_back(h_remoteVx[i]);
        mesh.vdataSender[targetRank].send_vy.push_back(h_remoteVy[i]);
        mesh.vdataSender[targetRank].send_vz.push_back(h_remoteVz[i]);
    }

    std::cout << "rank = " << mesh.rank_ << " particleIndex = " << numParticles << std::endl;
    for (int i = 0; i < mesh.numRanks_; i++)
        std::cout << "rank = " << mesh.rank_ << " send_count = " << mesh.send_count[i] << std::endl;

    // ========== MPI Communication ==========
    MPI_Alltoall(mesh.send_count.data(), 1, MpiType<int>{}, mesh.recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.send_disp[i + 1] = mesh.send_disp[i] + mesh.send_count[i];
        mesh.recv_disp[i + 1] = mesh.recv_disp[i] + mesh.recv_count[i];
    }

    // Prepare send buffers
    mesh.send_index.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_distance.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_vx.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_vy.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_vz.resize(mesh.send_disp[mesh.numRanks_]);
    std::cout << "rank = " << mesh.rank_ << " buffers allocated" << std::endl;

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        for (int j = mesh.send_disp[i]; j < mesh.send_disp[i + 1]; j++)
        {
            mesh.send_index[j]    = mesh.vdataSender[i].send_index[j - mesh.send_disp[i]];
            mesh.send_distance[j] = mesh.vdataSender[i].send_distance[j - mesh.send_disp[i]];
            mesh.send_vx[j]       = mesh.vdataSender[i].send_vx[j - mesh.send_disp[i]];
            mesh.send_vy[j]       = mesh.vdataSender[i].send_vy[j - mesh.send_disp[i]];
            mesh.send_vz[j]       = mesh.vdataSender[i].send_vz[j - mesh.send_disp[i]];
        }
    }
    std::cout << "rank = " << mesh.rank_ << " buffers transformed" << std::endl;

    // Prepare receive buffers
    mesh.recv_index.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_distance.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_vx.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_vy.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_vz.resize(mesh.recv_disp[mesh.numRanks_]);

    MPI_Alltoallv(mesh.send_index.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<uint64_t>{}, mesh.recv_index.data(),
                    mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_distance.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{}, mesh.recv_distance.data(),
                    mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_vx.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{}, mesh.recv_vx.data(),
                    mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_vy.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{}, mesh.recv_vy.data(),
                    mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_vz.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{}, mesh.recv_vz.data(),
                    mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    std::cout << "rank = " << mesh.rank_ << " alltoallv done!" << std::endl;

    // ========== Update Mesh with Received Data on GPU ==========
    if (mesh.recv_disp[mesh.numRanks_] > 0)
    {
        // Allocate device memory for received data
        uint64_t* d_recvIndices;
        T *       d_recvDistances, *d_recvVx, *d_recvVy, *d_recvVz;

        checkCudaError(cudaMalloc(&d_recvIndices, mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t)),
                        "Allocating d_recvIndices");
        checkCudaError(cudaMalloc(&d_recvDistances, mesh.recv_disp[mesh.numRanks_] * sizeof(T)),
                        "Allocating d_recvDistances");
        checkCudaError(cudaMalloc(&d_recvVx, mesh.recv_disp[mesh.numRanks_] * sizeof(T)), "Allocating d_recvVx");
        checkCudaError(cudaMalloc(&d_recvVy, mesh.recv_disp[mesh.numRanks_] * sizeof(T)), "Allocating d_recvVy");
        checkCudaError(cudaMalloc(&d_recvVz, mesh.recv_disp[mesh.numRanks_] * sizeof(T)), "Allocating d_recvVz");

        // Copy received data to device
        checkCudaError(cudaMemcpy(d_recvIndices, mesh.recv_index.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice),
                        "Copying recv indices to device");
        checkCudaError(cudaMemcpy(d_recvDistances, mesh.recv_distance.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T),
                                    cudaMemcpyHostToDevice),
                        "Copying recv distances to device");
        checkCudaError(
            cudaMemcpy(d_recvVx, mesh.recv_vx.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
            "Copying recv vx to device");
        checkCudaError(
            cudaMemcpy(d_recvVy, mesh.recv_vy.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
            "Copying recv vy to device");
        checkCudaError(
            cudaMemcpy(d_recvVz, mesh.recv_vz.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
            "Copying recv vz to device");

        // Launch kernel to update mesh
        int recvBlocks = (mesh.recv_disp[mesh.numRanks_] + threadsPerBlock - 1) / threadsPerBlock;
        updateMeshRecvKernel<T><<<recvBlocks, threadsPerBlock>>>(d_recvIndices, d_recvDistances, d_recvVx, d_recvVy,
                                                                d_recvVz, mesh.recv_disp[mesh.numRanks_], d_meshVelX,
                                                                d_meshVelY, d_meshVelZ, d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Received mesh update kernel");

        // Free temporary receive buffers
        cudaFree(d_recvIndices);
        cudaFree(d_recvDistances);
        cudaFree(d_recvVx);
        cudaFree(d_recvVy);
        cudaFree(d_recvVz);
    }

    // ========== Copy Updated Mesh Back to Host ==========
    checkCudaError(cudaMemcpy(mesh.velX_.data(), d_meshVelX, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh velX back to host");
    checkCudaError(cudaMemcpy(mesh.velY_.data(), d_meshVelY, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh velY back to host");
    checkCudaError(cudaMemcpy(mesh.velZ_.data(), d_meshVelZ, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh velZ back to host");
    checkCudaError(cudaMemcpy(mesh.distance_.data(), d_meshDistance, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh distance back to host");

    // ========== Free Device Memory ==========
    cudaFree(d_keys);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_inboxLow);
    cudaFree(d_inboxHigh);
    cudaFree(d_inboxSize);
    cudaFree(d_procGrid);
    cudaFree(d_localIndices);
    cudaFree(d_localDistances);
    cudaFree(d_localVx);
    cudaFree(d_localVy);
    cudaFree(d_localVz);
    cudaFree(d_localCount);
    cudaFree(d_remoteRanks);
    cudaFree(d_remoteIndices);
    cudaFree(d_remoteDistances);
    cudaFree(d_remoteVx);
    cudaFree(d_remoteVy);
    cudaFree(d_remoteVz);
    cudaFree(d_remoteCount);
    cudaFree(d_meshVelX);
    cudaFree(d_meshVelY);
    cudaFree(d_meshVelZ);
    cudaFree(d_meshDistance);

    // Clear send buffers
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSender[i].send_index.clear();
        mesh.vdataSender[i].send_distance.clear();
        mesh.vdataSender[i].send_vx.clear();
        mesh.vdataSender[i].send_vy.clear();
        mesh.vdataSender[i].send_vz.clear();
    }

    // Extrapolate empty cells
    mesh.extrapolateEmptyCellsFromNeighbors();

    std::cout << "rank = " << mesh.rank_ << " rasterize (CUDA) done!" << std::endl;
}

// Explicit template instantiation for double
template void rasterize_particles_to_mesh_cuda<double>(Mesh<double>&, std::vector<KeyType>, std::vector<double>,
                                                        std::vector<double>, std::vector<double>, std::vector<double>,
                                                        std::vector<double>, std::vector<double>, int);