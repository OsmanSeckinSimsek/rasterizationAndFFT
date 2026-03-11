#include "mesh.hpp"
#include <cuda_runtime.h>
#include <type_traits>

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>

__device__ __forceinline__ double nvshmem_double_atomic_compare_swap(double* target, double compare, double value,
                                                                     int pe)
{
    unsigned long long compare_bits = __double_as_longlong(compare);
    unsigned long long value_bits   = __double_as_longlong(value);
    unsigned long long prior_bits =
        nvshmem_ulonglong_atomic_compare_swap(reinterpret_cast<unsigned long long*>(target), compare_bits, value_bits,
                                              pe);
    return __longlong_as_double(prior_bits);
}
#endif

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

// Pass 1: atomically record the minimum distance per cell — no velocity writes.
// Both local and recv variants share the same logic; a single kernel suffices.
template<typename T>
__global__ void updateMeshDistanceKernel(uint64_t* indices, T* distances, int count, T* meshDistance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t meshIndex = indices[idx];
    T        distance  = distances[idx];

    unsigned long long int* distance_as_ull = (unsigned long long int*)&meshDistance[meshIndex];
    unsigned long long int  old             = *distance_as_ull;
    unsigned long long int  assumed;

    do
    {
        assumed = old;
        if (distance >= __longlong_as_double(old)) break;
        old = atomicCAS(distance_as_ull, assumed, __double_as_longlong(distance));
    } while (assumed != old);
}

// Pass 2: write velocities only for the particle that owns the settled minimum.
// Must be launched after a cudaDeviceSynchronize() following all distance-update passes,
// so that meshDistance contains the true global minimum before any velocity is written.
template<typename T>
__global__ void assignVelocitiesKernel(uint64_t* indices, T* distances, T* vx, T* vy, T* vz, int count,
                                       T* meshVelX, T* meshVelY, T* meshVelZ, T* meshDistance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t meshIndex = indices[idx];

    // Bitwise comparison: only the particle whose distance exactly matches the
    // settled minimum gets to write — eliminates the race between CAS and stores.
    unsigned long long int stored = *(unsigned long long int*)&meshDistance[meshIndex];
    if (stored == __double_as_longlong(distances[idx]))
    {
        meshVelX[meshIndex] = vx[idx];
        meshVelY[meshIndex] = vy[idx];
        meshVelZ[meshIndex] = vz[idx];
    }
}

#ifdef USE_NVSHMEM
template<typename T>
__global__ void fillValueKernel(T* data, uint64_t offset, uint64_t count, T value)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    data[offset + idx] = value;
}

__global__ void updateMeshRemoteNvshmemKernelDouble(const int* remoteRanks, const uint64_t* remoteIndices,
                                                    const double* remoteDistances, const double* remoteVx,
                                                    const double* remoteVy, const double* remoteVz, int remoteCount,
                                                    double* meshVelX, double* meshVelY, double* meshVelZ,
                                                    double* meshDistance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= remoteCount) return;

    int      target    = remoteRanks[idx];
    uint64_t meshIndex = remoteIndices[idx];
    double   distance  = remoteDistances[idx];

    double prev = nvshmem_double_g(meshDistance + meshIndex, target);
    while (distance < prev)
    {
        double prior = nvshmem_double_atomic_compare_swap(meshDistance + meshIndex, prev, distance, target);
        if (prior == prev)
        {
            nvshmem_double_p(meshVelX + meshIndex, remoteVx[idx], target);
            nvshmem_double_p(meshVelY + meshIndex, remoteVy[idx], target);
            nvshmem_double_p(meshVelZ + meshIndex, remoteVz[idx], target);
            break;
        }
        prev = prior;
    }
}
#endif

template<typename T>
void rasterize_particles_to_mesh_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                          std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                          int powerDim)
{
    std::cout << "rank" << mesh.rank_ << " rasterize start (CUDA) " << powerDim << std::endl;
    std::cout << "rank" << mesh.rank_ << " keys between " << keys.front() << " - " << keys.back() << std::endl;

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

    // ========== Pass 1a: Settle minimum distances for local particles ==========
    if (h_localCount > 0)
    {
        int localBlocks = (h_localCount + threadsPerBlock - 1) / threadsPerBlock;
        updateMeshDistanceKernel<T><<<localBlocks, threadsPerBlock>>>(
            d_localIndices, d_localDistances, h_localCount, d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Local distance CAS kernel");
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

        // Pass 1b: settle minimum distances for received particles
        int recvBlocks = (mesh.recv_disp[mesh.numRanks_] + threadsPerBlock - 1) / threadsPerBlock;
        updateMeshDistanceKernel<T><<<recvBlocks, threadsPerBlock>>>(
            d_recvIndices, d_recvDistances, mesh.recv_disp[mesh.numRanks_], d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Recv distance CAS kernel");

        // Pass 2: all minimum distances are now settled — assign velocities.
        // Local particles first, then received; both check against the same meshDistance.
        if (h_localCount > 0)
        {
            int localBlocks = (h_localCount + threadsPerBlock - 1) / threadsPerBlock;
            assignVelocitiesKernel<T><<<localBlocks, threadsPerBlock>>>(
                d_localIndices, d_localDistances, d_localVx, d_localVy, d_localVz,
                h_localCount, d_meshVelX, d_meshVelY, d_meshVelZ, d_meshDistance);
        }
        assignVelocitiesKernel<T><<<recvBlocks, threadsPerBlock>>>(
            d_recvIndices, d_recvDistances, d_recvVx, d_recvVy, d_recvVz,
            mesh.recv_disp[mesh.numRanks_], d_meshVelX, d_meshVelY, d_meshVelZ, d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Velocity assignment kernel");

        // Free temporary receive buffers
        cudaFree(d_recvIndices);
        cudaFree(d_recvDistances);
        cudaFree(d_recvVx);
        cudaFree(d_recvVy);
        cudaFree(d_recvVz);
    }
    else if (h_localCount > 0)
    {
        // No recv data — pass 2 for local particles only
        int localBlocks = (h_localCount + threadsPerBlock - 1) / threadsPerBlock;
        assignVelocitiesKernel<T><<<localBlocks, threadsPerBlock>>>(
            d_localIndices, d_localDistances, d_localVx, d_localVy, d_localVz,
            h_localCount, d_meshVelX, d_meshVelY, d_meshVelZ, d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Velocity assignment kernel (local only)");
    }

    // ========== Keep Device Memory for FFT ==========
    if (mesh.d_velX_) cudaFree(mesh.d_velX_);
    if (mesh.d_velY_) cudaFree(mesh.d_velY_);
    if (mesh.d_velZ_) cudaFree(mesh.d_velZ_);
    if (mesh.d_distance_) cudaFree(mesh.d_distance_);

    mesh.d_velX_       = d_meshVelX;
    mesh.d_velY_       = d_meshVelY;
    mesh.d_velZ_       = d_meshVelZ;
    mesh.d_distance_   = d_meshDistance;
    mesh.gpuDataValid_ = true;

    // ========== Free Other Device Memory ==========
    cudaFree(d_keys);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_inboxLow);
    cudaFree(d_inboxHigh);
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
    // Note: d_meshVelX/Y/Z and d_meshDistance NOT freed — stored in mesh

    // Clear send buffers
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSender[i].send_index.clear();
        mesh.vdataSender[i].send_distance.clear();
        mesh.vdataSender[i].send_vx.clear();
        mesh.vdataSender[i].send_vy.clear();
        mesh.vdataSender[i].send_vz.clear();
    }

    // ========== GPU Extrapolation ==========
    T* d_srcVelX, *d_srcVelY, *d_srcVelZ;
    checkCudaError(cudaMalloc(&d_srcVelX, inboxSize * sizeof(T)), "d_srcVelX extrap");
    checkCudaError(cudaMalloc(&d_srcVelY, inboxSize * sizeof(T)), "d_srcVelY extrap");
    checkCudaError(cudaMalloc(&d_srcVelZ, inboxSize * sizeof(T)), "d_srcVelZ extrap");
    checkCudaError(cudaMemcpy(d_srcVelX, mesh.d_velX_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velX");
    checkCudaError(cudaMemcpy(d_srcVelY, mesh.d_velY_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velY");
    checkCudaError(cudaMemcpy(d_srcVelZ, mesh.d_velZ_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velZ");

    launchExtrapolateEmptyCellsKernel<T>(d_srcVelX, d_srcVelY, d_srcVelZ,
                                          mesh.d_velX_, mesh.d_velY_, mesh.d_velZ_,
                                          mesh.d_distance_, d_inboxSize,
                                          mesh.inbox_.size[0], mesh.inbox_.size[1], mesh.inbox_.size[2]);
    checkCudaError(cudaDeviceSynchronize(), "GPU extrapolate");

    cudaFree(d_srcVelX);
    cudaFree(d_srcVelY);
    cudaFree(d_srcVelZ);
    cudaFree(d_inboxSize); // freed here after extrapolation kernel

    std::cout << "rank = " << mesh.rank_ << " rasterize (CUDA) done!" << std::endl;
}

#ifdef USE_NVSHMEM
template<typename T>
void rasterize_particles_to_mesh_nvshmem(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                         std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                         int powerDim)
{
    static_assert(std::is_same_v<T, double>, "NVSHMEM rasterization currently supports double precision.");

    std::cout << "rank" << mesh.rank_ << " rasterize start (NVSHMEM) " << powerDim << std::endl;
    std::cout << "rank" << mesh.rank_ << " keys between " << keys.front() << " - " << keys.back() << std::endl;

    const int pe     = nvshmem_my_pe();
    const int npes   = nvshmem_n_pes();
    if (npes != mesh.numRanks_)
    {
        std::cerr << "NVSHMEM communicator size (" << npes << ") does not match mesh communicator (" << mesh.numRanks_
                  << ")." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (pe != mesh.rank_)
    {
        std::cerr << "NVSHMEM PE (" << pe << ") does not match mesh rank (" << mesh.rank_ << ")." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int      numParticles  = keys.size();
    uint64_t inboxSize     = static_cast<uint64_t>(mesh.inbox_.size[0]) * mesh.inbox_.size[1] * mesh.inbox_.size[2];
    uint64_t maxInboxSize  = 0;
    uint64_t localInboxSz  = inboxSize;
    MPI_Allreduce(&localInboxSz, &maxInboxSize, 1, MpiType<uint64_t>{}, MPI_MAX, MPI_COMM_WORLD);
    if (maxInboxSize == 0) { return; }

    // Input particle data
    KeyType* d_keys;
    T *      d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;

    // Mesh configuration
    int *d_inboxLow, *d_inboxHigh, *d_inboxSize, *d_procGrid;

    // Local assignment buffers
    uint64_t* d_localIndices;
    T *       d_localDistances, *d_localVx, *d_localVy, *d_localVz;
    int*      d_localCount;

    // Remote assignment buffers
    int*      d_remoteRanks;
    uint64_t* d_remoteIndices;
    T *       d_remoteDistances, *d_remoteVx, *d_remoteVy, *d_remoteVz;
    int*      d_remoteCount;

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

    // NVSHMEM symmetric mesh storage
    T* d_meshVelX = static_cast<T*>(nvshmem_malloc(maxInboxSize * sizeof(T)));
    T* d_meshVelY = static_cast<T*>(nvshmem_malloc(maxInboxSize * sizeof(T)));
    T* d_meshVelZ = static_cast<T*>(nvshmem_malloc(maxInboxSize * sizeof(T)));
    T* d_meshDistance = static_cast<T*>(nvshmem_malloc(maxInboxSize * sizeof(T)));

    if (!d_meshVelX || !d_meshVelY || !d_meshVelZ || !d_meshDistance)
    {
        std::cerr << "Failed to allocate NVSHMEM symmetric buffers of size " << maxInboxSize << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // ========== Copy Data to Device ==========
    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice),
                    "Copying keys to device");
    checkCudaError(cudaMemcpy(d_x, x.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying x to device");
    checkCudaError(cudaMemcpy(d_y, y.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying y to device");
    checkCudaError(cudaMemcpy(d_z, z.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying z to device");
    checkCudaError(cudaMemcpy(d_vx, vx.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying vx to device");
    checkCudaError(cudaMemcpy(d_vy, vy.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying vy to device");
    checkCudaError(cudaMemcpy(d_vz, vz.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying vz to device");

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

    int zeroCount = 0;
    checkCudaError(cudaMemcpy(d_localCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                    "Initializing local count");
    checkCudaError(cudaMemcpy(d_remoteCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                    "Initializing remote count");

    checkCudaError(cudaMemcpy(d_meshVelX, mesh.velX_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh velX to device");
    checkCudaError(cudaMemcpy(d_meshVelY, mesh.velY_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh velY to device");
    checkCudaError(cudaMemcpy(d_meshVelZ, mesh.velZ_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh velZ to device");
    checkCudaError(cudaMemcpy(d_meshDistance, mesh.distance_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Copying mesh distance to device");

    const int threadsPerBlock = 256;

    if (maxInboxSize > inboxSize)
    {
        uint64_t tail = maxInboxSize - inboxSize;
        checkCudaError(cudaMemset(d_meshVelX + inboxSize, 0, tail * sizeof(T)), "Zeroing mesh velX tail");
        checkCudaError(cudaMemset(d_meshVelY + inboxSize, 0, tail * sizeof(T)), "Zeroing mesh velY tail");
        checkCudaError(cudaMemset(d_meshVelZ + inboxSize, 0, tail * sizeof(T)), "Zeroing mesh velZ tail");
        int fillBlocks = (tail + threadsPerBlock - 1) / threadsPerBlock;
        fillValueKernel<T><<<fillBlocks, threadsPerBlock>>>(d_meshDistance, inboxSize, tail,
                                                            std::numeric_limits<T>::infinity());
        checkCudaError(cudaDeviceSynchronize(), "Initializing mesh distance tail");
    }

    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    classifyParticlesKernel<T><<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_x, d_y, d_z, d_vx, d_vy, d_vz, numParticles, mesh.gridDim_, mesh.Lmin_, mesh.Lmax_, d_inboxLow, d_inboxHigh,
        d_inboxSize, d_procGrid, mesh.rank_, d_localIndices, d_localDistances, d_localVx, d_localVy, d_localVz,
        d_localCount, d_remoteRanks, d_remoteIndices, d_remoteDistances, d_remoteVx, d_remoteVy, d_remoteVz,
        d_remoteCount);
    checkCudaError(cudaDeviceSynchronize(), "Classification kernel execution");

    int h_localCount, h_remoteCount;
    checkCudaError(cudaMemcpy(&h_localCount, d_localCount, sizeof(int), cudaMemcpyDeviceToHost),
                    "Copying local count to host");
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost),
                    "Copying remote count to host");

    std::cout << "rank = " << mesh.rank_ << " local count = " << h_localCount << " remote count = " << h_remoteCount
                << std::endl;

    if (h_localCount > 0)
    {
        int localBlocks = (h_localCount + threadsPerBlock - 1) / threadsPerBlock;
        updateMeshLocalKernel<T><<<localBlocks, threadsPerBlock>>>(d_localIndices, d_localDistances, d_localVx,
                                                                  d_localVy, d_localVz, h_localCount, d_meshVelX,
                                                                  d_meshVelY, d_meshVelZ, d_meshDistance);
        checkCudaError(cudaDeviceSynchronize(), "Local mesh update kernel");
    }

    int maxRemoteCount = 0;
    MPI_Allreduce(&h_remoteCount, &maxRemoteCount, 1, MpiType<int>{}, MPI_MAX, MPI_COMM_WORLD);

    if (maxRemoteCount > 0)
    {
        int remoteBlocks = (maxRemoteCount + threadsPerBlock - 1) / threadsPerBlock;
        void* kernelArgs[] = {&d_remoteRanks,     &d_remoteIndices, &d_remoteDistances, &d_remoteVx,
                              &d_remoteVy,        &d_remoteVz,      &h_remoteCount,     &d_meshVelX,
                              &d_meshVelY,        &d_meshVelZ,      &d_meshDistance};
        nvshmemx_collective_launch((const void*)updateMeshRemoteNvshmemKernelDouble, dim3(remoteBlocks),
                                   dim3(threadsPerBlock), kernelArgs, 0, 0);
        checkCudaError(cudaDeviceSynchronize(), "NVSHMEM remote update kernel");
    }

    nvshmem_barrier_all();

    checkCudaError(cudaMemcpy(mesh.velX_.data(), d_meshVelX, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh velX back to host");
    checkCudaError(cudaMemcpy(mesh.velY_.data(), d_meshVelY, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh velY back to host");
    checkCudaError(cudaMemcpy(mesh.velZ_.data(), d_meshVelZ, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh velZ back to host");
    checkCudaError(cudaMemcpy(mesh.distance_.data(), d_meshDistance, inboxSize * sizeof(T), cudaMemcpyDeviceToHost),
                    "Copying mesh distance back to host");

    // Free device memory
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

    nvshmem_free(d_meshVelX);
    nvshmem_free(d_meshVelY);
    nvshmem_free(d_meshVelZ);
    nvshmem_free(d_meshDistance);

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSender[i].send_index.clear();
        mesh.vdataSender[i].send_distance.clear();
        mesh.vdataSender[i].send_vx.clear();
        mesh.vdataSender[i].send_vy.clear();
        mesh.vdataSender[i].send_vz.clear();
        mesh.send_count[i] = 0;
    }

    mesh.extrapolateEmptyCellsFromNeighbors();
    std::cout << "rank = " << mesh.rank_ << " rasterize (NVSHMEM) done!" << std::endl;
}
#endif

// Device function for SPH cubic spline kernel
template<typename T>
__device__ T sphKernelDevice(T r, T h)
{
    if (h <= 0.0) return 0.0;
    
    T q = r / h;
    if (q >= 2.0) return 0.0;

    T sigma = 1.0 / (3.14159265358979323846 * h * h * h); // pi
    T factor = 0.0;

    if (q < 1.0)
    {
        factor = 1.0 - 1.5 * q * q + 0.75 * q * q * q;
    }
    else // 1.0 <= q < 2.0
    {
        T term = 2.0 - q;
        factor = 0.25 * term * term * term;
    }

    return sigma * factor;
}

// Device function to get cell center coordinate
template<typename T>
__device__ T getCellCenterDevice(int i, T Lmin, T Lmax, int gridDim)
{
    T deltaMesh = (Lmax - Lmin) / gridDim;
    T centerCoord = deltaMesh / 2;
    T startingCoord = Lmin + centerCoord;
    return startingCoord + deltaMesh * i;
}

// CUDA kernel to COUNT SPH contributions (first pass - counting only)
template<typename T>
__global__ void countSPHContributionsKernel(T* x, T* y, T* z, T* h,
                                             int numParticles, int gridDim, T Lmin, T Lmax,
                                             int* inboxSize, int* procGrid, int rank,
                                             int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    T px = x[idx];
    T py = y[idx];
    T pz = z[idx];
    T ph = h[idx];

    T deltaMesh = (Lmax - Lmin) / gridDim;
    T h_eff = (ph < deltaMesh) ? ph : deltaMesh;
    T searchRadius = 2.0 * h_eff;
    T searchRadiusSq = searchRadius * searchRadius;

    // Find the range of grid cells that could be within 2*h_eff
    int minI = static_cast<int>((px - searchRadius - Lmin) / deltaMesh);
    int maxI = static_cast<int>((px + searchRadius - Lmin) / deltaMesh) + 1;
    int minJ = static_cast<int>((py - searchRadius - Lmin) / deltaMesh);
    int maxJ = static_cast<int>((py + searchRadius - Lmin) / deltaMesh) + 1;
    int minK = static_cast<int>((pz - searchRadius - Lmin) / deltaMesh);
    int maxK = static_cast<int>((pz + searchRadius - Lmin) / deltaMesh) + 1;

    // Clamp to valid grid range
    minI = max(0, minI);
    maxI = min(gridDim, maxI);
    minJ = max(0, minJ);
    maxJ = min(gridDim, maxJ);
    minK = max(0, minK);
    maxK = min(gridDim, maxK);

    // Count remote contributions only
    int localCount = 0;
    for (int i = minI; i < maxI; i++)
    {
        for (int j = minJ; j < maxJ; j++)
        {
            for (int k = minK; k < maxK; k++)
            {
                T cellX = getCellCenterDevice(i, Lmin, Lmax, gridDim);
                T cellY = getCellCenterDevice(j, Lmin, Lmax, gridDim);
                T cellZ = getCellCenterDevice(k, Lmin, Lmax, gridDim);

                T dx = px - cellX;
                T dy = py - cellY;
                T dz = pz - cellZ;
                T distSq = dx * dx + dy * dy + dz * dz;

                if (distSq < searchRadiusSq)
                {
                    T dist = sqrt(distSq);
                    T weight = sphKernelDevice(dist, h_eff);

                    if (weight > 0.0)
                    {
                        // Determine which rank owns this mesh cell
                        int xBox = i / inboxSize[0];
                        int yBox = j / inboxSize[1];
                        int zBox = k / inboxSize[2];
                        int targetRank = xBox + yBox * procGrid[0] + zBox * procGrid[0] * procGrid[1];

                        if (targetRank != rank)
                        {
                            // Remote contribution - count it
                            atomicAdd(remoteCount, 1);
                        }
                    }
                }
            }
        }
    }
}

// CUDA kernel to compute SPH contributions for each particle
// For local contributions, directly accumulate to mesh arrays using atomics
// For remote contributions, store in buffers for MPI communication
template<typename T>
__global__ void computeSPHContributionsKernel(KeyType*  , T* x, T* y, T* z, T* vx, T* vy, T* vz, T* h,
                                              int numParticles, int gridDim, T Lmin, T Lmax,
                                              int* inboxLow, int* inboxHigh, int* inboxSize, int* procGrid, int rank,
                                              // Direct accumulation arrays for local contributions
                                              T* meshWeightSum, T* meshWeightedVelX, T* meshWeightedVelY, T* meshWeightedVelZ,
                                              // Output: contributions to send (remote only)
                                              int* remoteRanks, uint64_t* remoteIndices, T* remoteWeights,
                                              T* remoteWeightedVx, T* remoteWeightedVy, T* remoteWeightedVz,
                                              int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    T px = x[idx];
    T py = y[idx];
    T pz = z[idx];
    T pvx = vx[idx];
    T pvy = vy[idx];
    T pvz = vz[idx];
    T ph = h[idx];

    T deltaMesh = (Lmax - Lmin) / gridDim;
    // Use effective smoothing length: cap at cell size
    T h_eff = (ph < deltaMesh) ? ph : deltaMesh;
    T searchRadius = 2.0 * h_eff;
    T searchRadiusSq = searchRadius * searchRadius;

    // Find the range of grid cells that could be within 2*h_eff
    int minI = static_cast<int>((px - searchRadius - Lmin) / deltaMesh);
    int maxI = static_cast<int>((px + searchRadius - Lmin) / deltaMesh) + 1;
    int minJ = static_cast<int>((py - searchRadius - Lmin) / deltaMesh);
    int maxJ = static_cast<int>((py + searchRadius - Lmin) / deltaMesh) + 1;
    int minK = static_cast<int>((pz - searchRadius - Lmin) / deltaMesh);
    int maxK = static_cast<int>((pz + searchRadius - Lmin) / deltaMesh) + 1;

    // Clamp to valid grid range
    minI = max(0, minI);
    maxI = min(gridDim, maxI);
    minJ = max(0, minJ);
    maxJ = min(gridDim, maxJ);
    minK = max(0, minK);
    maxK = min(gridDim, maxK);

    // Iterate over potential cells
    for (int i = minI; i < maxI; i++)
    {
        for (int j = minJ; j < maxJ; j++)
        {
            for (int k = minK; k < maxK; k++)
            {
                // Calculate distance from particle to cell center
                T cellX = getCellCenterDevice(i, Lmin, Lmax, gridDim);
                T cellY = getCellCenterDevice(j, Lmin, Lmax, gridDim);
                T cellZ = getCellCenterDevice(k, Lmin, Lmax, gridDim);

                T dx = px - cellX;
                T dy = py - cellY;
                T dz = pz - cellZ;
                T distSq = dx * dx + dy * dy + dz * dz;

                // Check if within search radius
                if (distSq < searchRadiusSq)
                {
                    T dist = sqrt(distSq);
                    T weight = sphKernelDevice(dist, h_eff);

                    if (weight > 0.0)
                    {
                        // Calculate weighted velocity contribution
                        T weightedVx = pvx * weight;
                        T weightedVy = pvy * weight;
                        T weightedVz = pvz * weight;

                        // Determine which rank owns this mesh cell
                        int xBox = i / inboxSize[0];
                        int yBox = j / inboxSize[1];
                        int zBox = k / inboxSize[2];
                        int targetRank = xBox + yBox * procGrid[0] + zBox * procGrid[0] * procGrid[1];

                        // Calculate inbox index
                        int xLocal = i % inboxSize[0];
                        int yLocal = j % inboxSize[1];
                        int zLocal = k % inboxSize[2];
                        uint64_t inboxIndex = xLocal + yLocal * inboxSize[0] + zLocal * inboxSize[0] * inboxSize[1];

                        if (targetRank == rank)
                        {
                            // Local assignment - directly accumulate using atomics
                            atomicAdd(&meshWeightSum[inboxIndex], weight);
                            atomicAdd(&meshWeightedVelX[inboxIndex], weightedVx);
                            atomicAdd(&meshWeightedVelY[inboxIndex], weightedVy);
                            atomicAdd(&meshWeightedVelZ[inboxIndex], weightedVz);
                        }
                        else
                        {
                            // Remote assignment - store in buffer for MPI communication
                            int pos = atomicAdd(remoteCount, 1);
                            remoteRanks[pos] = targetRank;
                            remoteIndices[pos] = inboxIndex;
                            remoteWeights[pos] = weight;
                            remoteWeightedVx[pos] = weightedVx;
                            remoteWeightedVy[pos] = weightedVy;
                            remoteWeightedVz[pos] = weightedVz;
                        }
                    }
                }
            }
        }
    }
}

// CUDA kernel to accumulate SPH contributions atomically
template<typename T>
__global__ void accumulateSPHContributionsKernel(uint64_t* indices, T* weights, T* weightedVx, T* weightedVy,
                                                 T* weightedVz, int count, T* meshWeightSum, T* meshWeightedVelX,
                                                 T* meshWeightedVelY, T* meshWeightedVelZ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t meshIndex = indices[idx];
    T weight = weights[idx];

    // Atomically accumulate weights and weighted velocities
    atomicAdd(&meshWeightSum[meshIndex], weight);
    atomicAdd(&meshWeightedVelX[meshIndex], weightedVx[idx]);
    atomicAdd(&meshWeightedVelY[meshIndex], weightedVy[idx]);
    atomicAdd(&meshWeightedVelZ[meshIndex], weightedVz[idx]);
}

// CUDA kernel to normalize velocities by weight sum
template<typename T>
__global__ void normalizeSPHVelocitiesKernel(T* meshWeightSum, T* meshWeightedVelX, T* meshWeightedVelY,
                                             T* meshWeightedVelZ, T* meshVelX, T* meshVelY, T* meshVelZ,
                                             T* meshDistance, uint64_t inboxSize)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inboxSize) return;

    if (meshWeightSum[idx] > 0.0)
    {
        meshVelX[idx] = meshWeightedVelX[idx] / meshWeightSum[idx];
        meshVelY[idx] = meshWeightedVelY[idx] / meshWeightSum[idx];
        meshVelZ[idx] = meshWeightedVelZ[idx] / meshWeightSum[idx];
        meshDistance[idx] = T(0); // mark as filled for GPU extrapolation
    }
    // else: meshDistance stays infinity → cell will be extrapolated
}

// CUDA implementation of SPH rasterization
template<typename T>
void rasterize_particles_to_mesh_sph_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                         std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                         std::vector<T> h, int powerDim)
{
    std::cout << "rank" << mesh.rank_ << " rasterize start (CUDA SPH) " << powerDim << std::endl;
    std::cout << "rank" << mesh.rank_ << " keys between " << keys.front() << " - " << keys.back() << std::endl;

    int      numParticles = keys.size();
    uint64_t inboxSize    = static_cast<uint64_t>(mesh.inbox_.size[0]) * mesh.inbox_.size[1] * mesh.inbox_.size[2];

    // Reset SPH accumulation arrays and communication counters
    std::fill(mesh.weightSum_.begin(), mesh.weightSum_.end(), 0.0);
    std::fill(mesh.weightedVelX_.begin(), mesh.weightedVelX_.end(), 0.0);
    std::fill(mesh.weightedVelY_.begin(), mesh.weightedVelY_.end(), 0.0);
    std::fill(mesh.weightedVelZ_.begin(), mesh.weightedVelZ_.end(), 0.0);
    std::fill(mesh.send_count.begin(), mesh.send_count.end(), 0);

    // Clear SPH sender data
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSenderSPH[i].send_index.clear();
        mesh.vdataSenderSPH[i].send_weight.clear();
        mesh.vdataSenderSPH[i].send_weighted_vx.clear();
        mesh.vdataSenderSPH[i].send_weighted_vy.clear();
        mesh.vdataSenderSPH[i].send_weighted_vz.clear();
    }

    // ========== Device Memory Allocation ==========
    // Input particle data
    KeyType* d_keys;
    T *      d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_h;

    // Mesh configuration
    int *d_inboxLow, *d_inboxHigh, *d_inboxSize, *d_procGrid;

    // Remote contribution buffers (local contributions are accumulated directly to mesh arrays)
    int*      d_remoteRanks;
    uint64_t* d_remoteIndices;
    T *       d_remoteWeights, *d_remoteWeightedVx, *d_remoteWeightedVy, *d_remoteWeightedVz;
    int*      d_remoteCount;

    // SPH accumulation arrays
    T *d_meshWeightSum, *d_meshWeightedVelX, *d_meshWeightedVelY, *d_meshWeightedVelZ;
    T *d_meshVelX, *d_meshVelY, *d_meshVelZ;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_keys, numParticles * sizeof(KeyType)), "Allocating d_keys");
    checkCudaError(cudaMalloc(&d_x, numParticles * sizeof(T)), "Allocating d_x");
    checkCudaError(cudaMalloc(&d_y, numParticles * sizeof(T)), "Allocating d_y");
    checkCudaError(cudaMalloc(&d_z, numParticles * sizeof(T)), "Allocating d_z");
    checkCudaError(cudaMalloc(&d_vx, numParticles * sizeof(T)), "Allocating d_vx");
    checkCudaError(cudaMalloc(&d_vy, numParticles * sizeof(T)), "Allocating d_vy");
    checkCudaError(cudaMalloc(&d_vz, numParticles * sizeof(T)), "Allocating d_vz");
    checkCudaError(cudaMalloc(&d_h, numParticles * sizeof(T)), "Allocating d_h");

    checkCudaError(cudaMalloc(&d_inboxLow, 3 * sizeof(int)), "Allocating d_inboxLow");
    checkCudaError(cudaMalloc(&d_inboxHigh, 3 * sizeof(int)), "Allocating d_inboxHigh");
    checkCudaError(cudaMalloc(&d_inboxSize, 3 * sizeof(int)), "Allocating d_inboxSize");
    checkCudaError(cudaMalloc(&d_procGrid, 3 * sizeof(int)), "Allocating d_procGrid");

    // ========== First Pass: Count Remote Contributions Exactly ==========
    int* d_remoteCountTemp;
    checkCudaError(cudaMalloc(&d_remoteCountTemp, sizeof(int)), "Allocating d_remoteCountTemp");
    int zeroCount = 0;
    checkCudaError(cudaMemcpy(d_remoteCountTemp, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                    "Initializing remote count temp");

    // Copy mesh configuration for counting
    std::array<int, 3> inboxSizeArr = {mesh.inbox_.size[0], mesh.inbox_.size[1], mesh.inbox_.size[2]};
    checkCudaError(cudaMemcpy(d_inboxSize, inboxSizeArr.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying inbox size to device");
    checkCudaError(cudaMemcpy(d_procGrid, mesh.proc_grid_.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying proc_grid to device");

    // Launch counting kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    countSPHContributionsKernel<T><<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_z, d_h, numParticles, mesh.gridDim_, mesh.Lmin_, mesh.Lmax_,
        d_inboxSize, d_procGrid, mesh.rank_, d_remoteCountTemp);

    checkCudaError(cudaDeviceSynchronize(), "SPH counting kernel execution");

    // Get exact count
    int exactRemoteCount;
    checkCudaError(cudaMemcpy(&exactRemoteCount, d_remoteCountTemp, sizeof(int), cudaMemcpyDeviceToHost),
                    "Copying exact remote count to host");
    cudaFree(d_remoteCountTemp);

    std::cout << "rank = " << mesh.rank_ << " exact remote contributions count = " << exactRemoteCount << std::endl;

    // Allocate exact amount needed (with small safety margin)
    int maxRemoteContributions = exactRemoteCount + 1000; // Small safety margin for race conditions

    checkCudaError(cudaMalloc(&d_remoteRanks, maxRemoteContributions * sizeof(int)), "Allocating d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, maxRemoteContributions * sizeof(uint64_t)), "Allocating d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteWeights, maxRemoteContributions * sizeof(T)), "Allocating d_remoteWeights");
    checkCudaError(cudaMalloc(&d_remoteWeightedVx, maxRemoteContributions * sizeof(T)), "Allocating d_remoteWeightedVx");
    checkCudaError(cudaMalloc(&d_remoteWeightedVy, maxRemoteContributions * sizeof(T)), "Allocating d_remoteWeightedVy");
    checkCudaError(cudaMalloc(&d_remoteWeightedVz, maxRemoteContributions * sizeof(T)), "Allocating d_remoteWeightedVz");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "Allocating d_remoteCount");

    checkCudaError(cudaMalloc(&d_meshWeightSum, inboxSize * sizeof(T)), "Allocating d_meshWeightSum");
    checkCudaError(cudaMalloc(&d_meshWeightedVelX, inboxSize * sizeof(T)), "Allocating d_meshWeightedVelX");
    checkCudaError(cudaMalloc(&d_meshWeightedVelY, inboxSize * sizeof(T)), "Allocating d_meshWeightedVelY");
    checkCudaError(cudaMalloc(&d_meshWeightedVelZ, inboxSize * sizeof(T)), "Allocating d_meshWeightedVelZ");
    checkCudaError(cudaMalloc(&d_meshVelX, inboxSize * sizeof(T)), "Allocating d_meshVelX");
    checkCudaError(cudaMalloc(&d_meshVelY, inboxSize * sizeof(T)), "Allocating d_meshVelY");
    checkCudaError(cudaMalloc(&d_meshVelZ, inboxSize * sizeof(T)), "Allocating d_meshVelZ");

    // ========== Copy Data to Device ==========
    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice),
                    "Copying keys to device");
    checkCudaError(cudaMemcpy(d_x, x.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying x to device");
    checkCudaError(cudaMemcpy(d_y, y.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying y to device");
    checkCudaError(cudaMemcpy(d_z, z.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying z to device");
    checkCudaError(cudaMemcpy(d_vx, vx.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying vx to device");
    checkCudaError(cudaMemcpy(d_vy, vy.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying vy to device");
    checkCudaError(cudaMemcpy(d_vz, vz.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying vz to device");
    checkCudaError(cudaMemcpy(d_h, h.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "Copying h to device");

    // Copy remaining mesh configuration (inboxLow and inboxHigh needed for compute kernel)
    // Note: d_inboxSize and d_procGrid already copied during counting phase
    std::array<int, 3> inboxLow     = {mesh.inbox_.low[0], mesh.inbox_.low[1], mesh.inbox_.low[2]};
    std::array<int, 3> inboxHigh    = {mesh.inbox_.high[0], mesh.inbox_.high[1], mesh.inbox_.high[2]};
    checkCudaError(cudaMemcpy(d_inboxLow, inboxLow.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying inbox low to device");
    checkCudaError(cudaMemcpy(d_inboxHigh, inboxHigh.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
                    "Copying inbox high to device");

    // Initialize counters and accumulation arrays
    zeroCount = 0;
    checkCudaError(cudaMemcpy(d_remoteCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                    "Initializing remote count");
    checkCudaError(cudaMemset(d_meshWeightSum, 0, inboxSize * sizeof(T)), "Initializing mesh weight sum");
    checkCudaError(cudaMemset(d_meshWeightedVelX, 0, inboxSize * sizeof(T)), "Initializing mesh weighted velX");
    checkCudaError(cudaMemset(d_meshWeightedVelY, 0, inboxSize * sizeof(T)), "Initializing mesh weighted velY");
    checkCudaError(cudaMemset(d_meshWeightedVelZ, 0, inboxSize * sizeof(T)), "Initializing mesh weighted velZ");

    // ========== Launch SPH Contribution Kernel ==========
    threadsPerBlock = 256;
    blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    computeSPHContributionsKernel<T><<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, numParticles, mesh.gridDim_, mesh.Lmin_, mesh.Lmax_,
        d_inboxLow, d_inboxHigh, d_inboxSize, d_procGrid, mesh.rank_, d_meshWeightSum, d_meshWeightedVelX,
        d_meshWeightedVelY, d_meshWeightedVelZ, d_remoteRanks, d_remoteIndices, d_remoteWeights,
        d_remoteWeightedVx, d_remoteWeightedVy, d_remoteWeightedVz, d_remoteCount);

    checkCudaError(cudaDeviceSynchronize(), "SPH contribution kernel execution");

    // ========== Get Counts from Device ==========
    int h_remoteCount;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost),
                    "Copying remote count to host");

    std::cout << "rank = " << mesh.rank_ << " remote count = " << h_remoteCount << std::endl;
    
    // Verify count matches (should be very close, small difference possible due to race conditions)
    if (h_remoteCount > maxRemoteContributions)
    {
        std::cerr << "ERROR: Remote contributions buffer overflow! Count = " << h_remoteCount 
                  << " but buffer size = " << maxRemoteContributions << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // ========== Process Remote Contributions ==========
    std::vector<int>      h_remoteRanks(h_remoteCount);
    std::vector<uint64_t> h_remoteIndices(h_remoteCount);
    std::vector<T>        h_remoteWeights(h_remoteCount);
    std::vector<T>        h_remoteWeightedVx(h_remoteCount);
    std::vector<T>        h_remoteWeightedVy(h_remoteCount);
    std::vector<T>        h_remoteWeightedVz(h_remoteCount);

    if (h_remoteCount > 0)
    {
        checkCudaError(
            cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, h_remoteCount * sizeof(int), cudaMemcpyDeviceToHost),
            "Copying remote ranks to host");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, h_remoteCount * sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote indices to host");
        checkCudaError(cudaMemcpy(h_remoteWeights.data(), d_remoteWeights, h_remoteCount * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote weights to host");
        checkCudaError(cudaMemcpy(h_remoteWeightedVx.data(), d_remoteWeightedVx, h_remoteCount * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote weighted vx to host");
        checkCudaError(cudaMemcpy(h_remoteWeightedVy.data(), d_remoteWeightedVy, h_remoteCount * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote weighted vy to host");
        checkCudaError(cudaMemcpy(h_remoteWeightedVz.data(), d_remoteWeightedVz, h_remoteCount * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                        "Copying remote weighted vz to host");
    }

    // Organize remote data by rank
    for (int i = 0; i < h_remoteCount; i++)
    {
        int targetRank = h_remoteRanks[i];
        mesh.send_count[targetRank]++;
        mesh.vdataSenderSPH[targetRank].send_index.push_back(h_remoteIndices[i]);
        mesh.vdataSenderSPH[targetRank].send_weight.push_back(h_remoteWeights[i]);
        mesh.vdataSenderSPH[targetRank].send_weighted_vx.push_back(h_remoteWeightedVx[i]);
        mesh.vdataSenderSPH[targetRank].send_weighted_vy.push_back(h_remoteWeightedVy[i]);
        mesh.vdataSenderSPH[targetRank].send_weighted_vz.push_back(h_remoteWeightedVz[i]);
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
    mesh.send_index_sph.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_weight.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_weighted_vx.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_weighted_vy.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_weighted_vz.resize(mesh.send_disp[mesh.numRanks_]);
    std::cout << "rank = " << mesh.rank_ << " buffers allocated" << std::endl;

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        for (int j = mesh.send_disp[i]; j < mesh.send_disp[i + 1]; j++)
        {
            int localIdx = j - mesh.send_disp[i];
            mesh.send_index_sph[j] = mesh.vdataSenderSPH[i].send_index[localIdx];
            mesh.send_weight[j] = mesh.vdataSenderSPH[i].send_weight[localIdx];
            mesh.send_weighted_vx[j] = mesh.vdataSenderSPH[i].send_weighted_vx[localIdx];
            mesh.send_weighted_vy[j] = mesh.vdataSenderSPH[i].send_weighted_vy[localIdx];
            mesh.send_weighted_vz[j] = mesh.vdataSenderSPH[i].send_weighted_vz[localIdx];
        }
    }
    std::cout << "rank = " << mesh.rank_ << " buffers transformed" << std::endl;

    // Prepare receive buffers
    mesh.recv_index_sph.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_weight.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_weighted_vx.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_weighted_vy.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_weighted_vz.resize(mesh.recv_disp[mesh.numRanks_]);

    MPI_Alltoallv(mesh.send_index_sph.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<uint64_t>{},
                  mesh.recv_index_sph.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<uint64_t>{},
                  MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_weight.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_weight.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_weighted_vx.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_weighted_vx.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{},
                  MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_weighted_vy.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_weighted_vy.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{},
                  MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_weighted_vz.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_weighted_vz.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{},
                  MPI_COMM_WORLD);
    std::cout << "rank = " << mesh.rank_ << " alltoallv done!" << std::endl;

    // ========== Accumulate Received Contributions on GPU ==========
    if (mesh.recv_disp[mesh.numRanks_] > 0)
    {
        // Allocate device memory for received data
        uint64_t* d_recvIndices;
        T *       d_recvWeights, *d_recvWeightedVx, *d_recvWeightedVy, *d_recvWeightedVz;

        checkCudaError(cudaMalloc(&d_recvIndices, mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t)),
                        "Allocating d_recvIndices");
        checkCudaError(cudaMalloc(&d_recvWeights, mesh.recv_disp[mesh.numRanks_] * sizeof(T)),
                        "Allocating d_recvWeights");
        checkCudaError(cudaMalloc(&d_recvWeightedVx, mesh.recv_disp[mesh.numRanks_] * sizeof(T)),
                        "Allocating d_recvWeightedVx");
        checkCudaError(cudaMalloc(&d_recvWeightedVy, mesh.recv_disp[mesh.numRanks_] * sizeof(T)),
                        "Allocating d_recvWeightedVy");
        checkCudaError(cudaMalloc(&d_recvWeightedVz, mesh.recv_disp[mesh.numRanks_] * sizeof(T)),
                        "Allocating d_recvWeightedVz");

        // Copy received data to device
        checkCudaError(cudaMemcpy(d_recvIndices, mesh.recv_index_sph.data(),
                                    mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t), cudaMemcpyHostToDevice),
                        "Copying recv indices to device");
        checkCudaError(cudaMemcpy(d_recvWeights, mesh.recv_weight.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T),
                                    cudaMemcpyHostToDevice),
                        "Copying recv weights to device");
        checkCudaError(cudaMemcpy(d_recvWeightedVx, mesh.recv_weighted_vx.data(),
                                    mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
                        "Copying recv weighted vx to device");
        checkCudaError(cudaMemcpy(d_recvWeightedVy, mesh.recv_weighted_vy.data(),
                                    mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
                        "Copying recv weighted vy to device");
        checkCudaError(cudaMemcpy(d_recvWeightedVz, mesh.recv_weighted_vz.data(),
                                    mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
                        "Copying recv weighted vz to device");

        // Launch kernel to accumulate received contributions
        int recvBlocks = (mesh.recv_disp[mesh.numRanks_] + threadsPerBlock - 1) / threadsPerBlock;
        accumulateSPHContributionsKernel<T><<<recvBlocks, threadsPerBlock>>>(
            d_recvIndices, d_recvWeights, d_recvWeightedVx, d_recvWeightedVy, d_recvWeightedVz,
            mesh.recv_disp[mesh.numRanks_], d_meshWeightSum, d_meshWeightedVelX, d_meshWeightedVelY,
            d_meshWeightedVelZ);
        checkCudaError(cudaDeviceSynchronize(), "Received SPH accumulation kernel");

        // Free temporary receive buffers
        cudaFree(d_recvIndices);
        cudaFree(d_recvWeights);
        cudaFree(d_recvWeightedVx);
        cudaFree(d_recvWeightedVy);
        cudaFree(d_recvWeightedVz);
    }

    // ========== Normalize Velocities ==========
    // Allocate and initialize d_meshDistance (infinity = empty, 0 = filled)
    T* d_meshDistance;
    checkCudaError(cudaMalloc(&d_meshDistance, inboxSize * sizeof(T)), "Allocating d_meshDistance SPH");
    checkCudaError(cudaMemcpy(d_meshDistance, mesh.distance_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                    "Initializing d_meshDistance SPH");

    int normalizeBlocks = (inboxSize + threadsPerBlock - 1) / threadsPerBlock;
    normalizeSPHVelocitiesKernel<T><<<normalizeBlocks, threadsPerBlock>>>(
        d_meshWeightSum, d_meshWeightedVelX, d_meshWeightedVelY, d_meshWeightedVelZ,
        d_meshVelX, d_meshVelY, d_meshVelZ, d_meshDistance, inboxSize);
    checkCudaError(cudaDeviceSynchronize(), "Normalize SPH velocities kernel");

    // ========== Keep Device Memory for FFT ==========
    if (mesh.d_velX_) cudaFree(mesh.d_velX_);
    if (mesh.d_velY_) cudaFree(mesh.d_velY_);
    if (mesh.d_velZ_) cudaFree(mesh.d_velZ_);
    if (mesh.d_distance_) cudaFree(mesh.d_distance_);

    mesh.d_velX_       = d_meshVelX;
    mesh.d_velY_       = d_meshVelY;
    mesh.d_velZ_       = d_meshVelZ;
    mesh.d_distance_   = d_meshDistance;
    mesh.gpuDataValid_ = true;

    // ========== Free Other Device Memory ==========
    cudaFree(d_keys);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_h);
    cudaFree(d_inboxLow);
    cudaFree(d_inboxHigh);
    cudaFree(d_procGrid);
    cudaFree(d_remoteRanks);
    cudaFree(d_remoteIndices);
    cudaFree(d_remoteWeights);
    cudaFree(d_remoteWeightedVx);
    cudaFree(d_remoteWeightedVy);
    cudaFree(d_remoteWeightedVz);
    cudaFree(d_remoteCount);
    cudaFree(d_meshWeightSum);
    cudaFree(d_meshWeightedVelX);
    cudaFree(d_meshWeightedVelY);
    cudaFree(d_meshWeightedVelZ);
    // Note: d_meshVelX/Y/Z and d_meshDistance NOT freed — stored in mesh

    // Clear send buffers
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSenderSPH[i].send_index.clear();
        mesh.vdataSenderSPH[i].send_weight.clear();
        mesh.vdataSenderSPH[i].send_weighted_vx.clear();
        mesh.vdataSenderSPH[i].send_weighted_vy.clear();
        mesh.vdataSenderSPH[i].send_weighted_vz.clear();
    }

    // ========== GPU Extrapolation ==========
    T* d_srcVelX, *d_srcVelY, *d_srcVelZ;
    checkCudaError(cudaMalloc(&d_srcVelX, inboxSize * sizeof(T)), "d_srcVelX extrap SPH");
    checkCudaError(cudaMalloc(&d_srcVelY, inboxSize * sizeof(T)), "d_srcVelY extrap SPH");
    checkCudaError(cudaMalloc(&d_srcVelZ, inboxSize * sizeof(T)), "d_srcVelZ extrap SPH");
    checkCudaError(cudaMemcpy(d_srcVelX, mesh.d_velX_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velX SPH");
    checkCudaError(cudaMemcpy(d_srcVelY, mesh.d_velY_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velY SPH");
    checkCudaError(cudaMemcpy(d_srcVelZ, mesh.d_velZ_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velZ SPH");

    launchExtrapolateEmptyCellsKernel<T>(d_srcVelX, d_srcVelY, d_srcVelZ,
                                          mesh.d_velX_, mesh.d_velY_, mesh.d_velZ_,
                                          mesh.d_distance_, d_inboxSize,
                                          mesh.inbox_.size[0], mesh.inbox_.size[1], mesh.inbox_.size[2]);
    checkCudaError(cudaDeviceSynchronize(), "GPU extrapolate SPH");

    cudaFree(d_srcVelX);
    cudaFree(d_srcVelY);
    cudaFree(d_srcVelZ);
    cudaFree(d_inboxSize); // freed here after extrapolation kernel is done

    std::cout << "rank = " << mesh.rank_ << " rasterize (CUDA SPH) done!" << std::endl;
}

// ============================================================
// Cell-average interpolation kernels
// ============================================================

// Classify each particle as local or remote (no distance — cell membership only).
template<typename T>
__global__ void classifyParticlesCellAvgKernel(KeyType* keys, T* vx, T* vy, T* vz, int numParticles,
                                               int gridDim, int* inboxSize, int* procGrid, int rank,
                                               uint64_t* localIndices, T* localVx, T* localVy, T* localVz,
                                               int* localCount,
                                               int* remoteRanks, uint64_t* remoteIndices,
                                               T* remoteVx, T* remoteVy, T* remoteVz, int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    auto     coords  = cstone::decodeHilbert(keys[idx]);
    unsigned divisor = 1 + (1 << 21) / gridDim;
    int      indexi  = util::get<0>(coords) / divisor;
    int      indexj  = util::get<1>(coords) / divisor;
    int      indexk  = util::get<2>(coords) / divisor;

    int xBox       = indexi / inboxSize[0];
    int yBox       = indexj / inboxSize[1];
    int zBox       = indexk / inboxSize[2];
    int targetRank = xBox + yBox * procGrid[0] + zBox * procGrid[0] * procGrid[1];

    int      xLocal     = indexi % inboxSize[0];
    int      yLocal     = indexj % inboxSize[1];
    int      zLocal     = indexk % inboxSize[2];
    uint64_t inboxIndex = xLocal + yLocal * inboxSize[0] + zLocal * inboxSize[0] * inboxSize[1];

    if (targetRank == rank)
    {
        int pos          = atomicAdd(localCount, 1);
        localIndices[pos] = inboxIndex;
        localVx[pos]      = vx[idx];
        localVy[pos]      = vy[idx];
        localVz[pos]      = vz[idx];
    }
    else
    {
        int pos              = atomicAdd(remoteCount, 1);
        remoteRanks[pos]     = targetRank;
        remoteIndices[pos]   = inboxIndex;
        remoteVx[pos]        = vx[idx];
        remoteVy[pos]        = vy[idx];
        remoteVz[pos]        = vz[idx];
    }
}

// Accumulate velocity contributions into per-cell sums (atomic).
template<typename T>
__global__ void accumulateCellAvgKernel(uint64_t* indices, T* vx, T* vy, T* vz, int count,
                                        T* velXSum, T* velYSum, T* velZSum, int* cellCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t cell = indices[idx];
    atomicAdd(&velXSum[cell], vx[idx]);
    atomicAdd(&velYSum[cell], vy[idx]);
    atomicAdd(&velZSum[cell], vz[idx]);
    atomicAdd(&cellCount[cell], 1);
}

// Divide accumulated sums by count; mark filled cells with distance = 0.
template<typename T>
__global__ void normalizeCellAvgKernel(T* velXSum, T* velYSum, T* velZSum, int* cellCount,
                                       T* velX, T* velY, T* velZ, T* meshDistance, uint64_t inboxSize)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inboxSize) return;

    if (cellCount[idx] > 0)
    {
        T n         = static_cast<T>(cellCount[idx]);
        velX[idx]   = velXSum[idx] / n;
        velY[idx]   = velYSum[idx] / n;
        velZ[idx]   = velZSum[idx] / n;
        meshDistance[idx] = T(0); // finite sentinel → cell is filled
    }
    // else: velX/Y/Z stays 0, meshDistance stays infinity → extrapolated on host
}

template<typename T>
void rasterize_particles_to_mesh_cell_avg_cuda(Mesh<T>& mesh, std::vector<KeyType> keys,
                                               std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                               std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                               int powerDim)
{
    std::cout << "rank" << mesh.rank_ << " rasterize start (CUDA cell_avg) " << powerDim << std::endl;
    std::cout << "rank" << mesh.rank_ << " keys between " << keys.front() << " - " << keys.back() << std::endl;

    int      numParticles = keys.size();
    uint64_t inboxSize    = static_cast<uint64_t>(mesh.inbox_.size[0]) * mesh.inbox_.size[1] * mesh.inbox_.size[2];

    // ===== Device allocations =====
    KeyType*  d_keys;
    T *       d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    int *     d_inboxSize, *d_procGrid;

    uint64_t* d_localIndices;
    T *       d_localVx, *d_localVy, *d_localVz;
    int*      d_localCount;

    int*      d_remoteRanks;
    uint64_t* d_remoteIndices;
    T *       d_remoteVx, *d_remoteVy, *d_remoteVz;
    int*      d_remoteCount;

    T *       d_velXSum, *d_velYSum, *d_velZSum;
    int*      d_cellCount;
    T *       d_meshVelX, *d_meshVelY, *d_meshVelZ, *d_meshDistance;

    checkCudaError(cudaMalloc(&d_keys, numParticles * sizeof(KeyType)), "d_keys");
    checkCudaError(cudaMalloc(&d_x, numParticles * sizeof(T)), "d_x");
    checkCudaError(cudaMalloc(&d_y, numParticles * sizeof(T)), "d_y");
    checkCudaError(cudaMalloc(&d_z, numParticles * sizeof(T)), "d_z");
    checkCudaError(cudaMalloc(&d_vx, numParticles * sizeof(T)), "d_vx");
    checkCudaError(cudaMalloc(&d_vy, numParticles * sizeof(T)), "d_vy");
    checkCudaError(cudaMalloc(&d_vz, numParticles * sizeof(T)), "d_vz");
    checkCudaError(cudaMalloc(&d_inboxSize, 3 * sizeof(int)), "d_inboxSize");
    checkCudaError(cudaMalloc(&d_procGrid,  3 * sizeof(int)), "d_procGrid");

    checkCudaError(cudaMalloc(&d_localIndices,  numParticles * sizeof(uint64_t)), "d_localIndices");
    checkCudaError(cudaMalloc(&d_localVx,       numParticles * sizeof(T)), "d_localVx");
    checkCudaError(cudaMalloc(&d_localVy,       numParticles * sizeof(T)), "d_localVy");
    checkCudaError(cudaMalloc(&d_localVz,       numParticles * sizeof(T)), "d_localVz");
    checkCudaError(cudaMalloc(&d_localCount,    sizeof(int)), "d_localCount");

    checkCudaError(cudaMalloc(&d_remoteRanks,    numParticles * sizeof(int)),      "d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices,  numParticles * sizeof(uint64_t)), "d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteVx,       numParticles * sizeof(T)), "d_remoteVx");
    checkCudaError(cudaMalloc(&d_remoteVy,       numParticles * sizeof(T)), "d_remoteVy");
    checkCudaError(cudaMalloc(&d_remoteVz,       numParticles * sizeof(T)), "d_remoteVz");
    checkCudaError(cudaMalloc(&d_remoteCount,    sizeof(int)), "d_remoteCount");

    checkCudaError(cudaMalloc(&d_velXSum,     inboxSize * sizeof(T)),   "d_velXSum");
    checkCudaError(cudaMalloc(&d_velYSum,     inboxSize * sizeof(T)),   "d_velYSum");
    checkCudaError(cudaMalloc(&d_velZSum,     inboxSize * sizeof(T)),   "d_velZSum");
    checkCudaError(cudaMalloc(&d_cellCount,   inboxSize * sizeof(int)), "d_cellCount");
    checkCudaError(cudaMalloc(&d_meshVelX,    inboxSize * sizeof(T)),   "d_meshVelX");
    checkCudaError(cudaMalloc(&d_meshVelY,    inboxSize * sizeof(T)),   "d_meshVelY");
    checkCudaError(cudaMalloc(&d_meshVelZ,    inboxSize * sizeof(T)),   "d_meshVelZ");
    checkCudaError(cudaMalloc(&d_meshDistance, inboxSize * sizeof(T)),  "d_meshDistance");

    // ===== Copy input data =====
    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice), "cp keys");
    checkCudaError(cudaMemcpy(d_x,   x.data(),   numParticles * sizeof(T), cudaMemcpyHostToDevice), "cp x");
    checkCudaError(cudaMemcpy(d_y,   y.data(),   numParticles * sizeof(T), cudaMemcpyHostToDevice), "cp y");
    checkCudaError(cudaMemcpy(d_z,   z.data(),   numParticles * sizeof(T), cudaMemcpyHostToDevice), "cp z");
    checkCudaError(cudaMemcpy(d_vx,  vx.data(),  numParticles * sizeof(T), cudaMemcpyHostToDevice), "cp vx");
    checkCudaError(cudaMemcpy(d_vy,  vy.data(),  numParticles * sizeof(T), cudaMemcpyHostToDevice), "cp vy");
    checkCudaError(cudaMemcpy(d_vz,  vz.data(),  numParticles * sizeof(T), cudaMemcpyHostToDevice), "cp vz");

    std::array<int, 3> inboxSizeArr = {mesh.inbox_.size[0], mesh.inbox_.size[1], mesh.inbox_.size[2]};
    checkCudaError(cudaMemcpy(d_inboxSize, inboxSizeArr.data(), 3 * sizeof(int), cudaMemcpyHostToDevice), "cp inboxSize");
    checkCudaError(cudaMemcpy(d_procGrid, mesh.proc_grid_.data(), 3 * sizeof(int), cudaMemcpyHostToDevice), "cp procGrid");

    // Initialise accumulators to zero; meshDistance to infinity
    checkCudaError(cudaMemset(d_velXSum,   0, inboxSize * sizeof(T)),   "memset velXSum");
    checkCudaError(cudaMemset(d_velYSum,   0, inboxSize * sizeof(T)),   "memset velYSum");
    checkCudaError(cudaMemset(d_velZSum,   0, inboxSize * sizeof(T)),   "memset velZSum");
    checkCudaError(cudaMemset(d_cellCount, 0, inboxSize * sizeof(int)), "memset cellCount");
    checkCudaError(cudaMemset(d_meshVelX,  0, inboxSize * sizeof(T)),   "memset meshVelX");
    checkCudaError(cudaMemset(d_meshVelY,  0, inboxSize * sizeof(T)),   "memset meshVelY");
    checkCudaError(cudaMemset(d_meshVelZ,  0, inboxSize * sizeof(T)),   "memset meshVelZ");
    checkCudaError(cudaMemcpy(d_meshDistance, mesh.distance_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice),
                   "cp distance (inf)");

    int zeroVal = 0;
    checkCudaError(cudaMemcpy(d_localCount,  &zeroVal, sizeof(int), cudaMemcpyHostToDevice), "init localCount");
    checkCudaError(cudaMemcpy(d_remoteCount, &zeroVal, sizeof(int), cudaMemcpyHostToDevice), "init remoteCount");

    // ===== Classify particles =====
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    classifyParticlesCellAvgKernel<T><<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_vx, d_vy, d_vz, numParticles, mesh.gridDim_,
        d_inboxSize, d_procGrid, mesh.rank_,
        d_localIndices, d_localVx, d_localVy, d_localVz, d_localCount,
        d_remoteRanks, d_remoteIndices, d_remoteVx, d_remoteVy, d_remoteVz, d_remoteCount);
    checkCudaError(cudaDeviceSynchronize(), "classify cell_avg kernel");

    int h_localCount, h_remoteCount;
    checkCudaError(cudaMemcpy(&h_localCount,  d_localCount,  sizeof(int), cudaMemcpyDeviceToHost), "cp localCount");
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost), "cp remoteCount");

    // ===== Accumulate local particles =====
    if (h_localCount > 0)
    {
        int lb = (h_localCount + threadsPerBlock - 1) / threadsPerBlock;
        accumulateCellAvgKernel<T><<<lb, threadsPerBlock>>>(
            d_localIndices, d_localVx, d_localVy, d_localVz, h_localCount,
            d_velXSum, d_velYSum, d_velZSum, d_cellCount);
        checkCudaError(cudaDeviceSynchronize(), "accumulate local cell_avg");
    }

    // ===== Copy remote particles to host, organise by rank, MPI exchange =====
    std::vector<int>      h_remoteRanks(h_remoteCount);
    std::vector<uint64_t> h_remoteIndices(h_remoteCount);
    std::vector<T>        h_remoteVx(h_remoteCount), h_remoteVy(h_remoteCount), h_remoteVz(h_remoteCount);

    if (h_remoteCount > 0)
    {
        checkCudaError(cudaMemcpy(h_remoteRanks.data(),   d_remoteRanks,    h_remoteCount * sizeof(int),      cudaMemcpyDeviceToHost), "cp remoteRanks");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices,  h_remoteCount * sizeof(uint64_t), cudaMemcpyDeviceToHost), "cp remoteIndices");
        checkCudaError(cudaMemcpy(h_remoteVx.data(),      d_remoteVx,       h_remoteCount * sizeof(T),        cudaMemcpyDeviceToHost), "cp remoteVx");
        checkCudaError(cudaMemcpy(h_remoteVy.data(),      d_remoteVy,       h_remoteCount * sizeof(T),        cudaMemcpyDeviceToHost), "cp remoteVy");
        checkCudaError(cudaMemcpy(h_remoteVz.data(),      d_remoteVz,       h_remoteCount * sizeof(T),        cudaMemcpyDeviceToHost), "cp remoteVz");
    }

    std::fill(mesh.send_count.begin(), mesh.send_count.end(), 0);
    std::fill(mesh.send_disp.begin(),  mesh.send_disp.end(),  0);
    for (int i = 0; i < h_remoteCount; i++)
    {
        int r = h_remoteRanks[i];
        mesh.send_count[r]++;
        mesh.vdataSenderCellAvg[r].send_index.push_back(h_remoteIndices[i]);
        mesh.vdataSenderCellAvg[r].send_vx.push_back(h_remoteVx[i]);
        mesh.vdataSenderCellAvg[r].send_vy.push_back(h_remoteVy[i]);
        mesh.vdataSenderCellAvg[r].send_vz.push_back(h_remoteVz[i]);
    }

    MPI_Alltoall(mesh.send_count.data(), 1, MpiType<int>{}, mesh.recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.send_disp[i + 1] = mesh.send_disp[i] + mesh.send_count[i];
        mesh.recv_disp[i + 1] = mesh.recv_disp[i] + mesh.recv_count[i];
    }

    mesh.send_index_cavg.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_vx_cavg.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_vy_cavg.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_vz_cavg.resize(mesh.send_disp[mesh.numRanks_]);

    for (int i = 0; i < mesh.numRanks_; i++)
        for (int j = mesh.send_disp[i]; j < mesh.send_disp[i + 1]; j++)
        {
            int local = j - mesh.send_disp[i];
            mesh.send_index_cavg[j] = mesh.vdataSenderCellAvg[i].send_index[local];
            mesh.send_vx_cavg[j]    = mesh.vdataSenderCellAvg[i].send_vx[local];
            mesh.send_vy_cavg[j]    = mesh.vdataSenderCellAvg[i].send_vy[local];
            mesh.send_vz_cavg[j]    = mesh.vdataSenderCellAvg[i].send_vz[local];
        }

    mesh.recv_index_cavg.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_vx_cavg.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_vy_cavg.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_vz_cavg.resize(mesh.recv_disp[mesh.numRanks_]);

    MPI_Alltoallv(mesh.send_index_cavg.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<uint64_t>{},
                  mesh.recv_index_cavg.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_vx_cavg.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_vx_cavg.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_vy_cavg.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_vy_cavg.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_vz_cavg.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                  mesh.recv_vz_cavg.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);

    // ===== Accumulate received particles on device =====
    int recvTotal = mesh.recv_disp[mesh.numRanks_];
    if (recvTotal > 0)
    {
        uint64_t* d_recvIndices;
        T *       d_recvVx, *d_recvVy, *d_recvVz;
        checkCudaError(cudaMalloc(&d_recvIndices, recvTotal * sizeof(uint64_t)), "d_recvIndices cavg");
        checkCudaError(cudaMalloc(&d_recvVx,      recvTotal * sizeof(T)),        "d_recvVx cavg");
        checkCudaError(cudaMalloc(&d_recvVy,      recvTotal * sizeof(T)),        "d_recvVy cavg");
        checkCudaError(cudaMalloc(&d_recvVz,      recvTotal * sizeof(T)),        "d_recvVz cavg");

        checkCudaError(cudaMemcpy(d_recvIndices, mesh.recv_index_cavg.data(), recvTotal * sizeof(uint64_t), cudaMemcpyHostToDevice), "cp recv idx cavg");
        checkCudaError(cudaMemcpy(d_recvVx,      mesh.recv_vx_cavg.data(),    recvTotal * sizeof(T),        cudaMemcpyHostToDevice), "cp recv vx cavg");
        checkCudaError(cudaMemcpy(d_recvVy,      mesh.recv_vy_cavg.data(),    recvTotal * sizeof(T),        cudaMemcpyHostToDevice), "cp recv vy cavg");
        checkCudaError(cudaMemcpy(d_recvVz,      mesh.recv_vz_cavg.data(),    recvTotal * sizeof(T),        cudaMemcpyHostToDevice), "cp recv vz cavg");

        int rb = (recvTotal + threadsPerBlock - 1) / threadsPerBlock;
        accumulateCellAvgKernel<T><<<rb, threadsPerBlock>>>(
            d_recvIndices, d_recvVx, d_recvVy, d_recvVz, recvTotal,
            d_velXSum, d_velYSum, d_velZSum, d_cellCount);
        checkCudaError(cudaDeviceSynchronize(), "accumulate recv cell_avg");

        cudaFree(d_recvIndices);
        cudaFree(d_recvVx);
        cudaFree(d_recvVy);
        cudaFree(d_recvVz);
    }

    // ===== Normalise =====
    int nb = (inboxSize + threadsPerBlock - 1) / threadsPerBlock;
    normalizeCellAvgKernel<T><<<nb, threadsPerBlock>>>(
        d_velXSum, d_velYSum, d_velZSum, d_cellCount,
        d_meshVelX, d_meshVelY, d_meshVelZ, d_meshDistance, inboxSize);
    checkCudaError(cudaDeviceSynchronize(), "normalize cell_avg");

    // Keep velocity and distance buffers on device
    if (mesh.d_velX_) cudaFree(mesh.d_velX_);
    if (mesh.d_velY_) cudaFree(mesh.d_velY_);
    if (mesh.d_velZ_) cudaFree(mesh.d_velZ_);
    if (mesh.d_distance_) cudaFree(mesh.d_distance_);

    mesh.d_velX_       = d_meshVelX;
    mesh.d_velY_       = d_meshVelY;
    mesh.d_velZ_       = d_meshVelZ;
    mesh.d_distance_   = d_meshDistance;
    mesh.gpuDataValid_ = true;

    // ===== Free temporaries =====
    cudaFree(d_keys);   cudaFree(d_x);    cudaFree(d_y);    cudaFree(d_z);
    cudaFree(d_vx);     cudaFree(d_vy);   cudaFree(d_vz);
    cudaFree(d_procGrid);
    cudaFree(d_localIndices); cudaFree(d_localVx);  cudaFree(d_localVy);  cudaFree(d_localVz);
    cudaFree(d_localCount);
    cudaFree(d_remoteRanks); cudaFree(d_remoteIndices);
    cudaFree(d_remoteVx); cudaFree(d_remoteVy); cudaFree(d_remoteVz); cudaFree(d_remoteCount);
    cudaFree(d_velXSum); cudaFree(d_velYSum); cudaFree(d_velZSum);
    cudaFree(d_cellCount);
    // Note: d_meshVelX/Y/Z and d_meshDistance NOT freed — stored in mesh

    // Clear send buffers
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSenderCellAvg[i].send_index.clear();
        mesh.vdataSenderCellAvg[i].send_vx.clear();
        mesh.vdataSenderCellAvg[i].send_vy.clear();
        mesh.vdataSenderCellAvg[i].send_vz.clear();
    }

    // ========== GPU Extrapolation ==========
    T* d_srcVelX, *d_srcVelY, *d_srcVelZ;
    checkCudaError(cudaMalloc(&d_srcVelX, inboxSize * sizeof(T)), "d_srcVelX extrap cell_avg");
    checkCudaError(cudaMalloc(&d_srcVelY, inboxSize * sizeof(T)), "d_srcVelY extrap cell_avg");
    checkCudaError(cudaMalloc(&d_srcVelZ, inboxSize * sizeof(T)), "d_srcVelZ extrap cell_avg");
    checkCudaError(cudaMemcpy(d_srcVelX, mesh.d_velX_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velX cell_avg");
    checkCudaError(cudaMemcpy(d_srcVelY, mesh.d_velY_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velY cell_avg");
    checkCudaError(cudaMemcpy(d_srcVelZ, mesh.d_velZ_, inboxSize * sizeof(T), cudaMemcpyDeviceToDevice), "snapshot velZ cell_avg");

    launchExtrapolateEmptyCellsKernel<T>(d_srcVelX, d_srcVelY, d_srcVelZ,
                                          mesh.d_velX_, mesh.d_velY_, mesh.d_velZ_,
                                          mesh.d_distance_, d_inboxSize,
                                          mesh.inbox_.size[0], mesh.inbox_.size[1], mesh.inbox_.size[2]);
    checkCudaError(cudaDeviceSynchronize(), "GPU extrapolate cell_avg");

    cudaFree(d_srcVelX);
    cudaFree(d_srcVelY);
    cudaFree(d_srcVelZ);
    cudaFree(d_inboxSize); // freed here after extrapolation kernel

    std::cout << "rank = " << mesh.rank_ << " rasterize (CUDA cell_avg) done!" << std::endl;
}

// GPU kernel to extrapolate empty mesh cells from filled neighbors.
// Reads from immutable src snapshot, writes to vel output — no races.
template<typename T>
__global__ void extrapolateEmptyCellsKernel(const T* __restrict__ srcVelX,
                                             const T* __restrict__ srcVelY,
                                             const T* __restrict__ srcVelZ,
                                             T* velX, T* velY, T* velZ,
                                             const T* distance, const int* inboxSize)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // fast (x)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // mid  (y)
    int i = blockIdx.z * blockDim.z + threadIdx.z; // slow (z)

    if (k >= inboxSize[0] || j >= inboxSize[1] || i >= inboxSize[2]) return;

    uint64_t idx =
        (uint64_t)k + (uint64_t)j * inboxSize[0] + (uint64_t)i * inboxSize[0] * inboxSize[1];

    if (!isinf(distance[idx])) return; // already filled — nothing to do

    T   velXSum = 0, velYSum = 0, velZSum = 0;
    int count   = 0;
    for (int ni = i - 1; ni <= i + 1; ni++)
    {
        if (ni < 0 || ni >= inboxSize[2]) continue;
        for (int nj = j - 1; nj <= j + 1; nj++)
        {
            if (nj < 0 || nj >= inboxSize[1]) continue;
            for (int nk = k - 1; nk <= k + 1; nk++)
            {
                if (nk < 0 || nk >= inboxSize[0]) continue;
                uint64_t nidx =
                    (uint64_t)nk + (uint64_t)nj * inboxSize[0] + (uint64_t)ni * inboxSize[0] * inboxSize[1];
                if (!isinf(distance[nidx]))
                {
                    velXSum += srcVelX[nidx];
                    velYSum += srcVelY[nidx];
                    velZSum += srcVelZ[nidx];
                    count++;
                }
            }
        }
    }
    if (count > 0)
    {
        velX[idx] = velXSum / count;
        velY[idx] = velYSum / count;
        velZ[idx] = velZSum / count;
    }
}

template<typename T>
void launchExtrapolateEmptyCellsKernel(const T* srcVelX, const T* srcVelY, const T* srcVelZ,
                                        T* velX, T* velY, T* velZ,
                                        const T* distance, const int* inboxSize_d,
                                        int sx, int sy, int sz)
{
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((sx + 7) / 8, (sy + 7) / 8, (sz + 7) / 8);
    extrapolateEmptyCellsKernel<T><<<gridSize, blockSize>>>(
        srcVelX, srcVelY, srcVelZ, velX, velY, velZ, distance, inboxSize_d);
}

// CUDA kernel to compute power spectrum from FFT output (squared magnitude)
template<typename T>
__global__ void computePowerSpectrumKernel(std::complex<T>* fftOutput, T* powerSpectrum, uint64_t size, T meshSize)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute magnitude manually (std::complex methods not available in device code)
    // std::complex<T> is layout-compatible with T[2], so we can access it directly
    T* complexData = reinterpret_cast<T*>(&fftOutput[idx]);
    T real = complexData[0];
    T imag = complexData[1];
    T magnitude = sqrt(real * real + imag * imag) / meshSize;
    powerSpectrum[idx] = magnitude * magnitude;
}

// CUDA kernel to compute freqVelo = velX + velY + velZ
template<typename T>
__global__ void computeFreqVeloKernel(T* velX, T* velY, T* velZ, T* freqVelo, uint64_t size)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    freqVelo[idx] = velX[idx] + velY[idx] + velZ[idx];
}

// CUDA kernel for spherical averaging
template<typename T>
__global__ void sphericalAveragingKernel(T* freqVelo, T* k_values, T* k_1d, T* ps_rad, int* count,
                                         int* inboxSize, int* inboxLow, int gridDim, int numShells)
{
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= inboxSize[2] || j >= inboxSize[1] || k >= inboxSize[0]) return;
    
    uint64_t freq_index = k + j * inboxSize[0] + i * inboxSize[0] * inboxSize[1];
    
    // Calculate the k indices with respect to the global mesh
    uint64_t k_index_i = i + inboxLow[2];
    uint64_t k_index_j = j + inboxLow[1];
    uint64_t k_index_k = k + inboxLow[0];
    
    T kdist = sqrt(k_values[k_index_i] * k_values[k_index_i] +
                   k_values[k_index_j] * k_values[k_index_j] +
                   k_values[k_index_k] * k_values[k_index_k]);
    
    // Find closest k_1d bin
    T minDiff = fabs(k_1d[0] - kdist);
    uint64_t k_index = 0;
    for (int kind = 1; kind < gridDim; kind++)
    {
        T diff = fabs(k_1d[kind] - kdist);
        if (diff < minDiff)
        {
            minDiff = diff;
            k_index = kind;
        }
    }
    
    // Clamp k_index to valid range
    if (k_index >= static_cast<uint64_t>(numShells)) k_index = numShells - 1;
    
    // Atomic accumulation
    atomicAdd(&ps_rad[k_index], freqVelo[freq_index]);
    atomicAdd(&count[k_index], 1);
}

// Kernel launcher functions
template<typename T>
void launchComputePowerSpectrumKernel(std::complex<T>* fftOutput, T* powerSpectrum, uint64_t size, T meshSize, int blocks, int threads)
{
    computePowerSpectrumKernel<T><<<blocks, threads>>>(fftOutput, powerSpectrum, size, meshSize);
}

template<typename T>
void launchComputeFreqVeloKernel(T* velX, T* velY, T* velZ, T* freqVelo, uint64_t size, int blocks, int threads)
{
    computeFreqVeloKernel<T><<<blocks, threads>>>(velX, velY, velZ, freqVelo, size);
}

template<typename T>
void launchSphericalAveragingKernel(T* freqVelo, T* k_values, T* k_1d, T* ps_rad, int* count,
                                     int* inboxSize, int* inboxLow, int gridDim, int numShells,
                                     dim3 gridSize, dim3 blockSize)
{
    sphericalAveragingKernel<T><<<gridSize, blockSize>>>(freqVelo, k_values, k_1d, ps_rad, count,
                                                          inboxSize, inboxLow, gridDim, numShells);
}

// Explicit template instantiations
template void launchComputePowerSpectrumKernel<double>(std::complex<double>*, double*, uint64_t, double, int, int);
template void launchComputeFreqVeloKernel<double>(double*, double*, double*, double*, uint64_t, int, int);
template void launchSphericalAveragingKernel<double>(double*, double*, double*, double*, int*,
                                                      int*, int*, int, int, dim3, dim3);

// Explicit template instantiation for double
template void launchExtrapolateEmptyCellsKernel<double>(const double*, const double*, const double*,
                                                         double*, double*, double*,
                                                         const double*, const int*, int, int, int);

template void rasterize_particles_to_mesh_cuda<double>(Mesh<double>&, std::vector<KeyType>, std::vector<double>,
                                                        std::vector<double>, std::vector<double>, std::vector<double>,
                                                        std::vector<double>, std::vector<double>, int);
template void rasterize_particles_to_mesh_sph_cuda<double>(Mesh<double>&, std::vector<KeyType>, std::vector<double>,
                                                            std::vector<double>, std::vector<double>, std::vector<double>,
                                                            std::vector<double>, std::vector<double>, std::vector<double>,
                                                            int);
template void rasterize_particles_to_mesh_cell_avg_cuda<double>(Mesh<double>&, std::vector<KeyType>, std::vector<double>,
                                                                 std::vector<double>, std::vector<double>, std::vector<double>,
                                                                 std::vector<double>, std::vector<double>, int);

#ifdef USE_NVSHMEM
template void rasterize_particles_to_mesh_nvshmem<double>(Mesh<double>&, std::vector<KeyType>, std::vector<double>,
                                                          std::vector<double>, std::vector<double>, std::vector<double>,
                                                          std::vector<double>, std::vector<double>, int);
#endif