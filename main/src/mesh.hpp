#pragma once

#include <vector>
#include <limits>
#include <numbers>
#include <iostream>
#include <cstdlib>
#include "heffte.h"
#include "cstone/domain/domain.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <complex>
#endif

using KeyType = uint64_t;

#ifdef USE_CUDA
// Forward declarations for CUDA kernel launchers
template<typename T>
void launchComputePowerSpectrumKernel(std::complex<T>* fftOutput, T* powerSpectrum, uint64_t size, T meshSize, int blocks, int threads);

template<typename T>
void launchComputeFreqVeloKernel(T* velX, T* velY, T* velZ, T* freqVelo, uint64_t size, int blocks, int threads);

template<typename T>
void launchSphericalAveragingKernel(T* freqVelo, T* k_values, T* k_1d, T* ps_rad, int* count,
                                     int* inboxSize, int* inboxLow, int gridDim, int numShells,
                                     dim3 gridSize, dim3 blockSize);

template<typename T>
void launchExtrapolateEmptyCellsKernel(const T* srcVelX, const T* srcVelY, const T* srcVelZ,
                                        T* velX, T* velY, T* velZ,
                                        const T* distance, const int* inboxSize_d,
                                        int sx, int sy, int sz);
#endif

struct DataSender
{
    // vectors to send to each rank in all_to_allv
    std::vector<uint64_t> send_index;
    std::vector<double>   send_distance;
    std::vector<double>   send_vx;
    std::vector<double>   send_vy;
    std::vector<double>   send_vz;
};

struct DataSenderSPH
{
    // vectors to send to each rank in all_to_allv for SPH interpolation
    std::vector<uint64_t> send_index;
    std::vector<double>   send_weight;
    std::vector<double>   send_weighted_vx;
    std::vector<double>   send_weighted_vy;
    std::vector<double>   send_weighted_vz;
};

struct DataSenderCellAvg
{
    // vectors to send to each rank in all_to_allv for cell-average interpolation
    std::vector<uint64_t> send_index;
    std::vector<double>   send_vx;
    std::vector<double>   send_vy;
    std::vector<double>   send_vz;
};

template<typename T>
class Mesh
{
public:
    int                rank_;
    int                numRanks_;
    int                gridDim_; // specifically integer because heffte library uses int
    int                numShells_;
    T                  Lmin_;
    T                  Lmax_;
    bool               usePencils_ = false; // heFFTe decomposition: false=slabs, true=pencils
    std::array<int, 3> proc_grid_;

    heffte::box3d<> inbox_;
    // coordinate centers in the mesh
    std::vector<T> x_;
    // std::vector<T>  y_;
    // std::vector<T>  z_;
    // corresponding velocities in the mesh
    std::vector<T> velX_;
    std::vector<T> velY_;
    std::vector<T> velZ_;
    // particle's distance to mesh point
    std::vector<T> distance_;
    
#ifdef USE_CUDA
    // Device pointers for velocity arrays (kept on GPU after CUDA rasterization)
    T* d_velX_ = nullptr;
    T* d_velY_ = nullptr;
    T* d_velZ_ = nullptr;
    T* d_freqVelo_ = nullptr; // Device pointer for freqVelo (velX + velY + velZ)
    T* d_distance_ = nullptr; // Device pointer for distance/fill sentinel array
    bool gpuDataValid_ = false; // Flag to indicate if GPU data is valid

    // Cleanup function to free GPU memory
    ~Mesh()
    {
        if (d_velX_) cudaFree(d_velX_);
        if (d_velY_) cudaFree(d_velY_);
        if (d_velZ_) cudaFree(d_velZ_);
        if (d_freqVelo_) cudaFree(d_freqVelo_);
        if (d_distance_) cudaFree(d_distance_);
    }
#endif
    std::vector<T> power_spectrum_;

    // communication counters
    std::vector<int> send_disp;  //(numRanks_+1, 0);
    std::vector<int> send_count; //(numRanks_, 0);

    std::vector<int> recv_disp;  //(numRanks_+1, 0);
    std::vector<int> recv_count; //(numRanks_, 0);

    std::vector<DataSender> vdataSender;

    // flattened send buffers assembled from vdataSender before all_to_allv
    std::vector<uint64_t> send_index;
    std::vector<T>        send_distance;
    std::vector<T>        send_vx;
    std::vector<T>        send_vy;
    std::vector<T>        send_vz;

    // vectors to receive from each rank in all_to_allv
    std::vector<uint64_t> recv_index;
    std::vector<T>        recv_distance;
    std::vector<T>        recv_vx;
    std::vector<T>        recv_vy;
    std::vector<T>        recv_vz;

    // SPH interpolation data structures
    std::vector<DataSenderSPH> vdataSenderSPH;
    std::vector<T>             weightSum_;      // sum of weights for each cell
    std::vector<T>             weightedVelX_;   // weighted velocity sum for each cell
    std::vector<T>             weightedVelY_;   // weighted velocity sum for each cell
    std::vector<T>             weightedVelZ_;   // weighted velocity sum for each cell
    std::vector<uint64_t>      send_index_sph;
    std::vector<T>             send_weight;
    std::vector<T>             send_weighted_vx;
    std::vector<T>             send_weighted_vy;
    std::vector<T>             send_weighted_vz;
    std::vector<uint64_t>      recv_index_sph;
    std::vector<T>             recv_weight;
    std::vector<T>             recv_weighted_vx;
    std::vector<T>             recv_weighted_vy;
    std::vector<T>             recv_weighted_vz;

    // Cell-average interpolation data structures
    std::vector<DataSenderCellAvg> vdataSenderCellAvg;
    std::vector<T>                 cellAvgVelX_;  // velocity sum per cell
    std::vector<T>                 cellAvgVelY_;
    std::vector<T>                 cellAvgVelZ_;
    std::vector<int>               cellCount_;    // particle count per cell
    std::vector<uint64_t>          send_index_cavg;
    std::vector<T>                 send_vx_cavg;
    std::vector<T>                 send_vy_cavg;
    std::vector<T>                 send_vz_cavg;
    std::vector<uint64_t>          recv_index_cavg;
    std::vector<T>                 recv_vx_cavg;
    std::vector<T>                 recv_vy_cavg;
    std::vector<T>                 recv_vz_cavg;

    // sim box -0.5 to 0.5 by default
    Mesh(int rank, int numRanks, int gridDim, int numShells)
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , numShells_(numShells)
        , Lmin_(-0.5)
        , Lmax_(0.5)
        , inbox_(initInbox())
    {
        uint64_t inboxSize = static_cast<uint64_t>(inbox_.size[0]) * static_cast<uint64_t>(inbox_.size[1]) *
                             static_cast<uint64_t>(inbox_.size[2]);
        std::cout << "rank = " << rank << " griddim = " << gridDim << " inboxSize = " << inboxSize << std::endl;
        std::cout << "rank = " << rank << " inbox low = " << inbox_.low[0] << " " << inbox_.low[1] << " "
                  << inbox_.low[2] << std::endl;
        std::cout << "rank = " << rank << " inbox high = " << inbox_.high[0] << " " << inbox_.high[1] << " "
                  << inbox_.high[2] << std::endl;
        velX_.resize(inboxSize);
        velY_.resize(inboxSize);
        velZ_.resize(inboxSize);
        x_.resize(inbox_.size[0]);
        // y_.resize(inbox_.size[1]);
        // z_.resize(inbox_.size[2]);
        distance_.resize(inboxSize);
        power_spectrum_.resize(numShells);

        // SPH interpolation arrays
        weightSum_.resize(inboxSize, 0.0);
        weightedVelX_.resize(inboxSize, 0.0);
        weightedVelY_.resize(inboxSize, 0.0);
        weightedVelZ_.resize(inboxSize, 0.0);

        // Cell-average interpolation arrays
        cellAvgVelX_.resize(inboxSize, 0.0);
        cellAvgVelY_.resize(inboxSize, 0.0);
        cellAvgVelZ_.resize(inboxSize, 0.0);
        cellCount_.resize(inboxSize, 0);

        resize_comm_size(numRanks);
        std::fill(distance_.begin(), distance_.end(), std::numeric_limits<T>::infinity());

        // populate the x_, y_, z_ vectors with the center coordinates of the mesh cells
        setCoordinates(Lmin_, Lmax_);
    }

    void rasterize_particles_to_mesh(const std::vector<KeyType>& keys, const std::vector<T>& x, const std::vector<T>& y,
                                     const std::vector<T>& z, const std::vector<T>& vx, const std::vector<T>& vy,
                                     const std::vector<T>& vz, int powerDim)
    {
        std::cout << "rank" << rank_ << " rasterize start " << powerDim << std::endl;
        std::cout << "rank" << rank_ << " keys between " << *keys.begin() << " - " << keys.back() << std::endl;

        int particleIndex = 0;
        // iterate over keys vector
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            auto crd    = calculateKeyIndices(*it, gridDim_);
            int  indexi = std::get<0>(crd);
            int  indexj = std::get<1>(crd);
            int  indexk = std::get<2>(crd);

            assert(indexi < gridDim_);
            assert(indexj < gridDim_);
            assert(indexk < gridDim_);

            double distance =
                calculateDistance(x[particleIndex], y[particleIndex], z[particleIndex], indexi, indexj, indexk);
            assignVelocityByMeshCoord(indexi, indexj, indexk, distance, vx[particleIndex], vy[particleIndex],
                                      vz[particleIndex]);
            particleIndex++;
        }

        std::cout << "rank = " << rank_ << " particleIndex = " << particleIndex << std::endl;
        for (int i = 0; i < numRanks_; i++)
            std::cout << "rank = " << rank_ << " send_count = " << send_count[i] << std::endl;

        MPI_Alltoall(send_count.data(), 1, MpiType<int>{}, recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

        for (int i = 0; i < numRanks_; i++)
        {
            send_disp[i + 1] = send_disp[i] + send_count[i];
            recv_disp[i + 1] = recv_disp[i] + recv_count[i];
        }

        // prepare send buffers
        send_index.resize(send_disp[numRanks_]);
        send_distance.resize(send_disp[numRanks_]);
        send_vx.resize(send_disp[numRanks_]);
        send_vy.resize(send_disp[numRanks_]);
        send_vz.resize(send_disp[numRanks_]);
        std::cout << "rank = " << rank_ << " buffers allocated" << std::endl;

        for (int i = 0; i < numRanks_; i++)
        {
            for (int j = send_disp[i]; j < send_disp[i + 1]; j++)
            {
                send_index[j]    = vdataSender[i].send_index[j - send_disp[i]];
                send_distance[j] = vdataSender[i].send_distance[j - send_disp[i]];
                send_vx[j]       = vdataSender[i].send_vx[j - send_disp[i]];
                send_vy[j]       = vdataSender[i].send_vy[j - send_disp[i]];
                send_vz[j]       = vdataSender[i].send_vz[j - send_disp[i]];
            }
        }
        std::cout << "rank = " << rank_ << " buffers transformed" << std::endl;

        // prepare receive buffers
        recv_index.resize(recv_disp[numRanks_]);
        recv_distance.resize(recv_disp[numRanks_]);
        recv_vx.resize(recv_disp[numRanks_]);
        recv_vy.resize(recv_disp[numRanks_]);
        recv_vz.resize(recv_disp[numRanks_]);

        MPI_Alltoallv(send_index.data(), send_count.data(), send_disp.data(), MpiType<uint64_t>{}, recv_index.data(),
                      recv_count.data(), recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_distance.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_distance.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vx.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_vx.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vy.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_vy.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vz.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_vz.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        std::cout << "rank = " << rank_ << " alltoallv done!" << std::endl;

        for (int i = 0; i < recv_disp[numRanks_]; i++)
        {
            uint64_t index = recv_index[i];
            if (recv_distance[i] < distance_[index])
            {
                velX_[index]     = recv_vx[i];
                velY_[index]     = recv_vy[i];
                velZ_[index]     = recv_vz[i];
                distance_[index] = recv_distance[i];
            }
        }

        // clear the vectors
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSender[i].send_index.clear();
            vdataSender[i].send_distance.clear();
            vdataSender[i].send_vx.clear();
            vdataSender[i].send_vy.clear();
            vdataSender[i].send_vz.clear();
        }

        // extrapolate mesh cells which doesn't have any particles assigned
        extrapolateEmptyCellsFromNeighbors();
    }

    // SPH interpolation rasterization function
    void rasterize_particles_to_mesh_sph(const std::vector<KeyType>& keys, const std::vector<T>& x,
                                         const std::vector<T>& y, const std::vector<T>& z, const std::vector<T>& vx,
                                         const std::vector<T>& vy, const std::vector<T>& vz, const std::vector<T>& h,
                                         int powerDim)
    {
        std::cout << "rank" << rank_ << " rasterize start (SPH) " << powerDim << std::endl;
        std::cout << "rank" << rank_ << " keys between " << *keys.begin() << " - " << keys.back() << std::endl;

        // Reset SPH accumulation arrays
        uint64_t inboxSize = static_cast<uint64_t>(inbox_.size[0]) * static_cast<uint64_t>(inbox_.size[1]) *
                             static_cast<uint64_t>(inbox_.size[2]);
        std::fill(weightSum_.begin(), weightSum_.end(), 0.0);
        std::fill(weightedVelX_.begin(), weightedVelX_.end(), 0.0);
        std::fill(weightedVelY_.begin(), weightedVelY_.end(), 0.0);
        std::fill(weightedVelZ_.begin(), weightedVelZ_.end(), 0.0);
        std::fill(send_count.begin(), send_count.end(), 0);

        // Clear SPH sender data
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSenderSPH[i].send_index.clear();
            vdataSenderSPH[i].send_weight.clear();
            vdataSenderSPH[i].send_weighted_vx.clear();
            vdataSenderSPH[i].send_weighted_vy.clear();
            vdataSenderSPH[i].send_weighted_vz.clear();
        }

        T deltaMesh = (Lmax_ - Lmin_) / gridDim_;

        int particleIndex = 0;
        // iterate over keys vector
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            T px = x[particleIndex];
            T py = y[particleIndex];
            T pz = z[particleIndex];
            T pvx = vx[particleIndex];
            T pvy = vy[particleIndex];
            T pvz = vz[particleIndex];
            T ph = h[particleIndex];

            // Use effective smoothing length: cap at cell size to reduce excessive smoothing
            // This preserves more high-k power while still providing SPH interpolation
            T h_eff = std::min(ph, deltaMesh);
            
            // Calculate the search radius (2 * effective smoothing length)
            T searchRadius = 2.0 * h_eff;
            T searchRadiusSq = searchRadius * searchRadius;

            // Find the range of grid cells that could be within 2*h
            // Convert particle position to grid indices
            T cellSize = deltaMesh;
            int minI = static_cast<int>((px - searchRadius - Lmin_) / cellSize);
            int maxI = static_cast<int>((px + searchRadius - Lmin_) / cellSize) + 1;
            int minJ = static_cast<int>((py - searchRadius - Lmin_) / cellSize);
            int maxJ = static_cast<int>((py + searchRadius - Lmin_) / cellSize) + 1;
            int minK = static_cast<int>((pz - searchRadius - Lmin_) / cellSize);
            int maxK = static_cast<int>((pz + searchRadius - Lmin_) / cellSize) + 1;

            // Clamp to valid grid range
            minI = std::max(0, minI);
            maxI = std::min(gridDim_, maxI);
            minJ = std::max(0, minJ);
            maxJ = std::min(gridDim_, maxJ);
            minK = std::max(0, minK);
            maxK = std::min(gridDim_, maxK);

            // Iterate over potential cells
            for (int i = minI; i < maxI; i++)
            {
                for (int j = minJ; j < maxJ; j++)
                {
                    for (int k = minK; k < maxK; k++)
                    {
                        // Calculate distance from particle to cell center
                        T cellX = getCellCenterX(i);
                        T cellY = getCellCenterY(j);
                        T cellZ = getCellCenterZ(k);

                        T dx = px - cellX;
                        T dy = py - cellY;
                        T dz = pz - cellZ;
                        T distSq = dx * dx + dy * dy + dz * dz;

                        // Check if within search radius
                        if (distSq < searchRadiusSq)
                        {
                            T dist = std::sqrt(distSq);
                            // Use effective smoothing length for kernel evaluation
                            T weight = sphKernel(dist, h_eff);

                            if (weight > 0.0)
                            {
                                // Calculate weighted velocity contribution
                                T weightedVx = pvx * weight;
                                T weightedVy = pvy * weight;
                                T weightedVz = pvz * weight;

                                assignVelocityByMeshCoordSPH(i, j, k, weight, weightedVx, weightedVy, weightedVz);
                            }
                        }
                    }
                }
            }

            particleIndex++;
        }

        std::cout << "rank = " << rank_ << " particleIndex = " << particleIndex << std::endl;
        for (int i = 0; i < numRanks_; i++)
            std::cout << "rank = " << rank_ << " send_count = " << send_count[i] << std::endl;

        // MPI communication for SPH data
        MPI_Alltoall(send_count.data(), 1, MpiType<int>{}, recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

        for (int i = 0; i < numRanks_; i++)
        {
            send_disp[i + 1] = send_disp[i] + send_count[i];
            recv_disp[i + 1] = recv_disp[i] + recv_count[i];
        }

        // Prepare send buffers for SPH
        send_index_sph.resize(send_disp[numRanks_]);
        send_weight.resize(send_disp[numRanks_]);
        send_weighted_vx.resize(send_disp[numRanks_]);
        send_weighted_vy.resize(send_disp[numRanks_]);
        send_weighted_vz.resize(send_disp[numRanks_]);
        std::cout << "rank = " << rank_ << " buffers allocated" << std::endl;

        for (int i = 0; i < numRanks_; i++)
        {
            for (int j = send_disp[i]; j < send_disp[i + 1]; j++)
            {
                int localIdx = j - send_disp[i];
                send_index_sph[j] = vdataSenderSPH[i].send_index[localIdx];
                send_weight[j] = vdataSenderSPH[i].send_weight[localIdx];
                send_weighted_vx[j] = vdataSenderSPH[i].send_weighted_vx[localIdx];
                send_weighted_vy[j] = vdataSenderSPH[i].send_weighted_vy[localIdx];
                send_weighted_vz[j] = vdataSenderSPH[i].send_weighted_vz[localIdx];
            }
        }
        std::cout << "rank = " << rank_ << " buffers transformed" << std::endl;

        // Prepare receive buffers
        recv_index_sph.resize(recv_disp[numRanks_]);
        recv_weight.resize(recv_disp[numRanks_]);
        recv_weighted_vx.resize(recv_disp[numRanks_]);
        recv_weighted_vy.resize(recv_disp[numRanks_]);
        recv_weighted_vz.resize(recv_disp[numRanks_]);

        MPI_Alltoallv(send_index_sph.data(), send_count.data(), send_disp.data(), MpiType<uint64_t>{}, recv_index_sph.data(),
                      recv_count.data(), recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_weight.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_weight.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_weighted_vx.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_weighted_vx.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_weighted_vy.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_weighted_vy.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_weighted_vz.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_weighted_vz.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        std::cout << "rank = " << rank_ << " alltoallv done!" << std::endl;

        // Accumulate received contributions
        for (int i = 0; i < recv_disp[numRanks_]; i++)
        {
            uint64_t index = recv_index_sph[i];
            weightSum_[index] += recv_weight[i];
            weightedVelX_[index] += recv_weighted_vx[i];
            weightedVelY_[index] += recv_weighted_vy[i];
            weightedVelZ_[index] += recv_weighted_vz[i];
        }

        // Clear the vectors
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSenderSPH[i].send_index.clear();
            vdataSenderSPH[i].send_weight.clear();
            vdataSenderSPH[i].send_weighted_vx.clear();
            vdataSenderSPH[i].send_weighted_vy.clear();
            vdataSenderSPH[i].send_weighted_vz.clear();
        }

        // Normalize velocities by dividing by weight sum
#pragma omp parallel for
        for (uint64_t i = 0; i < inboxSize; i++)
        {
            if (weightSum_[i] > 0.0)
            {
                velX_[i] = weightedVelX_[i] / weightSum_[i];
                velY_[i] = weightedVelY_[i] / weightSum_[i];
                velZ_[i] = weightedVelZ_[i] / weightSum_[i];
            }
        }

        // extrapolate mesh cells which doesn't have any particles assigned
        // extrapolateEmptyCellsFromNeighbors();
    }

    void assignVelocityByMeshCoordSPH(int meshx, int meshy, int meshz, T weight, T weightedVx, T weightedVy, T weightedVz)
    {
        int      targetRank  = calculateRankFromMeshCoord(meshx, meshy, meshz);
        uint64_t targetIndex = calculateInboxIndexFromMeshCoord(meshx, meshy, meshz);

        if (targetRank == rank_)
        {
            // if the corresponding mesh cell belongs to this rank
            uint64_t index = targetIndex;
            if (index >= weightSum_.size())
            {
                std::cout << "rank = " << rank_ << " index = " << index << " size " << weightSum_.size() << std::endl;
            }
            else
            {
                weightSum_[index] += weight;
                weightedVelX_[index] += weightedVx;
                weightedVelY_[index] += weightedVy;
                weightedVelZ_[index] += weightedVz;
            }
        }
        else
        {
            // if the corresponding mesh cell belongs another rank
            send_count[targetRank]++;
            vdataSenderSPH[targetRank].send_index.push_back(targetIndex);
            vdataSenderSPH[targetRank].send_weight.push_back(weight);
            vdataSenderSPH[targetRank].send_weighted_vx.push_back(weightedVx);
            vdataSenderSPH[targetRank].send_weighted_vy.push_back(weightedVy);
            vdataSenderSPH[targetRank].send_weighted_vz.push_back(weightedVz);
        }
    }

    void assignVelocityByMeshCoord(int meshx, int meshy, int meshz, T distance, T velox, T veloy, T veloz)
    {
        int      targetRank  = calculateRankFromMeshCoord(meshx, meshy, meshz);
        uint64_t targetIndex = calculateInboxIndexFromMeshCoord(meshx, meshy, meshz);

        if (targetRank == rank_)
        {
            // if the corresponding mesh cell belongs to this rank
            uint64_t index = targetIndex; // meshx + meshy * inbox_.size[0] + meshz * inbox_.size[0] * inbox_.size[1];
            if (index >= velX_.size())
            {
                std::cout << "rank = " << rank_ << " index = " << index << " size " << velX_.size() << std::endl;
            }

            if (distance < distance_[index])
            {
                velX_[index]     = velox;
                velY_[index]     = veloy;
                velZ_[index]     = veloz;
                distance_[index] = distance;
            }
        }
        else
        {
            // if the corresponding mesh cell belongs another rank
            send_count[targetRank]++;
            vdataSender[targetRank].send_index.push_back(targetIndex);
            vdataSender[targetRank].send_distance.push_back(distance);
            vdataSender[targetRank].send_vx.push_back(velox);
            vdataSender[targetRank].send_vy.push_back(veloy);
            vdataSender[targetRank].send_vz.push_back(veloz);
        }
    }

    void extrapolateEmptyCellsFromNeighbors()
    {
        std::cout << "rank = " << rank_ << " extrapolate cells" << std::endl;

        // Read from immutable snapshots so parallel threads don't see each
        // other's writes when sampling neighbors.
        const std::vector<T> srcVelX = velX_;
        const std::vector<T> srcVelY = velY_;
        const std::vector<T> srcVelZ = velZ_;

#pragma omp parallel for collapse(3)
        for (int i = 0; i < inbox_.size[2]; i++)
        {
            for (int j = 0; j < inbox_.size[1]; j++)
            {
                for (int k = 0; k < inbox_.size[0]; k++)
                {
                    uint64_t index = k + j * inbox_.size[0] + i * inbox_.size[0] * inbox_.size[1];
                    if (distance_[index] == std::numeric_limits<T>::infinity())
                    {
                        // iterate over the neighbors and average the velocities of the neighbors which have a distance
                        // assigned
                        T   velXSum = 0;
                        T   velYSum = 0;
                        T   velZSum = 0;
                        int count   = 0;
                        for (int ni = i - 1; ni <= i + 1; ni++)
                        {
                            if (ni < 0 || ni >= inbox_.size[2]) continue;
                            for (int nj = j - 1; nj <= j + 1; nj++)
                            {
                                if (nj < 0 || nj >= inbox_.size[1]) continue;
                                for (int nk = k - 1; nk <= k + 1; nk++)
                                {
                                    if (nk < 0 || nk >= inbox_.size[0])
                                        continue;
                                    else
                                    {
                                        uint64_t neighborIndex = (ni * inbox_.size[1] + nj) * inbox_.size[0] + nk;
                                        assert(neighborIndex < inbox_.size[0] * inbox_.size[1] * inbox_.size[2]);
                                        if (distance_[neighborIndex] != std::numeric_limits<T>::infinity())
                                        {
                                            velXSum += srcVelX[neighborIndex];
                                            velYSum += srcVelY[neighborIndex];
                                            velZSum += srcVelZ[neighborIndex];
                                            count++;
                                        }
                                    }
                                }
                            }
                        }

                        if (count > 0)
                        {
                            velX_[index] = velXSum / count;
                            velY_[index] = velYSum / count;
                            velZ_[index] = velZSum / count;
                        }
                    }
                }
            }
        }
    }

    void calculate_power_spectrum()
    {
        // returns the velocity field square
        calculate_fft();

#ifdef USE_CUDA
        // GPU path: d_velX_/Y_/Z_ hold per-component power spectrum after FFT;
        // perform spherical averaging entirely on device, copy only the small
        // power-spectrum result arrays to host at the end.
        perform_spherical_averaging_gpu();
        gpuDataValid_ = false;
        std::cout << "done." << std::endl;
        return;
#endif

        // CPU path
        std::vector<T> freqVelo(velX_.size());

#pragma omp parallel for
        for (uint64_t i = 0; i < velX_.size(); i++)
        {
            freqVelo[i] = velX_[i] + velY_[i] + velZ_[i];
        }

        perform_spherical_averaging(freqVelo.data());
        std::cout << "done." << std::endl;
    }

    void calculate_fft()
    {
        std::cout << "rank = " << rank_ << " fft calculation started." << std::endl;
        heffte::box3d<> outbox   = inbox_;
        uint64_t        meshSize = 1;
        meshSize                 = meshSize * gridDim_ * gridDim_ * gridDim_;

#ifdef USE_CUDA
        // Use CUDA backend for GPU cases
        heffte::plan_options options = heffte::default_options<heffte::backend::cufft>();
        options.use_pencils          = usePencils_;

        heffte::fft3d<heffte::backend::cufft> fft(inbox_, outbox, MPI_COMM_WORLD, options);

        std::cout << "rank=" << rank_
                  << " heFFTe outbox=" << fft.size_outbox()
                  << " workspace=" << fft.size_workspace()
                  << " (" << (fft.size_workspace() * sizeof(std::complex<T>) >> 20) << " MB GPU)" << std::endl;

        // Use existing GPU data if available (from CUDA rasterization), otherwise allocate and copy
        uint64_t inboxSize = static_cast<uint64_t>(inbox_.size[0]) * inbox_.size[1] * inbox_.size[2];
        T* d_velX, *d_velY, *d_velZ;

        cudaError_t err;
        
        if (gpuDataValid_ && d_velX_ && d_velY_ && d_velZ_)
        {
            // Reuse existing GPU data from rasterization
            d_velX = d_velX_;
            d_velY = d_velY_;
            d_velZ = d_velZ_;
        }
        else
        {
            // Allocate and copy from host (CPU rasterization case)
            err = cudaMalloc(&d_velX, inboxSize * sizeof(T));
            if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_velX: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
            err = cudaMalloc(&d_velY, inboxSize * sizeof(T));
            if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_velY: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
            err = cudaMalloc(&d_velZ, inboxSize * sizeof(T));
            if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_velZ: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
            
            err = cudaMemcpy(d_velX, velX_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { std::cerr << "CUDA Error copying d_velX: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
            err = cudaMemcpy(d_velY, velY_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { std::cerr << "CUDA Error copying d_velY: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
            err = cudaMemcpy(d_velZ, velZ_.data(), inboxSize * sizeof(T), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { std::cerr << "CUDA Error copying d_velZ: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        }
        
        // heffte::fft3d<cufft> requires complex input.  Passing a raw T* (double*)
        // causes the cuFFT backend to operate in-place on the input buffer and leave
        // d_output untouched (all zeros).  We therefore explicitly promote each real
        // velocity component to complex<T> (imaginary = 0) before calling forward().
        //
        // In the multi-rank case with outbox == inbox_, heFFTe performs an extra
        // back-transpose to redistribute the FFT result back to the inbox layout.
        // During this step heFFTe writes the final output into the INPUT buffer
        // (d_input_cplx) rather than a separate output buffer — leaving a separate
        // d_output all-zeros.  Using d_input_cplx as both input and output (in-place
        // FFT) ensures the result is always in d_input_cplx regardless of rank count.
        std::complex<T>* d_input_cplx = nullptr;
        err = cudaMalloc(&d_input_cplx, inboxSize * sizeof(std::complex<T>));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_input_cplx: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

        int threadsPerBlock = 256;
        int blocksPerGrid = (inboxSize + threadsPerBlock - 1) / threadsPerBlock;
        T   meshSizeT     = static_cast<T>(meshSize);

        // Helper lambda: zero imaginary parts, copy real velocities, then in-place
        // FFT + power spectrum.  d_input_cplx serves as both input and output so
        // the result is always available in d_input_cplx after forward().
        auto fftComponent = [&](T* d_vel) {
            // Zero the complex buffer (sets both re and im to 0).
            cudaMemset(d_input_cplx, 0, inboxSize * sizeof(std::complex<T>));
            // Copy real velocities into the real part of each complex element.
            // cudaMemcpy2D: dst stride = sizeof(complex<T>), src stride = sizeof(T),
            // so d_input_cplx[i].re = d_vel[i], d_input_cplx[i].im stays 0.
            cudaMemcpy2D(d_input_cplx,            sizeof(std::complex<T>),
                         d_vel,                   sizeof(T),
                         sizeof(T),               inboxSize,
                         cudaMemcpyDeviceToDevice);

            // In-place FFT: input == output, result written back to d_input_cplx.
            fft.forward(d_input_cplx, d_input_cplx);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { std::cerr << "CUDA Error synchronizing FFT: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

            launchComputePowerSpectrumKernel(d_input_cplx, d_vel, inboxSize, meshSizeT, blocksPerGrid, threadsPerBlock);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { std::cerr << "CUDA Error synchronizing power spectrum kernel: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        };

        fftComponent(d_velX);
        fftComponent(d_velY);
        fftComponent(d_velZ);

        cudaFree(d_input_cplx);

        // Keep per-component power spectrum on device for GPU spherical averaging.
        // If d_velX/Y/Z were locally allocated (CPU rasterization path), transfer
        // ownership to the persistent mesh device pointers.
        if (!gpuDataValid_ || !d_velX_ || !d_velY_ || !d_velZ_)
        {
            if (d_velX_) cudaFree(d_velX_);
            if (d_velY_) cudaFree(d_velY_);
            if (d_velZ_) cudaFree(d_velZ_);
            d_velX_ = d_velX;
            d_velY_ = d_velY;
            d_velZ_ = d_velZ;
        }
        // d_velX_/Y_/Z_ now hold per-component |FFT|²/N² values; keep for spherical averaging.
        gpuDataValid_ = true;
#else
        // Use FFTW backend for CPU cases
        heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();
        options.use_pencils          = usePencils_;

        heffte::fft3d<heffte::backend::fftw> fft(inbox_, outbox, MPI_COMM_WORLD, options);

        std::cout << "rank=" << rank_
                  << " heFFTe outbox=" << fft.size_outbox()
                  << " workspace=" << fft.size_workspace()
                  << " (" << (fft.size_workspace() * sizeof(std::complex<T>) >> 20) << " MB host)" << std::endl;

        std::vector<std::complex<T>> output(fft.size_outbox());

        // divide the fft.forward results by the mesh size as the first step of normalization
        fft.forward(velX_.data(), output.data());

#pragma omp parallel for
        for (uint64_t i = 0; i < velX_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velX_[i] = out * out;
        }

        fft.forward(velY_.data(), output.data());

#pragma omp parallel for
        for (uint64_t i = 0; i < velY_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velY_[i] = out * out;
        }

        fft.forward(velZ_.data(), output.data());

#pragma omp parallel for
        for (uint64_t i = 0; i < velZ_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velZ_[i] = out * out;
        }
#endif
    }

    // Implemented following numpy.fft.fftfreq
    void fftfreq(std::vector<T>& freq, int n, double dt)
    {
        if (n % 2 == 0)
        {
            for (int i = 0; i < n / 2; i++)
            {
                freq[i] = i / (n * dt);
            }
            for (int i = n / 2; i < n; i++)
            {
                freq[i] = (i - n) / (n * dt);
            }
        }
        else
        {
            for (int i = 0; i < (n - 1) / 2; i++)
            {
                freq[i] = i / (n * dt);
            }
            for (int i = (n - 1) / 2; i < n; i++)
            {
                freq[i] = (i - n + 1) / (n * dt);
            }
        }
    }

#ifdef USE_CUDA
    // GPU version of spherical averaging
    void perform_spherical_averaging_gpu()
    {
        std::cout << "rank = " << rank_ << " spherical averaging started (GPU)." << std::endl;
        
        uint64_t inboxSize = static_cast<uint64_t>(inbox_.size[0]) * inbox_.size[1] * inbox_.size[2];
        
        // Allocate device memory for freqVelo if not already allocated
        if (!d_freqVelo_)
        {
            cudaError_t err = cudaMalloc(&d_freqVelo_, inboxSize * sizeof(T));
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA Error allocating d_freqVelo_: " << cudaGetErrorString(err) << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        
        // Compute freqVelo = velX + velY + velZ on GPU
        int threadsPerBlock = 256;
        int blocksPerGrid = (inboxSize + threadsPerBlock - 1) / threadsPerBlock;
        launchComputeFreqVeloKernel(d_velX_, d_velY_, d_velZ_, d_freqVelo_, inboxSize, blocksPerGrid, threadsPerBlock);
        
        // Prepare k_values and k_1d on host (small arrays)
        std::vector<T> k_values(gridDim_);
        std::vector<T> k_1d(gridDim_);
        fftfreq(k_values, gridDim_, 1.0 / gridDim_);
        for (int i = 0; i < gridDim_; i++)
        {
            k_1d[i] = std::abs(k_values[i]);
        }
        
        // Allocate device memory for k arrays and accumulation arrays
        T* d_k_values, *d_k_1d, *d_ps_rad;
        int* d_count, *d_inboxSize, *d_inboxLow;
        
        cudaError_t err;
        err = cudaMalloc(&d_k_values, gridDim_ * sizeof(T));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_k_values: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMalloc(&d_k_1d, gridDim_ * sizeof(T));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_k_1d: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMalloc(&d_ps_rad, numShells_ * sizeof(T));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_ps_rad: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMalloc(&d_count, numShells_ * sizeof(int));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_count: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMalloc(&d_inboxSize, 3 * sizeof(int));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_inboxSize: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMalloc(&d_inboxLow, 3 * sizeof(int));
        if (err != cudaSuccess) { std::cerr << "CUDA Error allocating d_inboxLow: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        
        // Copy data to device
        err = cudaMemcpy(d_k_values, k_values.data(), gridDim_ * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "CUDA Error copying k_values: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMemcpy(d_k_1d, k_1d.data(), gridDim_ * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "CUDA Error copying k_1d: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        
        std::array<int, 3> inboxSizeArr = {inbox_.size[0], inbox_.size[1], inbox_.size[2]};
        std::array<int, 3> inboxLowArr = {inbox_.low[0], inbox_.low[1], inbox_.low[2]};
        err = cudaMemcpy(d_inboxSize, inboxSizeArr.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "CUDA Error copying inboxSize: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMemcpy(d_inboxLow, inboxLowArr.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "CUDA Error copying inboxLow: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        
        // Initialize accumulation arrays
        err = cudaMemset(d_ps_rad, 0, numShells_ * sizeof(T));
        if (err != cudaSuccess) { std::cerr << "CUDA Error memset d_ps_rad: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMemset(d_count, 0, numShells_ * sizeof(int));
        if (err != cudaSuccess) { std::cerr << "CUDA Error memset d_count: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        
        // Launch spherical averaging kernel
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((inbox_.size[0] + blockSize.x - 1) / blockSize.x,
                      (inbox_.size[1] + blockSize.y - 1) / blockSize.y,
                      (inbox_.size[2] + blockSize.z - 1) / blockSize.z);
        launchSphericalAveragingKernel(d_freqVelo_, d_k_values, d_k_1d, d_ps_rad, d_count,
                                       d_inboxSize, d_inboxLow, gridDim_, numShells_,
                                       gridSize, blockSize);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { std::cerr << "CUDA Error synchronizing spherical averaging: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }

        // Copy results back to host
        std::vector<T> ps_rad(numShells_);
        std::vector<int> count(numShells_);
        err = cudaMemcpy(ps_rad.data(), d_ps_rad, numShells_ * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { std::cerr << "CUDA Error copying ps_rad: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        err = cudaMemcpy(count.data(), d_count, numShells_ * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { std::cerr << "CUDA Error copying count: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }
        
        // MPI reduction and normalization (same as CPU version)
        std::vector<int> counts(numShells_, 0);
        MPI_Reduce(ps_rad.data(), power_spectrum_.data(), numShells_, MpiType<T>{}, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(count.data(), counts.data(), numShells_, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Normalize the power spectrum
        if (rank_ == 0)
        {
            T sum_ps_radial = std::accumulate(power_spectrum_.begin(), power_spectrum_.end(), 0.0);
            std::cout << "sum_ps_radial: " << sum_ps_radial << std::endl;
            
#pragma omp parallel for
            for (int i = 0; i < numShells_; i++)
            {
                if (counts[i] != 0)
                    power_spectrum_[i] =
                        (power_spectrum_[i] * 4.0 * std::numbers::pi * std::pow(k_1d[i], 2)) / counts[i];
            }
        }
        
        // Free device memory
        cudaFree(d_k_values);
        cudaFree(d_k_1d);
        cudaFree(d_ps_rad);
        cudaFree(d_count);
        cudaFree(d_inboxSize);
        cudaFree(d_inboxLow);
    }
#endif

    // The normalized power spectrum results will be stored in rank 0
    void perform_spherical_averaging(T* ps)
    {
        std::cout << "rank = " << rank_ << " spherical averaging started." << std::endl;
        std::vector<T>   k_values(gridDim_);
        std::vector<T>   k_1d(gridDim_);
        std::vector<T>   ps_rad(numShells_);
        std::vector<int> count(numShells_, 0);
        std::vector<int> counts(numShells_, 0);

        fftfreq(k_values, gridDim_, 1.0 / gridDim_);

#pragma omp parallel for
        for (int i = 0; i < gridDim_; i++)
        {
            k_1d[i] = std::abs(k_values[i]);
        }

// iterate over the ps array and assign the values to the correct radial bin
#pragma omp parallel for collapse(3)
        for (int i = 0; i < inbox_.size[2]; i++) // slow heffte order
        {
            for (int j = 0; j < inbox_.size[1]; j++) // mid heffte order
            {
                for (int k = 0; k < inbox_.size[0]; k++) // fast heffte order
                {
                    uint64_t freq_index = k + j * inbox_.size[0] + i * inbox_.size[0] * inbox_.size[1];

                    // Calculate the k indices with respect to the global mesh
                    uint64_t       k_index_i = i + inbox_.low[2];
                    uint64_t       k_index_j = j + inbox_.low[1];
                    uint64_t       k_index_k = k + inbox_.low[0];
                    T              kdist     = std::sqrt(k_values[k_index_i] * k_values[k_index_i] +
                                                         k_values[k_index_j] * k_values[k_index_j] +
                                                         k_values[k_index_k] * k_values[k_index_k]);
                    uint64_t k_index = std::min(static_cast<uint64_t>(std::round(kdist)),
                                                static_cast<uint64_t>(numShells_ - 1));

#pragma omp atomic
                    ps_rad[k_index] += ps[freq_index];
#pragma omp atomic
                    count[k_index]++;
                }
            }
        }

        MPI_Reduce(ps_rad.data(), power_spectrum_.data(), numShells_, MpiType<T>{}, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(count.data(), counts.data(), numShells_, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // normalize the power spectrum
        if (rank_ == 0)
        {
            T sum_ps_radial = std::accumulate(power_spectrum_.begin(), power_spectrum_.end(), 0.0);

            std::cout << "sum_ps_radial: " << sum_ps_radial << std::endl;

#pragma omp parallel for
            for (int i = 0; i < numShells_; i++)
            {
                if (counts[i] != 0)
                    power_spectrum_[i] =
                        (power_spectrum_[i] * 4.0 * std::numbers::pi * std::pow(k_1d[i], 2)) / counts[i];
            }
        }
    }

    void rasterize_particles_to_mesh_cell_avg(const std::vector<KeyType>& keys, const std::vector<T>& x,
                                               const std::vector<T>& y, const std::vector<T>& z,
                                               const std::vector<T>& vx, const std::vector<T>& vy,
                                               const std::vector<T>& vz, int powerDim)
    {
        std::cout << "rank" << rank_ << " rasterize start (cell_avg) " << powerDim << std::endl;
        std::cout << "rank" << rank_ << " keys between " << keys.front() << " - " << keys.back() << std::endl;

        uint64_t inboxSize = static_cast<uint64_t>(inbox_.size[0]) * static_cast<uint64_t>(inbox_.size[1]) *
                             static_cast<uint64_t>(inbox_.size[2]);

        // Reset accumulators and communication state
        std::fill(cellAvgVelX_.begin(), cellAvgVelX_.end(), T(0));
        std::fill(cellAvgVelY_.begin(), cellAvgVelY_.end(), T(0));
        std::fill(cellAvgVelZ_.begin(), cellAvgVelZ_.end(), T(0));
        std::fill(cellCount_.begin(), cellCount_.end(), 0);
        std::fill(send_count.begin(), send_count.end(), 0);
        std::fill(send_disp.begin(), send_disp.end(), 0);
        std::fill(recv_disp.begin(), recv_disp.end(), 0);

        int particleIndex = 0;
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            auto crd    = calculateKeyIndices(*it, gridDim_);
            int  indexi = std::get<0>(crd);
            int  indexj = std::get<1>(crd);
            int  indexk = std::get<2>(crd);

            int      targetRank  = calculateRankFromMeshCoord(indexi, indexj, indexk);
            uint64_t targetIndex = calculateInboxIndexFromMeshCoord(indexi, indexj, indexk);

            if (targetRank == rank_)
            {
                cellAvgVelX_[targetIndex] += vx[particleIndex];
                cellAvgVelY_[targetIndex] += vy[particleIndex];
                cellAvgVelZ_[targetIndex] += vz[particleIndex];
                cellCount_[targetIndex]++;
            }
            else
            {
                send_count[targetRank]++;
                vdataSenderCellAvg[targetRank].send_index.push_back(targetIndex);
                vdataSenderCellAvg[targetRank].send_vx.push_back(vx[particleIndex]);
                vdataSenderCellAvg[targetRank].send_vy.push_back(vy[particleIndex]);
                vdataSenderCellAvg[targetRank].send_vz.push_back(vz[particleIndex]);
            }
            particleIndex++;
        }

        MPI_Alltoall(send_count.data(), 1, MpiType<int>{}, recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

        for (int i = 0; i < numRanks_; i++)
        {
            send_disp[i + 1] = send_disp[i] + send_count[i];
            recv_disp[i + 1] = recv_disp[i] + recv_count[i];
        }

        // Pack flattened send buffers
        send_index_cavg.resize(send_disp[numRanks_]);
        send_vx_cavg.resize(send_disp[numRanks_]);
        send_vy_cavg.resize(send_disp[numRanks_]);
        send_vz_cavg.resize(send_disp[numRanks_]);

        for (int i = 0; i < numRanks_; i++)
        {
            for (int j = send_disp[i]; j < send_disp[i + 1]; j++)
            {
                int local = j - send_disp[i];
                send_index_cavg[j] = vdataSenderCellAvg[i].send_index[local];
                send_vx_cavg[j]    = vdataSenderCellAvg[i].send_vx[local];
                send_vy_cavg[j]    = vdataSenderCellAvg[i].send_vy[local];
                send_vz_cavg[j]    = vdataSenderCellAvg[i].send_vz[local];
            }
        }

        recv_index_cavg.resize(recv_disp[numRanks_]);
        recv_vx_cavg.resize(recv_disp[numRanks_]);
        recv_vy_cavg.resize(recv_disp[numRanks_]);
        recv_vz_cavg.resize(recv_disp[numRanks_]);

        MPI_Alltoallv(send_index_cavg.data(), send_count.data(), send_disp.data(), MpiType<uint64_t>{},
                      recv_index_cavg.data(), recv_count.data(), recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vx_cavg.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                      recv_vx_cavg.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vy_cavg.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                      recv_vy_cavg.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vz_cavg.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                      recv_vz_cavg.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);

        // Accumulate remote contributions
        for (int i = 0; i < recv_disp[numRanks_]; i++)
        {
            uint64_t index = recv_index_cavg[i];
            cellAvgVelX_[index] += recv_vx_cavg[i];
            cellAvgVelY_[index] += recv_vy_cavg[i];
            cellAvgVelZ_[index] += recv_vz_cavg[i];
            cellCount_[index]++;
        }

        // Finalise: write averages into velX_/Y_/Z_; mark filled/empty for extrapolation
        std::fill(distance_.begin(), distance_.end(), std::numeric_limits<T>::infinity());
        for (uint64_t i = 0; i < inboxSize; i++)
        {
            if (cellCount_[i] > 0)
            {
                velX_[i]     = cellAvgVelX_[i] / static_cast<T>(cellCount_[i]);
                velY_[i]     = cellAvgVelY_[i] / static_cast<T>(cellCount_[i]);
                velZ_[i]     = cellAvgVelZ_[i] / static_cast<T>(cellCount_[i]);
                distance_[i] = T(0); // finite sentinel → cell is filled
            }
            // else: velX_/Y_/Z_ remains 0, distance_ remains infinity → will be extrapolated
        }

        // Clear send buffers
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSenderCellAvg[i].send_index.clear();
            vdataSenderCellAvg[i].send_vx.clear();
            vdataSenderCellAvg[i].send_vy.clear();
            vdataSenderCellAvg[i].send_vz.clear();
        }

        extrapolateEmptyCellsFromNeighbors();
        std::cout << "rank = " << rank_ << " rasterize (cell_avg) done!" << std::endl;
    }

    void setSimBox(T Lmin, T Lmax)
    {
        Lmin_ = Lmin;
        Lmax_ = Lmax;
    }

    void resize_comm_size(const int size)
    {
        send_disp.resize(size + 1, 0);
        send_count.resize(size, 0);
        recv_disp.resize(size + 1, 0);
        recv_count.resize(size, 0);

        vdataSender.resize(size);
        vdataSenderSPH.resize(size);
        vdataSenderCellAvg.resize(size);
    }

    T calculateDistance(T x, T y, T z, int i, int j, int k)
    {
        T deltaMesh = (Lmax_ - Lmin_) / gridDim_;
        T cellX     = Lmin_ + (i + 0.5) * deltaMesh;
        T cellY     = Lmin_ + (j + 0.5) * deltaMesh;
        T cellZ     = Lmin_ + (k + 0.5) * deltaMesh;

        T xDistance = std::pow(x - cellX, 2);
        T yDistance = std::pow(y - cellY, 2);
        T zDistance = std::pow(z - cellZ, 2);
        return std::sqrt(xDistance + yDistance + zDistance);
    }

    // Get cell center coordinates for a given global mesh index
    T getCellCenterX(int i) const
    {
        T deltaMesh     = (Lmax_ - Lmin_) / gridDim_;
        T centerCoord   = deltaMesh / 2;
        T startingCoord = Lmin_ + centerCoord;
        return startingCoord + deltaMesh * i;
    }

    T getCellCenterY(int j) const
    {
        return getCellCenterX(j); // same calculation for y and z
    }

    T getCellCenterZ(int k) const
    {
        return getCellCenterX(k); // same calculation for y and z
    }

    // SPH cubic spline kernel (Monaghan 1992)
    // Returns the kernel weight W(r, h) where r is distance and h is smoothing length
    // The kernel is normalized and has compact support: W(r, h) = 0 for r >= 2*h
    T sphKernel(T r, T h) const
    {
        if (h <= 0.0) return 0.0;
        
        T q = r / h;
        if (q >= 2.0) return 0.0;

        T sigma = 1.0 / (std::numbers::pi * h * h * h); // normalization constant for 3D
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

    inline int calculateRankFromMeshCoord(int i, int j, int k)
    {
        int xBox = i / inbox_.size[0];
        int yBox = j / inbox_.size[1];
        int zBox = k / inbox_.size[2];

        int rank = xBox + yBox * proc_grid_[0] + zBox * proc_grid_[0] * proc_grid_[1];

        return rank;
    }

    inline int calculateInboxIndexFromMeshCoord(int i, int j, int k)
    {
        int xBox = i % inbox_.size[0];
        int yBox = j % inbox_.size[1];
        int zBox = k % inbox_.size[2];

        int index = xBox + yBox * inbox_.size[0] + zBox * inbox_.size[0] * inbox_.size[1];

        return index;
    }

    std::tuple<int, int, int> calculateKeyIndices(KeyType key, int gridDim)
    {
        auto mesh_indices = cstone::decodeHilbert(key);
        // unsigned divisor      = std::pow(2, (21 - powerDim));
        unsigned divisor = 1 + (1 << 21) / gridDim;

        int meshCoordX_base = util::get<0>(mesh_indices) / divisor;
        int meshCoordY_base = util::get<1>(mesh_indices) / divisor;
        int meshCoordZ_base = util::get<2>(mesh_indices) / divisor;

        // std::cout << "key: " << key << " mesh indices: " << meshCoordX_base << " " << meshCoordY_base << " " <<
        // meshCoordZ_base << std::endl;
        return std::tie(meshCoordX_base, meshCoordY_base, meshCoordZ_base);
    }

private:
    heffte::box3d<> initInbox()
    {
        heffte::box3d<> all_indexes({0, 0, 0}, {gridDim_ - 1, gridDim_ - 1, gridDim_ - 1});

        proc_grid_ = heffte::proc_setup_min_surface(all_indexes, numRanks_);
        // print proc_grid
        std::cout << "rank = " << rank_ << " proc_grid: " << proc_grid_[0] << " " << proc_grid_[1] << " "
                  << proc_grid_[2] << std::endl;

        // split all indexes across the processor grid, defines a set of boxes
        std::vector<heffte::box3d<>> all_boxes = heffte::split_world(all_indexes, proc_grid_);

        return all_boxes[rank_];
    }

    // Calculates the volume centers instead of starting with Lmin and adding deltaMesh
    void setCoordinates(T Lmin, T Lmax)
    {
        T deltaMesh     = (Lmax - Lmin) / static_cast<T>(gridDim_);
        T centerCoord   = deltaMesh / 2;
        T startingCoord = Lmin + centerCoord;
        T displacement  = static_cast<T>(inbox_.low[0]) / static_cast<T>(gridDim_) * (Lmax - Lmin);

#pragma omp parallel for
        for (int i = 0; i < inbox_.size[0]; i++)
        {
            x_[i] = (startingCoord + displacement) + deltaMesh * i;
        }
    }
};

// Forward declaration for CUDA rasterization function
#ifdef USE_CUDA
template<typename T>
void rasterize_particles_to_mesh_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                      std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                      int powerDim);
template<typename T>
void rasterize_particles_to_mesh_sph_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                          std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                          std::vector<T> h, int powerDim);
template<typename T>
void rasterize_particles_to_mesh_cell_avg_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x,
                                               std::vector<T> y, std::vector<T> z, std::vector<T> vx,
                                               std::vector<T> vy, std::vector<T> vz, int powerDim);
#endif

#ifdef USE_NVSHMEM
template<typename T>
void rasterize_particles_to_mesh_nvshmem(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                         std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz,
                                         int powerDim);
#endif