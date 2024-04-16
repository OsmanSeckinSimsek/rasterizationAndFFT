#include <mpi.h>
#include "gtest/gtest.h"
#include "heffte.h"
#include "mesh.hpp"

TEST(meshTest, testFFTFreq)
{
    int                 size = 8;
    std::vector<double> freqs(size);
    std::vector<double> solution_freqs = {0, 1, 2, 3, -4, -3, -2, -1};

    Mesh<double> mesh(0, 1, 10, 5);
    mesh.fftfreq(freqs, size, 1.0 / size);

    for (size_t i = 0; i < freqs.size(); i++)
    {
        EXPECT_NEAR(freqs[i], solution_freqs[i], 1e-15);
    }
}

TEST(mestTest, testSetCoordinates)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int          gridSize  = 10;
    int          numShells = gridSize / 2;
    Mesh<double> mesh(rank, numRanks, gridSize, numShells);

    std::vector<double> solution_x = {-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15,  0.25,  0.35,  0.45};
    std::vector<double> solution_y = {-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15,  0.25,  0.35,  0.45};
    std::vector<double> solution_z = {-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15,  0.25,  0.35,  0.45};

    for (size_t i = 0; i < mesh.x_.size(); i++)
    {
        EXPECT_NEAR(mesh.x_[i], solution_x[i], 1e-12);
        EXPECT_NEAR(mesh.y_[i], solution_y[i], 1e-12);
        EXPECT_NEAR(mesh.z_[i], solution_z[i], 1e-12);
    }
}

TEST(meshTest, testMeshInit)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int          gridSize  = 16;
    int          gridSize3 = gridSize * gridSize * gridSize;
    int          numShells = gridSize / 2;
    Mesh<double> mesh(rank, numRanks, gridSize, numShells);

    EXPECT_EQ(mesh.gridDim_, gridSize);
    EXPECT_EQ(mesh.velX_.size(), gridSize3 / numRanks);
    EXPECT_EQ(mesh.power_spectrum_.size(), numShells);
}

TEST(meshTest, testCalculateFFT)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int          gridSize  = 4;
    int          numShells = gridSize / 2;
    Mesh<double> mesh(rank, numRanks, gridSize, numShells);

    // initialize the velocity vectors
    std::iota(mesh.velX_.begin(), mesh.velX_.end(), 0);
    std::iota(mesh.velY_.begin(), mesh.velY_.end(), 0);
    std::iota(mesh.velZ_.begin(), mesh.velZ_.end(), 0);

    mesh.calculate_fft();

    // solution calculated using the code from ./scripts/power_spectra.py
    std::vector<double> solution = {
        9.9225e+02, 5.0000e-01, 2.5000e-01, 5.0000e-01, 8.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 8.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.2800e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        6.4000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.2800e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00};

    // check that the FFT was calculated correctly with a tolerance of 1e-10
    for (size_t i = 0; i < mesh.velX_.size(); i++)
    {
        EXPECT_NEAR(mesh.velX_[i], solution[i], 1e-10);
        EXPECT_NEAR(mesh.velY_[i], solution[i], 1e-10);
        EXPECT_NEAR(mesh.velZ_[i], solution[i], 1e-10);
    }
}

void setVelocitiesIota(Mesh<double> mesh, int gridDim)
{
    for (int i = 0; i < mesh.inbox_.size[2]; i++) // slow heffte order
    {
        for (int j = 0; j < mesh.inbox_.size[1]; j++) // mid heffte order
        {
            for (int k = 0; k < mesh.inbox_.size[0]; k++) // fast heffte order
            {
                int boxIndex = (i * mesh.inbox_.size[1] + j) * mesh.inbox_.size[0] + k;
                int gridIndex =
                    ((i + mesh.inbox_.low[2]) * gridDim + (j + mesh.inbox_.low[1])) * gridDim + mesh.inbox_.low[0] + k;
                mesh.velX_[boxIndex] = gridIndex;
                mesh.velY_[boxIndex] = gridIndex;
                mesh.velZ_[boxIndex] = gridIndex;
            }
        }
    }
}

TEST(meshTest, testSphericalAveraging)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int                 gridSize  = 4;
    int                 gridSize3 = gridSize * gridSize * gridSize;
    Mesh<double>        mesh(rank, numRanks, gridSize, gridSize / 2);
    std::vector<double> freqVelo(gridSize3);

    setVelocitiesIota(mesh, gridSize);
    std::iota(freqVelo.begin(), freqVelo.end(), 0);

    mesh.perform_spherical_averaging(freqVelo.data());
}


TEST(meshTest, testCornerstoneRasterization)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using KeyType = uint64_t;
    using CoordinateType = double;
    using MeshType = double;

    using Domain = cstone::Domain<KeyType, CoordinateType, cstone::CpuTag>;

    int domainSize = 8;
    int numParticles = domainSize * domainSize * domainSize;

    std::vector<double>  x(numParticles);
    std::vector<double>  y(numParticles);
    std::vector<double>  z(numParticles);
    std::vector<double>  h(numParticles);
    std::vector<double> vx(numParticles);
    std::vector<double> vy(numParticles);
    std::vector<double> vz(numParticles);
    std::vector<double> scratch1(x.size());
    std::vector<double> scratch2(x.size());
    std::vector<double> scratch3(x.size());

    std::vector<double> coords = {-0.4, -0.35, -0.2, -0.07, 0.07, 0.2, 0.35, 0.4};

    int powerDim = std::ceil(std::log(domainSize)/std::log(2));
    int gridDim = std::pow(2, powerDim); // dimension of the mesh
    int numShells = gridDim/2; // default number of shells is half of the mesh dimension

    for (int i = 0; i < domainSize; i++)
    {
        for (int j = 0; j < domainSize; j++)
        {
            for (int k = 0; k < domainSize; k++)
            {
                int index = (i * domainSize + j) * domainSize + k;
                x[index] = coords[i];
                y[index] = coords[j];
                z[index] = coords[k];
                vx[index] = index;
                vy[index] = index;
                vz[index] = index;
            }
        }
    }

    // init mesh, sim box -0.5 to 0.5 by default
    Mesh<MeshType> mesh(rank, numRanks, gridDim, numShells);

    // mesh.assign_velocities_to_mesh(x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(), simDim, gridDim);

    // create cornerstone tree
    std::vector<KeyType> keys(x.size());
    size_t               bucketSizeFocus = 64;
    size_t               bucketSize      = bucketSizeFocus;//std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float                theta           = 1.0;
    cstone::Box<double>  box(-0.5, 0.5, cstone::BoundaryType::open); // boundary type from file?
    Domain               domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(vx, vy, vz), std::tie(scratch1, scratch2, scratch3));

    std::vector<double> mesh_coords = {-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5};

    // mesh.rasterize_using_cornerstone(keys, x, y, z, vx, vy, vz, powerDim);
    int iter = domainSize;
    int meshKeySize = std::pow(2, (21-powerDim));

    for (int i = 0; i < iter - 1; i++)
    {
        for (int j = 0; j < iter - 1; j++)
        {
            for (int k = 0; k < iter - 1; k++)
            {
                unsigned iSFC_low = i * meshKeySize;
                unsigned iSFC_up = (i+1) * meshKeySize;
                unsigned jSFC_low = j * meshKeySize;
                unsigned jSFC_up = (j+1) * meshKeySize;
                unsigned kSFC_low = k * meshKeySize;
                unsigned kSFC_up = (k+1) * meshKeySize;

                KeyType lowerKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(iSFC_low, jSFC_low, kSFC_low);
                KeyType upperKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(iSFC_up, jSFC_up, kSFC_up);

                unsigned level = cstone::commonPrefix(lowerKey, upperKey) / 3;
                KeyType lowerBound = cstone::enclosingBoxCode(lowerKey, level);
                KeyType upperBound =  lowerBound + cstone::nodeRange<KeyType>(level);

                auto itlow = std::lower_bound(keys.begin(), keys.end(), lowerBound);
                auto itup = std::upper_bound(keys.begin(), keys.end(), upperBound);
                std::cout << "index from key: " << std::distance(itlow, itup) << std::endl;

                KeyType lowerKeyFromCoord = cstone::sfc3D<cstone::SfcKind<KeyType>>(mesh_coords[i], mesh_coords[j], mesh_coords[k], box);
                KeyType upperKeyFromCoord = cstone::sfc3D<cstone::SfcKind<KeyType>>(mesh_coords[i+1], mesh_coords[j+1], mesh_coords[k+1], box);
                std::cout << "Coord key search x = " << mesh_coords[i] << " and " << mesh_coords[i+1] << " y = " 
                                                     << mesh_coords[j] << " and " << mesh_coords[j+1] << " z = " 
                                                     << mesh_coords[k] << " and " << mesh_coords[k+1] << std::endl;

                level = cstone::commonPrefix(lowerKeyFromCoord, upperKeyFromCoord) / 3;
                lowerBound = cstone::enclosingBoxCode(lowerKeyFromCoord, level);
                upperBound =  lowerBound + cstone::nodeRange<KeyType>(level);

                itlow = std::lower_bound(keys.begin(), keys.end(), lowerBound);
                itup = std::upper_bound(keys.begin(), keys.end(), upperBound);
                std::cout << "index from coord: " << std::distance(itlow, itup) << std::endl;

                theta = 1.0;
            }
        }
    }

    std::cout << "rasterized" << std::endl;
}