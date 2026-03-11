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

    // x_ holds the cell-center x-coordinates for this rank's local inbox slice.
    // Expected values are the full sorted set; we check only the entries owned by this rank.
    std::vector<double> all_coords = {-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45};

    for (int i = 0; i < mesh.inbox_.size[0]; i++)
    {
        int global_i = mesh.inbox_.low[0] + i;
        EXPECT_NEAR(mesh.x_[i], all_coords[global_i], 1e-12);
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

TEST(meshTest, testCellAvgRasterization)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int          gridSize  = 4;
    int          numShells = gridSize / 2;
    Mesh<double> mesh(rank, numRanks, gridSize, numShells);

    // Two particles both mapping to global cell (0,0,0):
    //   key=0  →  decodeHilbert(0) = (0,0,0)
    //   divisor = 1 + (1<<21)/4 = 524289  →  mesh cell (0/524289, ...) = (0,0,0)
    // Velocities chosen so the per-cell average is easy to verify:
    //   avg vx = (1+3)/2 = 2,  avg vy = (2+4)/2 = 3,  avg vz = (0+6)/2 = 3
    // Since every MPI rank sends the same two particles to rank 0, the global
    // accumulator is  vx_sum = 4*numRanks, count = 2*numRanks → average = 2 ✓
    std::vector<KeyType> keys = {0, 0};
    std::vector<double>  x    = {-0.45, -0.45};
    std::vector<double>  y    = {-0.45, -0.45};
    std::vector<double>  z    = {-0.45, -0.45};
    std::vector<double>  vx   = {1.0, 3.0};
    std::vector<double>  vy   = {2.0, 4.0};
    std::vector<double>  vz   = {0.0, 6.0};

    mesh.rasterize_particles_to_mesh_cell_avg(keys, x, y, z, vx, vy, vz, /*powerDim=*/2);

    // Global cell (0,0,0) is owned by rank 0; its local inbox index is 0.
    if (rank == 0)
    {
        EXPECT_NEAR(mesh.velX_[0], 2.0, 1e-12);
        EXPECT_NEAR(mesh.velY_[0], 3.0, 1e-12);
        EXPECT_NEAR(mesh.velZ_[0], 3.0, 1e-12);
        // Filled cell must have finite distance sentinel.
        EXPECT_NEAR(mesh.distance_[0], 0.0, 1e-12);
    }
}

TEST(meshTest, testCalculateRankFromMeshCoord)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int          gridSize  = 4;
    int          numShells = gridSize / 2;
    Mesh<double> mesh(rank, numRanks, gridSize, numShells);

    // Expected rank per global cell, enumerated in (z, y, x) order matching
    // the solution for 8 MPI ranks with a 4x4x4 grid.
    std::vector<int> solution = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                  1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                  3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
                                  4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
                                  6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};

    int idx = 0;
    for (int k = 0; k < gridSize; k++)
    {
        for (int j = 0; j < gridSize; j++)
        {
            for (int i = 0; i < gridSize; i++)
            {
                int out_rank = mesh.calculateRankFromMeshCoord(i, j, k);
                EXPECT_EQ(out_rank, solution[idx++]);
            }
        }
    }
}