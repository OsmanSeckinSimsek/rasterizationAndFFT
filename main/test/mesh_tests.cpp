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

TEST(meshTest, testCalculateRankFromMeshCoord)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int          gridSize  = 4;
    int          numShells = gridSize / 2;
    Mesh<double> mesh(rank, numRanks, gridSize, numShells);

    std::vector<int> solution = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                  1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                  3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
                                  4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
                                  6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};

    for (size_t i = 0; i < mesh.x_.size(); i++)
    {
        int out_rank = mesh.calculateRankFromMeshCoord(mesh.x_[i], mesh.y_[i], mesh.z_[i]);
        EXPECT_EQ(out_rank, solution[i]);
    }
}