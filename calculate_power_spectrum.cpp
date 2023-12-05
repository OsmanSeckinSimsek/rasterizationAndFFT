#include <chrono>
#include <mpi.h>
#include "file_operations.hpp"
#include "raster.hpp"

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    for (int i = 0; i < argc; i++)
        std::cout << argv[i] << std::endl;

    std::string sphexa_filename = argv[1];
    std::string gridded_filename = "griddedFile.txt";
    std::string fft_filename = "FFTFile_100.txt";
    std::string ps_filename = "psFile_100.txt";
    std::string ps_rad_filename = "psFile_rad_100.txt";
    std::string spectra_filename = argv[2];
    int simDim = atoi(argv[3]);

    int numParticles = simDim * simDim * simDim;
    int gridDim = simDim * 2;
    int gridDim3 = gridDim * gridDim * gridDim;
    int numShells = simDim;

    std::vector<double> xpos(gridDim3);
    std::vector<double> ypos(gridDim3);
    std::vector<double> zpos(gridDim3);
    // std::vector<double> vx(numParticles);
    // std::vector<double> vy(numParticles);
    // std::vector<double> vz(numParticles);
    std::vector<double> gridX(gridDim3);
    std::vector<double> gridY(gridDim3);
    std::vector<double> gridZ(gridDim3);
    std::vector<double> E(numShells);
    std::vector<double> ps(gridDim3);

    auto start = chrono::steady_clock::now();

    read_sphexa_file(sphexa_filename, gridDim3, xpos.data(), ypos.data(),
                     zpos.data(), gridX.data(), gridY.data(), gridZ.data());

    auto end = chrono::steady_clock::now();
    std::cout << "Reading file took: "
              << chrono::duration_cast<chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    // assign_velocities(xpos.data(), ypos.data(), zpos.data(), vx.data(), vy.data(), vz.data(),
    //                     gridX.data(), gridY.data(), gridZ.data(), simDim, gridDim);

    // end = chrono::steady_clock::now();
    // std::cout << "Gridding took: "
    //           << chrono::duration_cast<chrono::milliseconds>(end - start).count()
    //           << " ms" << std::endl;

    // write_gridded3D_file(gridded_filename, gridDim3, gridX.data(), gridY.data(), gridZ.data());

    // end = chrono::steady_clock::now();
    // std::cout << "Grid file written: "
    //           << chrono::duration_cast<chrono::milliseconds>(end - start).count()
    //           << " ms" << std::endl;

    calculate_spectrum(gridX.data(), gridY.data(), gridZ.data(), gridDim);

    for (int i = 0; i < gridDim3; i++)
    {
        ps[i] = gridX[i] + gridY[i] + gridZ[i];
    }

    std::vector<double> ps_rad(gridDim);
    perform_spherical_averaging(ps.data(), ps_rad.data(), gridDim);

    end = chrono::steady_clock::now();
    std::cout << "Spectrum calculated: "
              << chrono::duration_cast<chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    // missing E and k related inputs
    // write_spectra_file(spectra_filename, numShells, E.data());
    // write_fft_file(fft_filename, gridX.data(), gridY.data(), gridZ.data(), gridDim);
    // write_ps_file(ps_filename, ps.data(), gridDim);
    write_spectra_file(ps_rad_filename, gridDim/2, ps_rad.data());

    end = chrono::steady_clock::now();
    std::cout << "Spectrum file written: "
              << chrono::duration_cast<chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    MPI_Finalize();
}