#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include "file_reader.hpp"

void read_sphexa_file(std::string in_filename, uint64_t numParticles, double *xpos,
                      double *ypos, double *zpos, double *vx, double *vy, double *vz)
{
    data_t data;
    data.reserve(numParticles);
    fileReaderFast(in_filename, data);

    for (size_t i = 0; i < data.size(); i++)
    {
        xpos[i] = data[i].x;
        ypos[i] = data[i].y;
        zpos[i] = data[i].z;
        vx[i] = data[i].vx;
        vy[i] = data[i].vy;
        vz[i] = data[i].vz;
    }
}

void write_spectra_file(std::string spectra_filename, int numShells, double *E)
{

    std::cout << "Writing in spectrum file..." << std::endl;

    std::ofstream spectra;
    spectra.open(spectra_filename, std::ios::trunc);
    for (uint64_t i = 0; i < numShells; i++)
    {
        spectra << std::setprecision(16) << E[i] << std::endl;
    }
    spectra.close();
}

void write_gridded3D_file(std::string v_filename, int gridDim3, double *gridX, double* gridY, double* gridZ)
{
    std::cout << "Writing in grid file..." << std::endl;

    std::ofstream vfile;
    vfile.open(v_filename, std::ios::trunc);
    for (uint64_t i = 0; i < gridDim3; i++)
    {
        vfile << std::setprecision(16) << gridX[i] << " "
                                      << gridY[i] << " "
                                      << gridZ[i] << " "
                                      << std::endl;
    }
    vfile.close();
}

void write_gridded_file(std::string v_filename, int gridDim3, double *gridData)
{
    std::cout << "Writing in grid file..." << std::endl;

    std::ofstream vfile;
    vfile.open(v_filename, std::ios::trunc);
    for (uint64_t i = 0; i < gridDim3; i++)
    {
        vfile << std::setprecision(16) << gridData[i] << std::endl;
    }
    vfile.close();
}
