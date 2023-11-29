#include <vector>
#include <limits>
#include "heffte.h"

void assign_velocities(double* xpos, double* ypos, double* zpos, double* vx, double* vy,
                        double* vz, double *gridX, double *gridY, double *gridZ, int simDim, int gridDim)
{
    int simDim3 = std::pow(simDim, 3);
    std::vector<double> mesh(gridDim);

    double Lmin = -0.5;
    double deltaMesh = 1.0/(gridDim-1);

    for (int i = 0; i < gridDim; i++)
    {
        mesh[i] = Lmin + i*deltaMesh;
    }

    for (int i = 0; i < gridDim; i++)
    {
        for (int j = 0; j < gridDim; j++)
        {
            #pragma omp parallel for
            for (int k = 0; k < gridDim; k++)
            {
                double min_distance = std::numeric_limits<double>::infinity();
                int min_index = -1;
                int gridIndex = (i*gridDim+j)*gridDim+k;

                for (int p = 0; p < simDim3; p++)
                {
                    double xDistance = std::pow(xpos[p] - mesh[i], 2);
                    double yDistance = std::pow(ypos[p] - mesh[j], 2);
                    double zDistance = std::pow(zpos[p] - mesh[k], 2);
                    double distance = xDistance + yDistance + zDistance;

                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        min_index = p;
                    }
                }
        // std::cout << "gridIndex " << gridIndex << " minIndex " << min_index << std::endl;
        // std::cout << "vx " << vx[min_index] << std::endl;
        // std::cout << "vy " << vy[min_index] << std::endl;
        // std::cout << "vz " << vz[min_index] << std::endl;
        // std::cout << "gridx " << gridX[gridIndex] << std::endl;
        // std::cout << "gridy " << gridY[gridIndex] << std::endl;
        // std::cout << "gridz " << gridY[gridIndex] << std::endl;

                gridX[gridIndex] = vx[min_index];
                gridY[gridIndex] = vy[min_index];
                gridZ[gridIndex] = vz[min_index];
            }
        }
    }

}

void calculate_spectrum(double *gridX, double *gridY, double *gridZ, int gridDim)
{
    int gridDim3 = gridDim * gridDim * gridDim;
    heffte::box3d<> inbox = {{0, 0, 0}, {gridDim - 1, gridDim - 1, gridDim - 1}};
    heffte::box3d<> outbox = {{0, 0, 0}, {gridDim - 1, gridDim - 1, gridDim - 1}};
    std::cout << "inbox and outbox box3d done." << std::endl;

    heffte::fft3d<heffte::backend::fftw> fft(inbox, outbox, MPI_COMM_WORLD);
    std::cout << "fft created." << std::endl;

    std::vector<double> input(fft.size_inbox());
    std::vector<std::complex<double>> output(fft.size_outbox());
    std::cout << "input and output created." << std::endl;

    for (size_t i = 0; i < gridDim3; i++)
    {
        input.at(i) = gridX[i];
    }

    fft.forward(input.data(), output.data());
    std::cout << "fft for X dim done." << std::endl;

    for (size_t i = 0; i < gridDim3; i++)
    {
        gridX[i] = abs(output.at(i));
        input.at(i) = gridY[i];
    }

    fft.forward(input.data(), output.data());
    std::cout << "fft for Y dim done." << std::endl;

    for (size_t i = 0; i < gridDim3; i++)
    {
        gridY[i] = abs(output.at(i));
        input.at(i) = gridZ[i];
    }

    fft.forward(input.data(), output.data());
    std::cout << "fft for Z dim done." << std::endl;

    for (size_t i = 0; i < gridDim3; i++)
    {
        gridZ[i] = abs(output.at(i));
    }
}