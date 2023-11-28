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

                gridX[gridIndex] = vx[min_index];
                gridY[gridIndex] = vy[min_index];
                gridZ[gridIndex] = vz[min_index];
            }
        }
    }

}

void calculate_spectrum()
{

}