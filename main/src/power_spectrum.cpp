#include "mesh.hpp"
#include "utils.hpp"
#include "arg_parser.hpp"
#include "ifile_io_impl.h"

using namespace sphexa;

void printSpectrumHelp(char* binName, int rank);
using MeshType = double;

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printSpectrumHelp(argv[0], rank);
        return exitSuccess();
    }

    const std::string initFile           = parser.get("--checkpoint");
    int               stepNo             = parser.get("--stepNo", 0);
    float             meshSizeMultiplier = parser.get("--meshSizeMultiplier", 1.0);
    size_t            numShells          = parser.get("--numShells", 0);

    // read HDF5 checkpoint
    auto reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(initFile, stepNo, FileMode::collective);

    size_t numParticles = reader->globalNumParticles(); // total number of particles in the simulation
    size_t simDim       = std::cbrt(numParticles);      // dimension of the simulation

    std::vector<double> x(reader->localNumParticles());
    std::vector<double> y(reader->localNumParticles());
    std::vector<double> z(reader->localNumParticles());
    std::vector<double> vx(reader->localNumParticles());
    std::vector<double> vy(reader->localNumParticles());
    std::vector<double> vz(reader->localNumParticles());

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    reader->readField("vx", vx.data());
    reader->readField("vy", vy.data());
    reader->readField("vz", vz.data());

    // get the dimensions from the checkpoint
    int gridDim = simDim * 2;               // dimension of the mesh
    if (numShells == 0) numShells = simDim; // default number of shells is half of the mesh dimension

    // init mesh
    Mesh<MeshType> mesh(rank, numRanks, gridDim, numShells);

    // convert cornerstone tree to mesh

    // calculate power spectrum
    mesh.calculate_power_spectrum();

    // write power spectrum to HDF5?
    if (rank == 0)
    {
        // write power spectrum to file mesh.power_spectrum_ vector has the normalized data
    }

    return exitSuccess();
}

void printSpectrumHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--checkpoint \t\t HDF5 checkpoint file with simulation data\n\n");
        printf("\t--stepNo \t\t Step number of the HDF5 checkpoint file with simulation data\n\n");
        printf("\t--meshSizeMultiplier \t\t Multiplier for the mesh size over the grid size.\n\n");
        printf("\t--numShells \t\t Number of shells for averaging. Default is half of mesh dimension read from the "
               "checkpoint data.\n\n");
    }
}