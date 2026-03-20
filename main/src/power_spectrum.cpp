#include <algorithm>
#include <cstring>
#include <string>

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "mesh.hpp"
#include "utils.hpp"
#include "arg_parser.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"

// Include CUDA runtime after MPI initialization to avoid conflicts
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace sphexa;

void printSpectrumHelp(char* binName, int rank);
using MeshType = double;

enum class RasterBackend
{
    Cpu,
    Cuda,
    Nvshmem
};

RasterBackend selectBackend(const ArgParser& parser)
{
    std::string mode = "auto";
    if (parser.exists("--backend"))
    {
        mode = parser.get("--backend");
    }

    if (mode == "cpu")
        return RasterBackend::Cpu;
    if (mode == "cuda" || mode == "gpudirect")
        return RasterBackend::Cuda;
    if (mode == "nvshmem")
        return RasterBackend::Nvshmem;

    // auto or unknown: prefer NVSHMEM if available, then CUDA, else CPU
#ifdef USE_NVSHMEM
    return RasterBackend::Nvshmem;
#elif defined(USE_CUDA)
    return RasterBackend::Cuda;
#else
    return RasterBackend::Cpu;
#endif
}

int main(int argc, char** argv)
{
    // For CUDA builds, we need to ensure MPI is initialized before any CUDA operations
    // The CUDA runtime might initialize automatically when libraries are loaded,
    // so we initialize MPI first to avoid conflicts
    auto [rank, numRanks] = initMpi();
    
    const ArgParser       parser(argc, (const char**)argv);
    RasterBackend         backend = selectBackend(parser);


    if (backend == RasterBackend::Nvshmem)
    {
#ifndef USE_NVSHMEM
        if (rank == 0)
        {
            std::cerr << "NVSHMEM backend requested but this binary was built without NVSHMEM support."
                      << " Reconfigure with -DRASTER_WITH_NVSHMEM=ON to enable it." << std::endl;
        }
        MPI_Finalize();
        return EXIT_FAILURE;
#else
        nvshmemx_init_attr_t nvshmem_attr;
        std::memset(&nvshmem_attr, 0, sizeof(nvshmem_attr));
        nvshmem_attr.mpi_comm = MPI_COMM_WORLD;
        int nvshmem_status    = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &nvshmem_attr);
        if (nvshmem_status != 0)
        {
            std::cerr << "Failed to initialize NVSHMEM (" << nvshmem_status << ")" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, nvshmem_status);
        }
#endif
    }

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printSpectrumHelp(argv[0], rank);
        return exitSuccess();
    }

    // Print selected backend
    if (rank == 0)
    {
        std::string backendName;
        switch (backend)
        {
            case RasterBackend::Cpu:
                backendName = "CPU/MPI";
                break;
            case RasterBackend::Cuda:
                backendName = "CUDA";
#ifdef USE_GPU_DIRECT
                backendName += " (GPU-Direct enabled)";
#endif
                break;
            case RasterBackend::Nvshmem:
                backendName = "NVSHMEM";
                break;
        }
        std::cout << "Selected rasterization backend: " << backendName << std::endl;
    }

    using KeyType        = uint64_t;
    using CoordinateType = double;

    using Domain = cstone::Domain<KeyType, CoordinateType, cstone::CpuTag>;
    
    const std::string initFile           = parser.get("--checkpoint");
    int               stepNo             = parser.get("--stepNo", 0);
    int               meshSize           = parser.get("--gridSize", 0);
    size_t            numShells          = parser.get("--numShells", 0);
    std::string       interpolationMode  = parser.get<std::string>("--interpolation", "nearest"); // "nearest" or "sph"
    std::string       outputFile         = parser.get<std::string>("--output", "power_spectrum.txt");
    bool              usePencils         = parser.exists("--pencils");

    Timer timer(std::cout);

    // read HDF5 checkpoint
    auto reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(initFile, stepNo, FileMode::collective);

    size_t numParticles = reader->globalNumParticles(); // total number of particles in the simulation
    size_t simDim       = std::cbrt(numParticles);      // dimension of the simulation

    std::vector<double> x(reader->localNumParticles());
    std::vector<double> y(reader->localNumParticles());
    std::vector<double> z(reader->localNumParticles());
    std::vector<double> h(reader->localNumParticles());
    std::vector<double> vx(reader->localNumParticles());
    std::vector<double> vy(reader->localNumParticles());
    std::vector<double> vz(reader->localNumParticles());
    std::vector<double> scratch1(x.size());
    std::vector<double> scratch2(x.size());
    std::vector<double> scratch3(x.size());

    timer.start();

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    reader->readField("h", h.data());
    reader->readField("vx", vx.data());
    reader->readField("vy", vy.data());
    reader->readField("vz", vz.data());
    reader->closeStep();

    timer.elapsed("Checkpoint read");

    std::cout << "Read " << reader->localNumParticles() << " particles on rank " << rank << std::endl;

    // get the dimensions from the checkpoint
    int powerDim = std::ceil(std::log(simDim) / std::log(2));// + 1;
    int gridDim;
    if (meshSize > 0)
    {
        gridDim = meshSize; // override if mesh size is provided as argument
    }
    else
    {
        gridDim = simDim; // dimension of the mesh // std::pow(2, powerDim);
    }
    if (numShells == 0) numShells = gridDim / 2; // default number of shells is half of the mesh dimension

    // init mesh, sim box -0.5 to 0.5 by default
    Mesh<MeshType> mesh(rank, numRanks, gridDim, numShells);
    mesh.usePencils_ = usePencils;

    // mesh.assign_velocities_to_mesh(x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(), simDim, gridDim);

    // create cornerstone tree
    std::vector<KeyType> keys(x.size());
    size_t               bucketSizeFocus = 64;
    size_t               bucketSize      = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float                theta           = 1.0;
    cstone::Box<double>  box(-0.5, 0.5, cstone::BoundaryType::periodic); // boundary type from file?
    Domain               domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(vx, vy, vz), std::tie(scratch1, scratch2, scratch3));
    // std::cout << "rank = " << rank << " numLocalParticles after sync = " << domain.nParticles() << std::endl;
    // std::cout << "rank = " << rank << " numLocalParticleswithHalos after sync = " << domain.nParticlesWithHalos()
    //           << std::endl;
    // std::cout << "rank = " << rank << " keys size after sync = " << keys.size() << std::endl;
    // std::cout << "rank = " << rank << " keys.begin = " << *keys.begin() << " keys.end = " << *keys.end() <<
    // std::endl;

    scratch1.clear();
    scratch2.clear();
    scratch3.clear();

    timer.elapsed("Sync");

    // Choose interpolation method
    if (interpolationMode == "sph")
    {
        if (rank == 0) std::cout << "Using SPH interpolation" << std::endl;
        if (backend == RasterBackend::Cuda)
        {
#ifdef USE_CUDA
            rasterize_particles_to_mesh_sph_cuda(mesh, keys, x, y, z, vx, vy, vz, h, powerDim);
#else
            mesh.rasterize_particles_to_mesh_sph(keys, x, y, z, vx, vy, vz, h, powerDim);
#endif
        }
        else
        {
            mesh.rasterize_particles_to_mesh_sph(keys, x, y, z, vx, vy, vz, h, powerDim);
        }
    }
    else if (interpolationMode == "cell_avg")
    {
        if (rank == 0) std::cout << "Using cell-average interpolation" << std::endl;
        if (backend == RasterBackend::Cuda)
        {
#ifdef USE_CUDA
            rasterize_particles_to_mesh_cell_avg_cuda(mesh, keys, x, y, z, vx, vy, vz, powerDim);
#else
            mesh.rasterize_particles_to_mesh_cell_avg(keys, x, y, z, vx, vy, vz, powerDim);
#endif
        }
        else
        {
            mesh.rasterize_particles_to_mesh_cell_avg(keys, x, y, z, vx, vy, vz, powerDim);
        }
    }
    else
    {
        // Default: nearest neighbor
        if (rank == 0) std::cout << "Using nearest neighbor interpolation" << std::endl;
        if (backend == RasterBackend::Nvshmem)
        {
#ifdef USE_NVSHMEM
            rasterize_particles_to_mesh_nvshmem(mesh, keys, x, y, z, vx, vy, vz, powerDim);
#else
            mesh.rasterize_particles_to_mesh(keys, x, y, z, vx, vy, vz, powerDim);
#endif
        }
        else if (backend == RasterBackend::Cuda)
        {
#ifdef USE_CUDA
            rasterize_particles_to_mesh_cuda(mesh, keys, x, y, z, vx, vy, vz, powerDim);
#else
            mesh.rasterize_particles_to_mesh(keys, x, y, z, vx, vy, vz, powerDim);
#endif
        }
        else
        {
            mesh.rasterize_particles_to_mesh(keys, x, y, z, vx, vy, vz, powerDim);
        }
    }

    // mesh.rasterize_using_cornerstone(keys, x, y, z, vx, vy, vz, powerDim);
    std::cout << "rasterized" << std::endl;
    timer.elapsed("Rasterization");
    // calculate power spectrum
    mesh.calculate_power_spectrum();
    timer.elapsed("Power Spectrum");

    // write power spectrum to HDF5?
    if (rank == 0)
    {
        // write power spectrum to file mesh.power_spectrum_ vector has the normalized data
        std::ofstream file(outputFile);
        for (size_t i = 1; i < mesh.numShells_; i++)
        {
            file << std::scientific << (double)(i) << " " << mesh.power_spectrum_[i] << std::endl;
        }
        file.close();
    }

    int exitCode = exitSuccess();
#ifdef USE_NVSHMEM
    if (backend == RasterBackend::Nvshmem)
    {
        nvshmem_barrier_all();
        nvshmem_finalize();
    }
#endif
    return exitCode;
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
        printf("\t--gridSize \t\t Grid size.\n\n");
        printf("\t--numShells \t\t Number of shells for averaging. Default is half of grid dimension read from the "
               "checkpoint data.\n\n");
        printf("\t--backend \t\t Rasterization backend: 'cpu', 'cuda' (or 'gpudirect'), 'nvshmem',"
               " or omit for automatic selection (prefers nvshmem, then cuda, then cpu).\n\n");
        printf("\t--interpolation \t\t Interpolation method: 'nearest' (default), 'sph', or 'cell_avg'.\n\n");
        printf("\t--output \t\t Output filename for the power spectrum (default: power_spectrum.txt).\n\n");
        printf("\t--pencils \t\t Use heFFTe pencil decomposition instead of the default slab decomposition.\n\n");
    }
}