#include <algorithm>
#include "mesh.hpp"
#include "utils.hpp"
#include "arg_parser.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"

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

    using KeyType = uint64_t;
    using CoordinateType = double;

    using Domain = cstone::Domain<KeyType, CoordinateType, cstone::CpuTag>;

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
    std::vector<double> h(reader->localNumParticles());
    std::vector<double> vx(reader->localNumParticles());
    std::vector<double> vy(reader->localNumParticles());
    std::vector<double> vz(reader->localNumParticles());
    std::vector<double> scratch1(x.size());
    std::vector<double> scratch2(x.size());
    std::vector<double> scratch3(x.size());

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    reader->readField("h", h.data());
    reader->readField("vx", vx.data());
    reader->readField("vy", vy.data());
    reader->readField("vz", vz.data());
    reader->closeStep();

    std::cout << "Read " << reader->localNumParticles() << " particles on rank " << rank << std::endl;

    // get the dimensions from the checkpoint
    int powerDim = std::ceil(std::log(simDim)/std::log(2)) + 1;
    int gridDim = std::pow(2, powerDim);               // dimension of the mesh
    if (numShells == 0) numShells = gridDim/2; // default number of shells is half of the mesh dimension

    // init mesh, sim box -0.5 to 0.5 by default
    Mesh<MeshType> mesh(rank, numRanks, gridDim, numShells);

    // mesh.assign_velocities_to_mesh(x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(), simDim, gridDim);

    // create cornerstone tree
    std::vector<KeyType> keys(x.size());
    size_t               bucketSizeFocus = 64;
    size_t               bucketSize      = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float                theta           = 1.0;
    cstone::Box<double>  box(-0.5, 0.5, cstone::BoundaryType::open); // boundary type from file?
    Domain               domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(vx, vy, vz), std::tie(scratch1, scratch2, scratch3));

    // unsigned low = 1;
    // unsigned high = 2;
    // KeyType lowerKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(low, low, low);
    // KeyType upperKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(high, high, high);
    // auto envelope = cstone::smallestCommonBox(lowerKey, upperKey);

    // // convert cornerstone tree to mesh
    // // mesh.rasterize_using_Kdtree(x.data(), y.data(), z.data(), simDim);
    // std::cout << "orig Key: " << keys[0] << std::endl;
    // std::cout << "lower Key: " << lowerKey << std::endl;
    // std::cout << "upper Key: " << upperKey << std::endl;

    // auto what = cstone::decodeHilbert<KeyType>(keys[0]);

    // // std::cout << "Hilbert key: " << iHilbert<KeyType>() << std::endl;
    // std::cout << "Key: " << cstone::iSfcKey<cstone::SfcKind<KeyType>>(std::get<0>(what), std::get<1>(what), std::get<2>(what)) << std::endl;

    // what = cstone::decodeHilbert<KeyType>(lowerKey);

    // // std::cout << "Hilbert key: " << iHilbert<KeyType>() << std::endl;
    // std::cout << "lowerKey: " << cstone::iSfcKey<cstone::SfcKind<KeyType>>(std::get<0>(what), std::get<1>(what), std::get<2>(what)) << std::endl;

    // what = cstone::decodeHilbert<KeyType>(upperKey);

    // // std::cout << "Hilbert key: " << iHilbert<KeyType>() << std::endl;
    // std::cout << "upperKey: " << cstone::iSfcKey<cstone::SfcKind<KeyType>>(std::get<0>(what), std::get<1>(what), std::get<2>(what)) << std::endl;

    // std::cout << "coords= " << x[0] << ", " << y[0] << ", " << z[0] << std::endl;
    // auto coord = cstone::sfc3D<cstone::SfcKind<KeyType>>(x[0], y[0], z[0], box);
    // std::cout << "coords tonbitint = " << cstone::toNBitInt<cstone::SfcKind<KeyType>>(x[0]+0.5) << ", " 
    //                                     << cstone::toNBitInt<cstone::SfcKind<KeyType>>(y[0]+0.5) << ", "
    //                                     << cstone::toNBitInt<cstone::SfcKind<KeyType>>(z[0]+0.5) << std::endl;
    // std::cout << "Key from coord: " << coord << std::endl;
    // what = cstone::decodeHilbert<KeyType>(coord);

    // std::cout << "coord from coord: " << cstone::iSfcKey<cstone::SfcKind<KeyType>>(std::get<0>(what), std::get<1>(what), std::get<2>(what)) << std::endl;

    // KeyType lowx = cstone::toNBitInt<cstone::SfcKind<KeyType>>(mesh.x_[0]+0.5);
    // KeyType lowy = cstone::toNBitInt<cstone::SfcKind<KeyType>>(mesh.y_[0]+0.5);
    // KeyType lowz = cstone::toNBitInt<cstone::SfcKind<KeyType>>(mesh.z_[0]+0.5);
    // KeyType highx = cstone::toNBitInt<cstone::SfcKind<KeyType>>(mesh.x_[1]+0.5);
    // KeyType highy = cstone::toNBitInt<cstone::SfcKind<KeyType>>(mesh.y_[1]+0.5);
    // KeyType highz = cstone::toNBitInt<cstone::SfcKind<KeyType>>(mesh.z_[1]+0.5);
    // std::cout << "mesh coords tonbitint = " << lowx << ", " 
    //                                     << lowy << ", "
    //                                     << lowz << std::endl;

    // KeyType lowerKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(lowx, lowy, lowz);
    // KeyType upperKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(highx, highy, highz);

    // KeyType lowerKey = cstone::sfc3D<cstone::SfcKind<KeyType>>(mesh.x_[0], mesh.y_[0], mesh.z_[0], box);
    // KeyType upperKey = cstone::sfc3D<cstone::SfcKind<KeyType>>(mesh.x_[1], mesh.y_[1], mesh.z_[1], box);

    // unsigned s = 0;
    // unsigned sm = s * std::pow(2, (21-powerDim));
    // unsigned e = 1;
    // unsigned em = e * std::pow(2, (21-powerDim));

    // KeyType lowerKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(sm, sm, sm);
    // KeyType upperKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(em, em, em);

    // unsigned level = cstone::commonPrefix(lowerKey, upperKey) / 3;
    // KeyType lowerBound = cstone::enclosingBoxCode(lowerKey, level);
    // KeyType upperBound =  lowerBound + cstone::nodeRange<KeyType>(level);

    // auto itlow = std::lower_bound(keys.begin(), keys.end(), lowerBound);
    // auto itup = std::upper_bound(keys.begin(), keys.end(), upperBound);

    // // iterate from itlow to itup
    // for (auto it = itlow; it != itup; it++)
    // {
    //     std::cout << "key: " << *it << std::endl;
    //     auto index = std::distance(keys.begin(), it);
    //     std::cout << "index: " << index << std::endl;
    // }

    // std::cout << "index of lowerKey: " << std::lower_bound(keys.begin(), keys.end(), lowerBound) << std::endl;
    // std::cout << "index of upperKey: " << std::upper_bound(keys.begin(), keys.end(), upperBound) << std::endl;

    // auto coord = cstone::sfc3D<cstone::SfcKind<KeyType>>(mesh.x_[1], mesh.y_[1], mesh.z_[1], box);
    // std::cout << "Key from coord: " << coord << std::endl;
    // auto what = cstone::decodeHilbert<KeyType>(upperKey);

    // std::cout << "coord from coord: " << std::get<0>(what) << ", " << std::get<1>(what) << ", " << std::get<2>(what) << std::endl;

    mesh.rasterize_using_cornerstone(keys, x, y, z, vx, vy, vz, powerDim);

    // calculate power spectrum
    mesh.calculate_power_spectrum();

    // write power spectrum to HDF5?
    if (rank == 0)
    {
        // write power spectrum to file mesh.power_spectrum_ vector has the normalized data
        std::ofstream file("power_spectrum.txt");
        for (size_t i = 0; i < mesh.numShells_; i++)
        {
            file << (double)(i+1) << " " << mesh.power_spectrum_[i] << std::endl;
        }
        file.close();
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