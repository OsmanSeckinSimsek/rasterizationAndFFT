#pragma once

#include <tuple>
#include <omp.h>
#include <chrono>

auto initMpi()
{
    int rank     = 0;
    int numRanks = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (rank == 0)
    {
        int mpi_version, mpi_subversion;
        MPI_Get_version(&mpi_version, &mpi_subversion);
#ifdef _OPENMP
        printf("# %d MPI-%d.%d process(es) with %d OpenMP-%u thread(s)/process\n", numRanks, mpi_version,
               mpi_subversion, omp_get_max_threads(), _OPENMP);
#else
        printf("# %d MPI-%d.%d process(es) without OpenMP\n", numRanks, mpi_version, mpi_subversion);
#endif
    }
    return std::make_tuple(rank, numRanks);
}

int exitSuccess()
{
    MPI_Finalize();
    return EXIT_SUCCESS;
}

class Timer
{
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<float>       Time;

public:
    Timer(std::ostream& out)
        : out(out)
    {
    }

    void start()
    {
        tstart = tlast = Clock::now();
    }

    //! @brief time elapsed between tstart and now
    void elapsed(const std::string func) 
    {
        tlast = Clock::now();
        out << func << " elapsed time: " << std::chrono::duration_cast<Time>(tlast - tstart).count() << std::endl;
        tstart = tlast;
    }


private:
    std::ostream&                  out;
    std::chrono::time_point<Clock> tstart, tlast;
};
