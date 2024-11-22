#include <vector>
#include <limits>
#include "heffte.h"
#include "cstone/domain/domain.hpp"

using KeyType = uint64_t;

struct DataSender
{
    // vectors to send to each rank in all_to_allv
    std::vector<uint64_t> send_index;
    std::vector<double> send_distance;
    std::vector<double> send_vx;
    std::vector<double> send_vy;
    std::vector<double> send_vz;
};

template<typename T>
class Mesh
{
public:
    int rank_;
    int numRanks_;
    int gridDim_; // specifically integer because heffte library uses int
    int numShells_;
    T Lmin_;
    T Lmax_;
    std::array<int, 3> proc_grid_;

    heffte::box3d<> inbox_;
    // coordinate centers in the mesh
    std::vector<T>  x_;
    // std::vector<T>  y_;
    // std::vector<T>  z_;
    // corresponding velocities in the mesh
    std::vector<T>  velX_;
    std::vector<T>  velY_;
    std::vector<T>  velZ_;
    // particle's distance to mesh point
    std::vector<T>  distance_;
    std::vector<T>  power_spectrum_;

    //communication counters
    std::vector<int> send_disp; //(numRanks_+1, 0);
    std::vector<int> send_count; //(numRanks_, 0);

    std::vector<int> recv_disp; //(numRanks_+1, 0);
    std::vector<int> recv_count; //(numRanks_, 0);

    std::vector<DataSender> vdataSender;

    // vectors to receive from each rank in all_to_allv
    std::vector<uint64_t> send_index;
    std::vector<T>        send_distance;
    std::vector<T>        send_vx;
    std::vector<T>        send_vy;
    std::vector<T>        send_vz;

    // vectors to receive from each rank in all_to_allv
    std::vector<uint64_t> recv_index;
    std::vector<T>        recv_distance;
    std::vector<T>        recv_vx;
    std::vector<T>        recv_vy;
    std::vector<T>        recv_vz;

    // sim box -0.5 to 0.5 by default
    Mesh(int rank, int numRanks, int gridDim, int numShells)
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , numShells_(numShells)
        , Lmin_(-0.5)
        , Lmax_(0.5)
        , inbox_(initInbox())
    {
        uint64_t inboxSize = static_cast<uint64_t>(inbox_.size[0]) * static_cast<uint64_t>(inbox_.size[1]) *
                           static_cast<uint64_t>(inbox_.size[2]);
        std::cout << "rank = " << rank << " griddim = " << gridDim << " inboxSize = " << inboxSize << std::endl;
        std::cout << "rank = " << rank << " inbox low = " << inbox_.low[0] << " " << inbox_.low[1] << " " << inbox_.low[2] << std::endl;
        std::cout << "rank = " << rank << " inbox high = " << inbox_.high[0] << " " << inbox_.high[1] << " " << inbox_.high[2] << std::endl;
        velX_.resize(inboxSize);
        velY_.resize(inboxSize);
        velZ_.resize(inboxSize);
        x_.resize(inbox_.size[0]);
        // y_.resize(inbox_.size[1]);
        // z_.resize(inbox_.size[2]);
        distance_.resize(inboxSize);
        power_spectrum_.resize(numShells);

        resize_comm_size(numRanks);
        std::fill(distance_.begin(), distance_.end(), std::numeric_limits<T>::infinity());

        // populate the x_, y_, z_ vectors with the center coordinates of the mesh cells
        setCoordinates(Lmin_, Lmax_);
    }

    void rasterize_particles_to_mesh(std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                     std::vector<T> z, std::vector<T> vx, std::vector<T> vy, std::vector<T> vz, int powerDim)
    {
        std::cout << "rank" << rank_ << " rasterize start " << powerDim << std::endl;
        std::cout << "rank" << rank_ << " keys between " << *keys.begin() << " - " << *keys.end() << std::endl;
        
        int particleIndex = 0;
        // iterate over keys vector
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            auto crd = calculateKeyIndices(*it, powerDim);
            int indexi = std::get<0>(crd);
            int indexj = std::get<1>(crd);
            int indexk = std::get<2>(crd);
            
            assert(indexi < std::pow(2, powerDim));
            assert(indexj < std::pow(2, powerDim));
            assert(indexk < std::pow(2, powerDim));
            
            double distance = calculateDistance(x[particleIndex], y[particleIndex], z[particleIndex], indexi, indexj, indexk);
            assignVelocityByMeshCoord(indexi, indexj, indexk, distance, vx[particleIndex], vy[particleIndex], vz[particleIndex]);
            particleIndex++;
        }
        x.clear();
        y.clear();
        z.clear();
        
        std::cout << "rank = " << rank_ << " particleIndex = " << particleIndex << std::endl;
        for (int i = 0; i < numRanks_; i++)
            std::cout << "rank = " << rank_ << " send_count = " << send_count[i] << std::endl;

        MPI_Alltoall(send_count.data(), 1, MpiType<int>{}, recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

        for (int i = 0; i < numRanks_; i++)
        {
            send_disp[i + 1] = send_disp[i] + send_count[i];
            recv_disp[i + 1] = recv_disp[i] + recv_count[i];
        }

        // prepare send buffers
        send_index.resize(send_disp[numRanks_]);
        send_distance.resize(send_disp[numRanks_]);
        send_vx.resize(send_disp[numRanks_]);
        send_vy.resize(send_disp[numRanks_]);
        send_vz.resize(send_disp[numRanks_]);
        std::cout << "rank = " << rank_ << " buffers allocated" << std::endl;

        for (int i = 0; i < numRanks_; i++)
        {
            for (int j = send_disp[i]; j < send_disp[i + 1]; j++)
            {
                send_index[j] = vdataSender[i].send_index[j - send_disp[i]];
                send_distance[j] = vdataSender[i].send_distance[j - send_disp[i]];
                send_vx[j] = vdataSender[i].send_vx[j - send_disp[i]];
                send_vy[j] = vdataSender[i].send_vy[j - send_disp[i]];
                send_vz[j] = vdataSender[i].send_vz[j - send_disp[i]];
            }
        }
        std::cout << "rank = " << rank_ << " buffers transformed" << std::endl;

        // prepare receive buffers
        recv_index.resize(recv_disp[numRanks_]);
        recv_distance.resize(recv_disp[numRanks_]);
        recv_vx.resize(recv_disp[numRanks_]);
        recv_vy.resize(recv_disp[numRanks_]);
        recv_vz.resize(recv_disp[numRanks_]);
        
        MPI_Alltoallv(send_index.data(), send_count.data(), send_disp.data(), MpiType<uint64_t>{},
                    recv_index.data(), recv_count.data(), recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_distance.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                    recv_distance.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vx.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                    recv_vx.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vy.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                    recv_vy.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vz.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                    recv_vz.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        std::cout << "rank = " << rank_ << " alltoallv done!" << std::endl;
        
        for (int i = 0; i < recv_disp[numRanks_]; i++)
        {
            uint64_t index = recv_index[i];
            if (recv_distance[i] < distance_[index])
            {
                velX_[index] = recv_vx[i];
                velY_[index] = recv_vy[i];
                velZ_[index] = recv_vz[i];
                distance_[index] = recv_distance[i];
            }
        }

        // clear the vectors
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSender[i].send_index.clear();
            vdataSender[i].send_distance.clear();
            vdataSender[i].send_vx.clear();
            vdataSender[i].send_vy.clear();
            vdataSender[i].send_vz.clear();
        }

        // extrapolate mesh cells which doesn't have any particles assigned
        extrapolateEmptyCellsFromNeighbors();
    }

    void assignVelocityByMeshCoord(int meshx, int meshy, int meshz, T distance, T velox, T veloy, T veloz)
    {
        int targetRank = calculateRankFromMeshCoord(meshx, meshy, meshz);
        uint64_t targetIndex = calculateInboxIndexFromMeshCoord(meshx, meshy, meshz);
        
        if (targetRank == rank_)
        {
            // if the corresponding mesh cell belongs to this rank
            uint64_t index = targetIndex; // meshx + meshy * inbox_.size[0] + meshz * inbox_.size[0] * inbox_.size[1];
            if(index >= velX_.size())
            {
                std::cout << "rank = " << rank_ << " index = " << index << " size " << velX_.size() << std::endl;
            }

            if (distance < distance_[index])
            {
                velX_[index] = velox;
                velY_[index] = veloy;
                velZ_[index] = veloz;
                distance_[index] = distance;
            }
        }
        else
        {
            // if the corresponding mesh cell belongs another rank
            send_count[targetRank]++;
            vdataSender[targetRank].send_index.push_back(targetIndex);
            vdataSender[targetRank].send_distance.push_back(distance);
            vdataSender[targetRank].send_vx.push_back(velox);
            vdataSender[targetRank].send_vy.push_back(veloy);
            vdataSender[targetRank].send_vz.push_back(veloz);
        }
        
    }

    void extrapolateEmptyCellsFromNeighbors()
    {
        std::cout << "rank = " << rank_ << " extrapolate cells" << std::endl;

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < inbox_.size[2]; i++)
        {
            for (int j = 0; j < inbox_.size[1]; j++)
            {
                for (int k = 0; k < inbox_.size[0]; k++)
                {
                    uint64_t index = k + j * inbox_.size[0] + i * inbox_.size[0] * inbox_.size[1];
                    if (distance_[index] == std::numeric_limits<T>::infinity())
                    {
                        // iterate over the neighbors and average the velocities of the neighbors which have a distance assigned
                        T velXSum = 0;
                        T velYSum = 0;
                        T velZSum = 0;
                        int count = 0;
                        for (int ni = i - 1; ni <= i + 1; ni++)
                        {
                            if (ni < 0 || ni >= inbox_.size[2])
                            continue;
                            for (int nj = j - 1; nj <= j + 1; nj++)
                            {
                                if (nj < 0 || nj >= inbox_.size[1])
                                    continue;                                
                                for (int nk = k - 1; nk <= k + 1; nk++)
                                {
                                    if (nk < 0 || nk >= inbox_.size[0])
                                    continue;
                                    else
                                    {
                                        uint64_t neighborIndex = (ni * inbox_.size[1] + nj) * inbox_.size[0] + nk;
                                        assert(neighborIndex < inbox_.size[0] * inbox_.size[1] * inbox_.size[2]);
                                        if (distance_[neighborIndex] != std::numeric_limits<T>::infinity())
                                        {
                                            velXSum += velX_[neighborIndex];
                                            velYSum += velY_[neighborIndex];
                                            velZSum += velZ_[neighborIndex];
                                            count++;
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (count > 0)
                        {
                            velX_[index] = velXSum / count;
                            velY_[index] = velYSum / count;
                            velZ_[index] = velZSum / count;
                        }
                    }
                }
            }
        }
    }

    void calculate_power_spectrum()
    {
        calculate_fft();

        std::vector<T> freqVelo(velX_.size());

        // calculate the modulus of the velocity frequencies
        #pragma omp parallel for
        for (uint64_t i = 0; i < velX_.size(); i++)
        {
            freqVelo[i] = velX_[i] + velY_[i] + velZ_[i];
        }

        // perform spherical averaging
        perform_spherical_averaging(freqVelo.data());
    }

    void calculate_fft()
    {
        std::cout << "rank = " << rank_ << " fft calculation started." << std::endl;
        heffte::box3d<> outbox   = inbox_;
        uint64_t          meshSize = gridDim_ * gridDim_ * gridDim_;

        heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();
        options.use_pencils = true;

        // change fftw depending on the configuration into cufft or rocmfft
        heffte::fft3d<heffte::backend::fftw> fft(inbox_, outbox, MPI_COMM_WORLD, options);

        std::vector<std::complex<T>> output(fft.size_outbox());

        // divide the fft.forward results by the mesh size as the first step of normalization
        fft.forward(velX_.data(), output.data());

        #pragma omp parallel for
        for (uint64_t i = 0; i < velX_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velX_[i] = out * out;
        }

        fft.forward(velY_.data(), output.data());

        #pragma omp parallel for
        for (uint64_t i = 0; i < velY_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velY_[i] = out * out;
        }

        fft.forward(velZ_.data(), output.data());

        #pragma omp parallel for
        for (uint64_t i = 0; i < velZ_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velZ_[i] = out * out;
        }
    }

    // Implemented following numpy.fft.fftfreq
    void fftfreq(std::vector<T>& freq, int n, double dt)
    {
        if (n % 2 == 0)
        {
            for (int i = 0; i < n / 2; i++)
            {
                freq[i] = i / (n * dt);
            }
            for (int i = n / 2; i < n; i++)
            {
                freq[i] = (i - n) / (n * dt);
            }
        }
        else
        {
            for (int i = 0; i < (n - 1) / 2; i++)
            {
                freq[i] = i / (n * dt);
            }
            for (int i = (n - 1) / 2; i < n; i++)
            {
                freq[i] = (i - n + 1) / (n * dt);
            }
        }
    }

    // The normalized power spectrum results will be stored in rank 0
    void perform_spherical_averaging(T* ps)
    {
        std::cout << "rank = " << rank_ << " spherical averaging started." << std::endl;
        std::vector<T> k_values(gridDim_);
        std::vector<T> k_1d(gridDim_);
        std::vector<T> ps_rad(numShells_);
        std::vector<int> count(k_1d.size());
        std::vector<int> counts(k_1d.size(), 0);

        fftfreq(k_values, gridDim_, 1.0 / gridDim_);

        #pragma omp parallel for
        for (int i = 0; i < gridDim_; i++)
        {
            k_1d[i] = std::abs(k_values[i]);
        }

        // iterate over the ps array and assign the values to the correct radial bin
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < inbox_.size[2]; i++) // slow heffte order
        {
            for (int j = 0; j < inbox_.size[1]; j++) // mid heffte order
            {
                for (int k = 0; k < inbox_.size[0]; k++) // fast heffte order
                {
                    uint64_t freq_index = k + j * inbox_.size[0] + i * inbox_.size[0] * inbox_.size[1];

                    // Calculate the k indices with respect to the global mesh
                    uint64_t k_index_i = i + inbox_.low[2];
                    uint64_t k_index_j = j + inbox_.low[1];
                    uint64_t k_index_k = k + inbox_.low[0];
                    T      kdist     = std::sqrt(k_values[k_index_i] * k_values[k_index_i] +
                                    k_values[k_index_j] * k_values[k_index_j] +
                                    k_values[k_index_k] * k_values[k_index_k]);
                    std::vector<T> k_dif(gridDim_);

                    for (int kind = 0; kind < gridDim_; kind++)
                    {
                        k_dif[kind] = std::abs(k_1d[kind] - kdist);
                    }
                    auto   it      = std::min_element(std::begin(k_dif), std::end(k_dif));
                    uint64_t k_index = std::distance(std::begin(k_dif), it);

                    #pragma omp atomic
                    ps_rad[k_index] += ps[freq_index];
                    #pragma omp atomic
                    count[k_index]++;
                }
            }
        }

        MPI_Reduce(ps_rad.data(), power_spectrum_.data(), numShells_, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(count.data(), counts.data(), numShells_, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // normalize the power spectrum
        if (rank_ == 0)
        {
            T sum_ps_radial = std::accumulate(power_spectrum_.begin(), power_spectrum_.end(), 0.0);

            std::cout << "sum_ps_radial: " << sum_ps_radial << std::endl;

            #pragma omp parallel for
            for (int i = 0; i < numShells_; i++)
            {
                if (count[i] != 0)
                    power_spectrum_[i] = (power_spectrum_[i] * 4.0 * std::numbers::pi * std::pow(k_1d[i],2) )/ counts[i];
            }
        }
    }

    void setSimBox(T Lmin, T Lmax)
    {
        Lmin_ = Lmin;
        Lmax_ = Lmax;
    }

    void resize_comm_size(const int size)
    {
        send_disp.resize(size + 1, 0);
        send_count.resize(size, 0);
        recv_disp.resize(size + 1, 0);
        recv_count.resize(size, 0);

        vdataSender.resize(size);
    }

    T calculateDistance(T x, T y, T z, int i, int j, int k)
    {
        T xDistance = std::pow(x - x_[i], 2);
        T yDistance = std::pow(y - x_[j], 2);
        T zDistance = std::pow(z - x_[k], 2);
        return std::sqrt(xDistance + yDistance + zDistance);
    }

    inline int calculateRankFromMeshCoord(int i, int j, int k)
    {
        int xBox = i / inbox_.size[0];
        int yBox = j / inbox_.size[1];
        int zBox = k / inbox_.size[2];

        int rank = xBox + yBox * proc_grid_[0] + zBox * proc_grid_[0] * proc_grid_[1];

        return rank;
    }

    inline int calculateInboxIndexFromMeshCoord(int i, int j, int k)
    {
        int xBox = i % inbox_.size[0];
        int yBox = j % inbox_.size[1];
        int zBox = k % inbox_.size[2];

        int index = xBox + yBox * inbox_.size[0] + zBox * inbox_.size[0] * inbox_.size[1];

        return index;
    }

    std::tuple<int,int,int> calculateKeyIndices(KeyType key, int powerDim)
    {
        auto mesh_indices = cstone::decodeHilbert(key);
        unsigned divisor = std::pow(2, (21-powerDim));

        int meshCoordX_base = std::get<0>(mesh_indices)/divisor;
        int meshCoordY_base = std::get<1>(mesh_indices)/divisor;
        int meshCoordZ_base = std::get<2>(mesh_indices)/divisor;

        // std::cout << "key: " << key << " mesh indices: " << meshCoordX_base << " " << meshCoordY_base << " " << meshCoordZ_base << std::endl;
        return std::tie(meshCoordX_base, meshCoordY_base, meshCoordZ_base);
    }

private:
    heffte::box3d<> initInbox()
    {
        heffte::box3d<> all_indexes({0, 0, 0}, {gridDim_ - 1, gridDim_ - 1, gridDim_ - 1});

        proc_grid_ = heffte::proc_setup_min_surface(all_indexes, numRanks_);
        // print proc_grid
        std::cout << "rank = " << rank_ << " proc_grid: " << proc_grid_[0] << " " << proc_grid_[1] << " " << proc_grid_[2] << std::endl;

        // split all indexes across the processor grid, defines a set of boxes
        std::vector<heffte::box3d<>> all_boxes = heffte::split_world(all_indexes, proc_grid_);

        return all_boxes[rank_];
    }

    // Calculates the volume centers instead of starting with Lmin and adding deltaMesh
    void setCoordinates(T Lmin, T Lmax)
    {
        T deltaMesh = (Lmax - Lmin) / (gridDim_);
        T centerCoord = deltaMesh / 2;
        T startingCoord = Lmin + centerCoord;
        T displacement = inbox_.low[0] / gridDim_;

        #pragma omp parallel for
        for (int i = 0; i < inbox_.size[0]; i++)
        {
            x_[i] = (startingCoord + displacement) + deltaMesh * i;
        }
    }
};