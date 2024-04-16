#include <vector>
#include <limits>
#include "heffte.h"
#include "cstone/domain/domain.hpp"

using KeyType = uint64_t;

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

    heffte::box3d<> inbox_;
    // coordinate centers in the mesh
    std::vector<T>  x_;
    std::vector<T>  y_;
    std::vector<T>  z_;
    // corresponding velocities in the mesh
    std::vector<T>  velX_;
    std::vector<T>  velY_;
    std::vector<T>  velZ_;
    // particle's distance to mesh point
    std::vector<T>  distance_;
    std::vector<T>  power_spectrum_;

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
        size_t inboxSize = static_cast<size_t>(inbox_.size[0]) * static_cast<size_t>(inbox_.size[1]) *
                           static_cast<size_t>(inbox_.size[2]);
        velX_.resize(inboxSize);
        velY_.resize(inboxSize);
        velZ_.resize(inboxSize);
        x_.resize(inbox_.size[0]);
        y_.resize(inbox_.size[1]);
        z_.resize(inbox_.size[2]);
        distance_.resize(inboxSize);
        power_spectrum_.resize(numShells);

        std::fill(distance_.begin(), distance_.end(), std::numeric_limits<T>::infinity());

        setCoordinates(Lmin_, Lmax_);
    }

    void setSimBox(T Lmin, T Lmax)
    {
        Lmin_ = Lmin;
        Lmax_ = Lmax;
    }

    // Calculates the volume centers instead of starting with Lmin and adding deltaMesh
    void setCoordinates(T Lmin, T Lmax)
    {
        T deltaMesh = (Lmax - Lmin) / (gridDim_);
        T centerCoord = deltaMesh / 2;
        T startingCoord = Lmin + centerCoord;

        T displacement = inbox_.low[0] / gridDim_;
        for (int i = 0; i < inbox_.size[0]; i++)
        {
            x_[i] = (startingCoord + displacement) + deltaMesh * i;
        }

        displacement = inbox_.low[1] / gridDim_;
        for (int i = 0; i < inbox_.size[1]; i++)
        {
            y_[i] = (startingCoord + displacement) + deltaMesh * i;
        }

        displacement = inbox_.low[2] / gridDim_;
        for (int i = 0; i < inbox_.size[2]; i++)
        {
            z_[i] = (startingCoord + displacement) + deltaMesh * i;
        }
    }

    void rasterize_using_cornerstone(std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                    std::vector<T> z, std::vector<T> vx, std::vector<T> vy,
                                    std::vector<T> vz,int powerDim)
    {
        // int powerDim = std::ceil(std::log(simDim)/std::log(2));

        // iterate over the the mesh volumes that belong to this rank
        // #pragma omp parallel for
        for (int i = inbox_.low[0]; i < inbox_.high[0] - 1; i++)
        {
            // std::cout << "index: " << i << std::endl;
            unsigned iSFC_low = i * std::pow(2, (21-powerDim));
            unsigned iSFC_up = (i+1) * std::pow(2, (21-powerDim));
            for (int j = inbox_.low[1]; j < inbox_.high[1] - 1; j++)
            {
                unsigned jSFC_low = j * std::pow(2, (21-powerDim));
                unsigned jSFC_up = (j+1) * std::pow(2, (21-powerDim));
                for (int k = inbox_.low[2]; k < inbox_.high[2] - 1; k++)
                {
                    unsigned kSFC_low = k * std::pow(2, (21-powerDim));
                    unsigned kSFC_up = (k+1) * std::pow(2, (21-powerDim));

                    KeyType lowerKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(iSFC_low, jSFC_low, kSFC_low);
                    KeyType upperKey = cstone::iSfcKey<cstone::SfcKind<KeyType>>(iSFC_up, jSFC_up, kSFC_up);

                    unsigned level = cstone::commonPrefix(lowerKey, upperKey) / 3;
                    KeyType lowerBound = cstone::enclosingBoxCode(lowerKey, level);
                    KeyType upperBound =  lowerBound + cstone::nodeRange<KeyType>(level);

                    auto itlow = std::lower_bound(keys.begin(), keys.end(), lowerBound);
                    auto itup = std::upper_bound(keys.begin(), keys.end(), upperBound);
                    std::cout << "index: " << std::distance(itlow, itup) << std::endl;

                    size_t min_distance_index = 0;
                    T min_distance = std::numeric_limits<T>::infinity();

                    // iterate from itlow to itup to find the min distance to the center of the volume
                    // std::cout << "i: " << i << " j: " << j << " k: " << k << "low: " << itlow << "high: " << itup<< std::endl;
                    for (auto it = itlow; it != itup; it++)
                    {
                        // std::cout << "key: " << *it << std::endl;
                        auto index = std::distance(keys.begin(), it);
                        // std::cout << "index: " << index << std::endl;

                        T xDistance = std::pow(x[index] - x_[i], 2);
                        T yDistance = std::pow(y[index] - y_[j], 2);
                        T zDistance = std::pow(z[index] - z_[k], 2);
                        T distance  = std::sqrt(xDistance + yDistance + zDistance);
                        if (distance < min_distance)
                        {
                            min_distance = distance;
                            min_distance_index = index;
                        }
                    }

                    // min distance found, assign the velocity to the mesh volume
                    if (min_distance != std::numeric_limits<T>::infinity())
                    {
                        size_t velIndex = (i * inbox_.size[1] + j) * inbox_.size[2] + k;
                        velX_[velIndex] = vx[min_distance_index];
                        velY_[velIndex] = vy[min_distance_index];
                        velZ_[velIndex] = vz[min_distance_index];
                    }
                }
            }
        }

        // if the volume belongs to another rank, transfer the info
    }

    // Placeholder, needs to be implemented mapping cornerstone tree to the mesh
    void assign_velocities_to_mesh(T* xpos, T* ypos, T* zpos, T* vx, T* vy, T* vz,
                                   size_t simDim, size_t gridDim)
    {
        size_t         simDim3 = simDim * simDim * simDim;
        std::vector<T> mesh(gridDim);

        T Lmin      = -0.5;
        T Lmax      = 0.5;
        T deltaMesh = (Lmax - Lmin) / (gridDim - 1);

        for (size_t i = 0; i < gridDim; i++)
        {
            mesh[i] = Lmin + i * deltaMesh;
        }

        for (size_t i = 0; i < gridDim; i++)
        {
            std::cout << "i: " << i << "\n";
            for (size_t j = 0; j < gridDim; j++)
            {
                for (size_t k = 0; k < gridDim; k++)
                {
                    T      min_distance = std::numeric_limits<T>::infinity();
                    int    min_index    = -1;
                    size_t gridIndex    = (i * gridDim + j) * gridDim + k;

                    for (size_t p = 0; p < simDim3; p++)
                    {
                        T xDistance = std::pow(xpos[p] - mesh[i], 2);
                        T yDistance = std::pow(ypos[p] - mesh[j], 2);
                        T zDistance = std::pow(zpos[p] - mesh[k], 2);
                        T distance  = xDistance + yDistance + zDistance;

                        if (distance < min_distance)
                        {
                            min_distance = distance;
                            min_index    = p;
                        }
                    }

                    velX_[gridIndex] = vx[min_index];
                    velY_[gridIndex] = vy[min_index];
                    velZ_[gridIndex] = vz[min_index];
                }
            }
        }
    }

    void calculate_power_spectrum()
    {
        calculate_fft();

        std::vector<T> freqVelo(velX_.size());

        // calculate the modulus of the velocity frequencies
        for (size_t i = 0; i < velX_.size(); i++)
        {
            freqVelo[i] = velX_[i] + velY_[i] + velZ_[i];
        }

        // perform spherical averaging
        perform_spherical_averaging(freqVelo.data());
    }

    void calculate_fft()
    {
        heffte::box3d<> outbox   = inbox_;
        size_t          meshSize = gridDim_ * gridDim_ * gridDim_;

        // change fftw depending on the configuration into cufft or rocmfft
        heffte::fft3d<heffte::backend::fftw> fft(inbox_, outbox, MPI_COMM_WORLD);

        std::vector<std::complex<T>> output(fft.size_outbox());

        // divide the fft.forward results by the mesh size as the first step of normalization
        fft.forward(velX_.data(), output.data());

        for (size_t i = 0; i < velX_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velX_[i] = out * out;
        }

        fft.forward(velY_.data(), output.data());

        for (size_t i = 0; i < velY_.size(); i++)
        {
            T out    = abs(output.at(i)) / meshSize;
            velY_[i] = out * out;
        }

        fft.forward(velZ_.data(), output.data());

        for (size_t i = 0; i < velZ_.size(); i++)
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
        std::vector<T> k_values(gridDim_);
        std::vector<T> k_1d(gridDim_);
        std::vector<T> ps_rad(numShells_);
        std::vector<int> count(k_1d.size());

        fftfreq(k_values, gridDim_, 1.0 / gridDim_);

        for (int i = 0; i < gridDim_; i++)
        {
            k_1d[i] = std::abs(k_values[i]);
        }

        // iterate over the ps array and assign the values to the correct radial bin
        for (int i = 0; i < inbox_.size[2]; i++) // slow heffte order
        {
            for (int j = 0; j < inbox_.size[1]; j++) // mid heffte order
            {
                for (int k = 0; k < inbox_.size[0]; k++) // fast heffte order
                {
                    size_t freq_index = (i * inbox_.size[1] + j) * inbox_.size[0] + k;

                    // Calculate the k indices with respect to the global mesh
                    size_t         k_index_i = i + inbox_.low[2];
                    size_t         k_index_j = j + inbox_.low[1];
                    size_t         k_index_k = k + inbox_.low[0];
                    T              kdist     = std::sqrt(k_values[k_index_i] * k_values[k_index_i] +
                                                         k_values[k_index_j] * k_values[k_index_j] +
                                                         k_values[k_index_k] * k_values[k_index_k]);
                    std::vector<T> k_dif(gridDim_);
                    for (int kind = 0; kind < gridDim_; kind++)
                    {
                        k_dif[kind] = std::abs(k_1d[kind] - kdist);
                    }
                    auto   it      = std::min_element(std::begin(k_dif), std::end(k_dif));
                    size_t k_index = std::distance(std::begin(k_dif), it);

                    ps_rad[k_index] += ps[freq_index];
                    count[k_index]++;
                }
            }
        }

        MPI_Reduce(ps_rad.data(), power_spectrum_.data(), numShells_, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // normalize the power spectrum
        if (rank_ == 0)
        {
            T sum_ps_radial = std::accumulate(power_spectrum_.begin(), power_spectrum_.end(), 0.0);

            std::cout << "sum_ps_radial: " << sum_ps_radial << std::endl;

            for (int i = 0; i < numShells_; i++)
            {
                if (count[i] != 0)
                    power_spectrum_[i] = (power_spectrum_[i] * 4.0 * std::numbers::pi * std::pow(k_1d[i],2) )/ count[i];
            }
        }
    }

private:
    heffte::box3d<> initInbox()
    {
        heffte::box3d<> all_indexes({0, 0, 0}, {gridDim_ - 1, gridDim_ - 1, gridDim_ - 1});

        std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(all_indexes, numRanks_);

        // split all indexes across the processor grid, defines a set of boxes
        std::vector<heffte::box3d<>> all_boxes = heffte::split_world(all_indexes, proc_grid);

        return all_boxes[rank_];
    }
};