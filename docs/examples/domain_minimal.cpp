# Example: Minimal Domain usage

```cpp
#include <vector>
#include "cstone/domain/domain.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    using KeyType = uint64_t;
    using Real    = double;
    cstone::Box<Real> box(-0.5, 0.5, cstone::BoundaryType::periodic);

    unsigned bucket = 1024, bucketFocus = 64; float theta = 1.0f;
    cstone::Domain<KeyType, Real, cstone::CpuTag> domain(rank, nranks, bucket, bucketFocus, theta, box);

    std::vector<Real> x(1000), y(1000), z(1000), h(1000, 0.01);
    std::vector<Real> vx(1000), vy(1000), vz(1000);
    std::vector<KeyType> keys(1000);
    std::vector<Real> s1(1000), s2(1000), s3(1000);

    domain.sync(keys, x, y, z, h, std::tie(vx, vy, vz), std::tie(s1, s2, s3));

    MPI_Finalize();
    return 0;
}
```