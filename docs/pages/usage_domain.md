# cstone::Domain usage

The `cstone::Domain<KeyType, RealType, Accelerator>` manages distributed particles and halos.

## Construction
```cpp
#include "cstone/domain/domain.hpp"

using KeyType = uint64_t;
using Real    = double;
using Domain  = cstone::Domain<KeyType, Real, cstone::CpuTag>;

int rank = ...; int numRanks = ...;
unsigned bucketSize = 1000;           // global octree bucket size
unsigned bucketSizeFocus = 64;        // per-leaf focus bucket size
float theta = 1.0f;                   // MAC parameter
cstone::Box<Real> box(-0.5, 0.5, cstone::BoundaryType::periodic);

Domain domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);
```

## Sync workflow
```cpp
std::vector<Real> x, y, z, h, vx, vy, vz;
std::vector<KeyType> keys;
std::vector<Real> scratch1(x.size()), scratch2(x.size()), scratch3(x.size());

// Distribute particles, build trees, exchange halos, sort by SFC
// vx,vy,vz are distributed alongside coordinates

domain.sync(keys, x, y, z, h, std::tie(vx, vy, vz), std::tie(scratch1, scratch2, scratch3));

// Access assigned range
auto i0 = domain.startIndex();
auto i1 = domain.endIndex();
```

## Reapply sync to additional fields
```cpp
std::vector<Real> density(x.size());
std::vector<Real> sendBuf, recvBuf, ordering;

domain.reapplySync(std::tie(density), sendBuf, recvBuf, ordering);

domain.exchangeHalos(std::tie(density), sendBuf, recvBuf);
```

## Notes
- Arrays must retain sizes between consecutive `sync` calls.
- Use GPU `Accelerator` tags to enable device-side execution.
