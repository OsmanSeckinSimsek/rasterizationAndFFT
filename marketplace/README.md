# AI Compute Marketplace

This module contains the first building block for an AI compute marketplace:
a deterministic order book that matches GPU-hour supply with GPU-hour demand.

Resource owners submit supply orders:

- `accountId`: owner/provider identifier
- `resourceType`: GPU class, for example `nvidia-a100` or `nvidia-h100`
- `gpuSeconds`: amount of compute offered
- `limitPriceCentsPerGpuHour`: minimum price requested by the owner

Compute users submit demand orders:

- `accountId`: user/customer identifier
- `resourceType`: required GPU class
- `gpuSeconds`: amount of compute requested
- `limitPriceCentsPerGpuHour`: maximum price the user is willing to pay

## Matching rules

Orders only match when they target the same `resourceType`.

Demand matches the cheapest eligible supply first. Supply matches the highest
eligible demand first. Orders with the same price use first-in, first-out
priority. Partial fills are supported: any unfilled quantity remains on the
book.

A trade is produced when:

```text
demand bid price >= supply ask price
```

The execution price is the resting order's limit price, which mirrors common
exchange order-book behavior and makes the result independent of the incoming
order's more aggressive limit.

## Scope

The module intentionally does not include networking, authentication, billing,
persistence, or cluster scheduling. Those can be layered above the order book:

```text
API / persistence / billing
        |
marketplace::OrderBook
        |
job scheduler / MPI launcher / GPU worker fleet
```

The current implementation is an in-memory C++20 component suitable for unit
testing the matching contract before introducing service infrastructure.
