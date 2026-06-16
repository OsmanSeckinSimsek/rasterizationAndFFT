#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace marketplace
{

enum class OrderSide
{
    Supply,
    Demand
};

struct OrderRequest
{
    OrderSide side{OrderSide::Supply};
    std::string accountId;
    std::string resourceType;
    std::int64_t gpuSeconds{0};
    std::int64_t limitPriceCentsPerGpuHour{0};
};

struct Order
{
    std::uint64_t id{0};
    OrderSide side{OrderSide::Supply};
    std::string accountId;
    std::string resourceType;
    std::int64_t remainingGpuSeconds{0};
    std::int64_t limitPriceCentsPerGpuHour{0};
    std::uint64_t sequence{0};
};

struct Trade
{
    std::uint64_t supplyOrderId{0};
    std::uint64_t demandOrderId{0};
    std::string resourceOwnerId;
    std::string computeUserId;
    std::string resourceType;
    std::int64_t gpuSeconds{0};
    std::int64_t priceCentsPerGpuHour{0};
};

class OrderBook
{
public:
    std::vector<Trade> submit(OrderRequest request)
    {
        validate(request);

        Order incoming{nextOrderId_++, request.side, std::move(request.accountId), std::move(request.resourceType),
                       request.gpuSeconds, request.limitPriceCentsPerGpuHour, nextSequence_++};

        std::vector<Trade> trades;
        auto& opposite = sideFor(oppositeSide(incoming.side));

        while (incoming.remainingGpuSeconds > 0)
        {
            auto match = bestMatch(incoming, opposite);
            if (match == opposite.end()) { break; }

            const std::int64_t matchedSeconds =
                std::min(incoming.remainingGpuSeconds, match->remainingGpuSeconds);
            trades.push_back(makeTrade(incoming, *match, matchedSeconds));

            incoming.remainingGpuSeconds -= matchedSeconds;
            match->remainingGpuSeconds -= matchedSeconds;

            if (match->remainingGpuSeconds == 0) { opposite.erase(match); }
        }

        if (incoming.remainingGpuSeconds > 0) { sideFor(incoming.side).push_back(std::move(incoming)); }

        return trades;
    }

    std::vector<Order> supplyOrders(const std::string& resourceType = "") const
    {
        return snapshot(supplyOrders_, resourceType);
    }

    std::vector<Order> demandOrders(const std::string& resourceType = "") const
    {
        return snapshot(demandOrders_, resourceType);
    }

private:
    using Orders = std::list<Order>;

    static void validate(const OrderRequest& request)
    {
        if (request.accountId.empty()) { throw std::invalid_argument("accountId must not be empty"); }
        if (request.resourceType.empty()) { throw std::invalid_argument("resourceType must not be empty"); }
        if (request.gpuSeconds <= 0) { throw std::invalid_argument("gpuSeconds must be positive"); }
        if (request.limitPriceCentsPerGpuHour < 0)
        {
            throw std::invalid_argument("limitPriceCentsPerGpuHour must be non-negative");
        }
    }

    static OrderSide oppositeSide(OrderSide side)
    {
        return side == OrderSide::Supply ? OrderSide::Demand : OrderSide::Supply;
    }

    Orders& sideFor(OrderSide side)
    {
        return side == OrderSide::Supply ? supplyOrders_ : demandOrders_;
    }

    static bool canMatch(const Order& incoming, const Order& resting)
    {
        if (incoming.resourceType != resting.resourceType) { return false; }

        if (incoming.side == OrderSide::Demand)
        {
            return resting.limitPriceCentsPerGpuHour <= incoming.limitPriceCentsPerGpuHour;
        }

        return incoming.limitPriceCentsPerGpuHour <= resting.limitPriceCentsPerGpuHour;
    }

    static Orders::iterator bestMatch(const Order& incoming, Orders& restingOrders)
    {
        Orders::iterator best = restingOrders.end();

        for (auto it = restingOrders.begin(); it != restingOrders.end(); ++it)
        {
            if (!canMatch(incoming, *it)) { continue; }
            if (best == restingOrders.end() || hasBetterPriority(*it, *best)) { best = it; }
        }

        return best;
    }

    static bool hasBetterPriority(const Order& candidate, const Order& incumbent)
    {
        if (candidate.side == OrderSide::Supply)
        {
            if (candidate.limitPriceCentsPerGpuHour != incumbent.limitPriceCentsPerGpuHour)
            {
                return candidate.limitPriceCentsPerGpuHour < incumbent.limitPriceCentsPerGpuHour;
            }
        }
        else if (candidate.limitPriceCentsPerGpuHour != incumbent.limitPriceCentsPerGpuHour)
        {
            return candidate.limitPriceCentsPerGpuHour > incumbent.limitPriceCentsPerGpuHour;
        }

        return candidate.sequence < incumbent.sequence;
    }

    static Trade makeTrade(const Order& incoming, const Order& resting, std::int64_t matchedSeconds)
    {
        const Order& supply = incoming.side == OrderSide::Supply ? incoming : resting;
        const Order& demand = incoming.side == OrderSide::Demand ? incoming : resting;

        return Trade{supply.id,
                     demand.id,
                     supply.accountId,
                     demand.accountId,
                     supply.resourceType,
                     matchedSeconds,
                     resting.limitPriceCentsPerGpuHour};
    }

    static std::vector<Order> snapshot(const Orders& orders, const std::string& resourceType)
    {
        std::vector<Order> result;
        for (const Order& order : orders)
        {
            if (resourceType.empty() || order.resourceType == resourceType) { result.push_back(order); }
        }

        std::sort(result.begin(), result.end(), hasBetterPriority);
        return result;
    }

    Orders supplyOrders_;
    Orders demandOrders_;
    std::uint64_t nextOrderId_{1};
    std::uint64_t nextSequence_{1};
};

} // namespace marketplace
