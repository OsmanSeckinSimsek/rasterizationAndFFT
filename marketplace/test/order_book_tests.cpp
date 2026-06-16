#include "marketplace/order_book.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "gtest/gtest.h"

namespace marketplace
{
namespace
{

constexpr std::int64_t oneGpuHour = 3600;

OrderRequest supply(std::string owner, std::string resource, std::int64_t gpuSeconds, std::int64_t centsPerHour)
{
    return {OrderSide::Supply, std::move(owner), std::move(resource), gpuSeconds, centsPerHour};
}

OrderRequest demand(std::string user, std::string resource, std::int64_t gpuSeconds, std::int64_t centsPerHour)
{
    return {OrderSide::Demand, std::move(user), std::move(resource), gpuSeconds, centsPerHour};
}

TEST(OrderBook, DemandConsumesCheapestSupplyFirst)
{
    OrderBook book;
    EXPECT_TRUE(book.submit(supply("owner-a", "nvidia-a100", oneGpuHour, 300)).empty());
    EXPECT_TRUE(book.submit(supply("owner-b", "nvidia-a100", oneGpuHour, 250)).empty());

    const auto trades = book.submit(demand("user-1", "nvidia-a100", oneGpuHour + oneGpuHour / 2, 400));

    ASSERT_EQ(trades.size(), 2);
    EXPECT_EQ(trades[0].resourceOwnerId, "owner-b");
    EXPECT_EQ(trades[0].computeUserId, "user-1");
    EXPECT_EQ(trades[0].gpuSeconds, oneGpuHour);
    EXPECT_EQ(trades[0].priceCentsPerGpuHour, 250);
    EXPECT_EQ(trades[1].resourceOwnerId, "owner-a");
    EXPECT_EQ(trades[1].gpuSeconds, oneGpuHour / 2);
    EXPECT_EQ(trades[1].priceCentsPerGpuHour, 300);

    const auto remainingSupply = book.supplyOrders("nvidia-a100");
    ASSERT_EQ(remainingSupply.size(), 1);
    EXPECT_EQ(remainingSupply[0].accountId, "owner-a");
    EXPECT_EQ(remainingSupply[0].remainingGpuSeconds, oneGpuHour / 2);
    EXPECT_TRUE(book.demandOrders("nvidia-a100").empty());
}

TEST(OrderBook, EqualPriceSupplyUsesFifoPriority)
{
    OrderBook book;
    EXPECT_TRUE(book.submit(supply("owner-a", "nvidia-a100", oneGpuHour, 300)).empty());
    EXPECT_TRUE(book.submit(supply("owner-b", "nvidia-a100", oneGpuHour, 300)).empty());

    const auto trades = book.submit(demand("user-1", "nvidia-a100", oneGpuHour + 1, 300));

    ASSERT_EQ(trades.size(), 2);
    EXPECT_EQ(trades[0].resourceOwnerId, "owner-a");
    EXPECT_EQ(trades[0].gpuSeconds, oneGpuHour);
    EXPECT_EQ(trades[1].resourceOwnerId, "owner-b");
    EXPECT_EQ(trades[1].gpuSeconds, 1);
}

TEST(OrderBook, SupplyConsumesHighestDemandFirst)
{
    OrderBook book;
    EXPECT_TRUE(book.submit(demand("user-low", "nvidia-h100", oneGpuHour, 200)).empty());
    EXPECT_TRUE(book.submit(demand("user-high-a", "nvidia-h100", oneGpuHour / 2, 350)).empty());
    EXPECT_TRUE(book.submit(demand("user-high-b", "nvidia-h100", oneGpuHour, 350)).empty());

    const auto trades = book.submit(supply("owner-1", "nvidia-h100", oneGpuHour, 100));

    ASSERT_EQ(trades.size(), 2);
    EXPECT_EQ(trades[0].computeUserId, "user-high-a");
    EXPECT_EQ(trades[0].gpuSeconds, oneGpuHour / 2);
    EXPECT_EQ(trades[0].priceCentsPerGpuHour, 350);
    EXPECT_EQ(trades[1].computeUserId, "user-high-b");
    EXPECT_EQ(trades[1].gpuSeconds, oneGpuHour / 2);
    EXPECT_EQ(trades[1].priceCentsPerGpuHour, 350);

    const auto remainingDemand = book.demandOrders("nvidia-h100");
    ASSERT_EQ(remainingDemand.size(), 2);
    EXPECT_EQ(remainingDemand[0].accountId, "user-high-b");
    EXPECT_EQ(remainingDemand[0].remainingGpuSeconds, oneGpuHour / 2);
    EXPECT_EQ(remainingDemand[1].accountId, "user-low");
    EXPECT_EQ(remainingDemand[1].remainingGpuSeconds, oneGpuHour);
}

TEST(OrderBook, UncrossedOrdersRemainOnBook)
{
    OrderBook book;
    EXPECT_TRUE(book.submit(supply("owner-1", "nvidia-a100", oneGpuHour, 500)).empty());

    const auto trades = book.submit(demand("user-1", "nvidia-a100", oneGpuHour, 400));

    EXPECT_TRUE(trades.empty());
    ASSERT_EQ(book.supplyOrders("nvidia-a100").size(), 1);
    ASSERT_EQ(book.demandOrders("nvidia-a100").size(), 1);
}

TEST(OrderBook, ResourceTypesAreMatchedIndependently)
{
    OrderBook book;
    EXPECT_TRUE(book.submit(supply("owner-a100", "nvidia-a100", oneGpuHour, 100)).empty());

    const auto trades = book.submit(demand("user-h100", "nvidia-h100", oneGpuHour, 1000));

    EXPECT_TRUE(trades.empty());
    EXPECT_EQ(book.supplyOrders("nvidia-a100").size(), 1);
    EXPECT_EQ(book.demandOrders("nvidia-h100").size(), 1);
}

TEST(OrderBook, RejectsInvalidOrders)
{
    OrderBook book;

    EXPECT_THROW(book.submit(supply("", "nvidia-a100", oneGpuHour, 100)), std::invalid_argument);
    EXPECT_THROW(book.submit(supply("owner-1", "", oneGpuHour, 100)), std::invalid_argument);
    EXPECT_THROW(book.submit(supply("owner-1", "nvidia-a100", 0, 100)), std::invalid_argument);
    EXPECT_THROW(book.submit(supply("owner-1", "nvidia-a100", oneGpuHour, -1)), std::invalid_argument);
}

} // namespace
} // namespace marketplace
