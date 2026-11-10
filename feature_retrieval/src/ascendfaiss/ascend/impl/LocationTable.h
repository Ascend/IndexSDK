/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#ifndef ASCEND_LOCATION_TABLE_INCLUDED
#define ASCEND_LOCATION_TABLE_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace faiss
{
namespace ascend
{
// 融合 host 侧 id 定位信息的 8 字节定长记录。
//   listId      : coarse list 序号（当前最大 32768，uint16_t 足够）
//   deviceIndex : deviceList 中的逻辑下标（非 ACL 物理 device ID）
//   flags       : bit0 为 valid，其余预留
class Location
{
   public:
    // 字段位宽决定的布局上限：listId∈uint16_t 共 65536 个取值，deviceIndex∈uint8_t 共 256 个。
    static constexpr size_t kMaxListCount = static_cast<size_t>(UINT16_MAX) + 1;
    static constexpr size_t kMaxDeviceCount = static_cast<size_t>(UINT8_MAX) + 1;

    // listCount / deviceCount 是否落在字段可表示范围内（均需 > 0）。
    static bool listCountInRange(size_t listCount) { return listCount != 0 && listCount <= kMaxListCount; }
    static bool deviceCountInRange(size_t deviceCount) { return deviceCount != 0 && deviceCount <= kMaxDeviceCount; }

    Location() = default;
    Location(uint8_t deviceIndex, uint16_t listId, uint32_t slot)
        : slot_(slot), listId_(listId), deviceIndex_(deviceIndex), flags_(kValid)
    {
    }

    uint32_t slot() const { return slot_; }
    uint16_t listId() const { return listId_; }
    uint8_t deviceIndex() const { return deviceIndex_; }
    bool valid() const { return (flags_ & kValid) != 0; }

    // 写入定位字段并置 valid。
    void assign(uint8_t deviceIndex, uint16_t listId, uint32_t slot)
    {
        slot_ = slot;
        listId_ = listId;
        deviceIndex_ = deviceIndex;
        flags_ = kValid;
    }

    // 逻辑删除：仅清 valid 位，保留其余字段。
    void markInvalid() { flags_ &= static_cast<uint8_t>(~kValid); }

   private:
    static constexpr uint8_t kValid = 0x1;

    uint32_t slot_ = 0;
    uint16_t listId_ = 0;
    uint8_t deviceIndex_ = 0;
    uint8_t flags_ = 0;
};

static_assert(sizeof(Location) == 8, "Location must stay 8 bytes for chunked storage");

// 一次 writer 事务的映射增量。
// reserveForIds 物化无效槽；validateDelta 在发布前检查完整增量。
// 发布后 applyLocationDeltaNoFail 只写已存在槽，不分配、不抛异常。
struct LocationDelta
{
    std::vector<int64_t> erasedIds;                      // 清 valid
    std::vector<std::pair<int64_t, Location>> movedIds;  // 尾部回填 source ID 的新 slot
    std::vector<std::pair<int64_t, Location>> addedIds;  // 新增 ID 的完整 Location
};

// 按数据块稀疏索引；块内为排序稀疏槽，达到阈值后变为稠密数组。
// 只由串行 writer 访问，不属于 reader 快照，也不需要 COW/hazard。
class LocationTable
{
   public:
    static constexpr size_t kBlockSlots = 4096;
    static constexpr size_t kDenseThreshold = 1024;
    using BlockKey = uint64_t;

    // maxId >= 0：固定 id 上界；maxId < 0：无固定上界。
    // 块键和块内存都按需分配，离散大 id 不会各自分配 32 KiB。
    LocationTable(size_t deviceCount, size_t listCount, int64_t maxId = -1)
        : maxId_(maxId), deviceCount_(deviceCount), listCount_(listCount)
    {
        if (!Location::listCountInRange(listCount))
        {
            throw std::invalid_argument("LocationTable: listCount " + std::to_string(listCount) +
                                        " out of layout range (0, 65536]");
        }
        if (!Location::deviceCountInRange(deviceCount))
        {
            throw std::invalid_argument("LocationTable: deviceCount " + std::to_string(deviceCount) +
                                        " out of layout range (0, 256]");
        }
    }

    LocationTable(const LocationTable &) = delete;
    LocationTable &operator=(const LocationTable &) = delete;

    int64_t maxId() const { return maxId_; }
    size_t deviceCount() const { return deviceCount_; }
    size_t listCount() const { return listCount_; }

    // 返回有效 Location；未命中或已失效返回 nullopt。稀疏块内二分查找。
    std::optional<Location> lookupLocation(int64_t id) const
    {
        const Location *p = slotPtr(id);
        if (p == nullptr || !p->valid())
        {
            return std::nullopt;
        }
        return *p;
    }

    bool contains(int64_t id) const
    {
        const Location *p = slotPtr(id);
        return p != nullptr && p->valid();
    }

    // 写入一条 Location 并置 valid。只有 ensure/reserve 路径允许分配和升级。
    void setLocation(int64_t id, uint8_t deviceIndex, uint16_t listId, uint32_t slot)
    {
        checkId(id);
        checkFields(deviceIndex, listId);
        ensureSlot(id)->assign(deviceIndex, listId, slot);
    }

    // 清除 valid（逻辑删除）。在事务结束前不释放槽或数据块。
    void clearValid(int64_t id)
    {
        Location *p = mutableSlotPtr(id);
        if (p != nullptr)
        {
            p->markInvalid();
        }
    }

    // 先校验整个批次，再物化槽。分配失败可留无效预留槽，但有效映射不变。
    void reserveForIds(const std::vector<int64_t> &ids)
    {
        for (int64_t id : ids)
        {
            checkId(id);
        }
        for (int64_t id : ids)
        {
            (void)ensureSlot(id);
        }
    }

    // 发布前完整校验。moved 的 source ID 必须当前有效；added 必须已预留槽。
    // erase 后 move/add 同一 ID 允许，校验始终基于尚未提交的映射。
    void validateDelta(const LocationDelta &delta) const
    {
        for (int64_t id : delta.erasedIds)
        {
            checkId(id);
        }
        for (const auto &kv : delta.movedIds)
        {
            checkDeltaWrite(kv.first, kv.second, true);
        }
        for (const auto &kv : delta.addedIds)
        {
            checkDeltaWrite(kv.first, kv.second, false);
        }
    }

    // 发布后先做不分配的全量不变量检查，再 erase/move/add；错误直接 fail-fast。
    void applyLocationDeltaNoFail(const LocationDelta &delta) noexcept
    {
        for (int64_t id : delta.erasedIds)
        {
            if (!idInRange(id)) std::terminate();
        }
        for (const auto &kv : delta.movedIds)
        {
            if (!deltaWriteValid(kv.first, kv.second, true)) std::terminate();
        }
        for (const auto &kv : delta.addedIds)
        {
            if (!deltaWriteValid(kv.first, kv.second, false)) std::terminate();
        }
        for (int64_t id : delta.erasedIds)
        {
            clearValid(id);
        }
        for (const auto &kv : delta.movedIds)
        {
            writeExistingNoFail(kv.first, kv.second);
        }
        for (const auto &kv : delta.addedIds)
        {
            writeExistingNoFail(kv.first, kv.second);
        }
    }

    // 发布前收集；排序去重，不返回指针或可失效的槽令牌。
    static std::vector<BlockKey> collectTouchedBlockKeys(const std::vector<int64_t> &ids)
    {
        std::vector<BlockKey> keys;
        keys.reserve(ids.size());
        for (int64_t id : ids)
        {
            if (id < 0) throw std::out_of_range("LocationTable: negative touched id");
            keys.push_back(blockKey(id));
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    // 仅事务结束且全部 futures 已结束后调用。原地移除无效稀疏槽、删除空块；
    // 不分配、不降级、不 shrink，不能在 reserve 与最后一次 apply 之间调用。
    void cleanupTouchedBlocks(const std::vector<BlockKey> &keys) noexcept
    {
        for (BlockKey key : keys)
        {
            auto it = blocks_.find(key);
            if (it == blocks_.end()) continue;
            Block &block = it->second;
            bool empty = true;
            if (block.dense)
            {
                for (size_t i = 0; i < kBlockSlots; ++i)
                {
                    if (block.dense[i].valid())
                    {
                        empty = false;
                        break;
                    }
                }
            }
            else
            {
                block.sparse.erase(std::remove_if(block.sparse.begin(), block.sparse.end(),
                                                  [](const SparseEntry &entry) { return !entry.location.valid(); }),
                                   block.sparse.end());
                empty = block.sparse.empty();
            }
            if (empty) blocks_.erase(it);
        }
    }

    // Successful reset reconciliation leaves other devices authoritative, without allocation.
    void clearDevice(uint8_t deviceIndex) noexcept
    {
        for (auto &kv : blocks_)
        {
            Block &block = kv.second;
            if (block.dense)
            {
                for (size_t i = 0; i < kBlockSlots; ++i)
                {
                    Location &loc = block.dense[i];
                    if (loc.valid() && loc.deviceIndex() == deviceIndex) loc.markInvalid();
                }
            }
            else
            {
                for (auto &entry : block.sparse)
                {
                    Location &loc = entry.location;
                    if (loc.valid() && loc.deviceIndex() == deviceIndex) loc.markInvalid();
                }
            }
        }
    }

    void reset() { blocks_.clear(); }
    size_t allocatedBlockCount() const { return blocks_.size(); }

    size_t denseBlockCount() const
    {
        size_t count = 0;
        for (const auto &kv : blocks_) count += kv.second.dense != nullptr;
        return count;
    }

    // 块载荷字节数（不含 unordered_map 节点/桶）；稀疏槽按 capacity 而非 size 计。
    size_t memoryBytes() const
    {
        size_t bytes = 0;
        for (const auto &kv : blocks_)
        {
            bytes +=
                kv.second.dense ? kBlockSlots * sizeof(Location) : kv.second.sparse.capacity() * sizeof(SparseEntry);
        }
        return bytes;
    }

    // 冷路径遍历全部有效 ID；只扫描稀疏槽或已分配的稠密数组。
    template <typename Fn>
    void forEachValidId(Fn &&fn) const
    {
        for (const auto &kv : blocks_)
        {
            const int64_t base = static_cast<int64_t>(kv.first * kBlockSlots);
            if (kv.second.dense)
            {
                for (size_t offset = 0; offset < kBlockSlots; ++offset)
                {
                    if (kv.second.dense[offset].valid()) fn(base + static_cast<int64_t>(offset));
                }
            }
            else
            {
                for (const auto &entry : kv.second.sparse)
                {
                    if (entry.location.valid()) fn(base + entry.offset);
                }
            }
        }
    }

   private:
    struct SparseEntry
    {
        uint16_t offset;
        Location location;
    };

    struct Block
    {
        std::vector<SparseEntry> sparse;
        std::unique_ptr<Location[]> dense;
    };

    static BlockKey blockKey(int64_t id) { return static_cast<uint64_t>(id) / kBlockSlots; }
    static uint16_t blockOffset(int64_t id) { return static_cast<uint16_t>(static_cast<uint64_t>(id) % kBlockSlots); }

    bool idInRange(int64_t id) const noexcept { return id >= 0 && (maxId_ < 0 || id <= maxId_); }
    bool fieldsInRange(const Location &loc) const noexcept
    {
        return static_cast<size_t>(loc.deviceIndex()) < deviceCount_ && static_cast<size_t>(loc.listId()) < listCount_;
    }

    void checkId(int64_t id) const
    {
        if (!idInRange(id))
        {
            throw std::out_of_range("LocationTable: id " + std::to_string(id) + " out of allowed range [0, " +
                                    (maxId_ >= 0 ? std::to_string(maxId_) : std::string("unbounded")) + "]");
        }
    }

    void checkFields(uint8_t deviceIndex, uint16_t listId) const
    {
        if (static_cast<size_t>(deviceIndex) >= deviceCount_)
        {
            throw std::out_of_range("LocationTable: deviceIndex " + std::to_string(deviceIndex) +
                                    " out of configured deviceCount " + std::to_string(deviceCount_));
        }
        if (static_cast<size_t>(listId) >= listCount_)
        {
            throw std::out_of_range("LocationTable: listId " + std::to_string(listId) +
                                    " out of configured listCount " + std::to_string(listCount_));
        }
    }

    const Location *slotPtr(int64_t id) const noexcept
    {
        if (!idInRange(id)) return nullptr;
        auto it = blocks_.find(blockKey(id));
        if (it == blocks_.end()) return nullptr;
        const Block &block = it->second;
        const uint16_t offset = blockOffset(id);
        if (block.dense) return &block.dense[offset];
        auto entry =
            std::lower_bound(block.sparse.begin(), block.sparse.end(), offset,
                             [](const SparseEntry &candidate, uint16_t value) { return candidate.offset < value; });
        return entry != block.sparse.end() && entry->offset == offset ? &entry->location : nullptr;
    }

    Location *mutableSlotPtr(int64_t id) noexcept { return const_cast<Location *>(slotPtr(id)); }

    // 先构造新载荷再交换；任意 bad_alloc 都不会丢失或改写原有有效映射。
    Location *ensureSlot(int64_t id)
    {
        const BlockKey key = blockKey(id);
        const uint16_t offset = blockOffset(id);
        auto it = blocks_.find(key);
        if (it == blocks_.end())
        {
            Block block;
            block.sparse.push_back({offset, Location()});
            it = blocks_.emplace(key, std::move(block)).first;
            return &it->second.sparse.front().location;
        }
        Block &block = it->second;
        if (block.dense) return &block.dense[offset];
        auto entry =
            std::lower_bound(block.sparse.begin(), block.sparse.end(), offset,
                             [](const SparseEntry &candidate, uint16_t value) { return candidate.offset < value; });
        if (entry != block.sparse.end() && entry->offset == offset) return &entry->location;
        const size_t index = static_cast<size_t>(entry - block.sparse.begin());
        if (block.sparse.size() + 1 >= kDenseThreshold)
        {
            auto dense = std::make_unique<Location[]>(kBlockSlots);
            for (const auto &existing : block.sparse) dense[existing.offset] = existing.location;
            block.dense.swap(dense);
            std::vector<SparseEntry>().swap(block.sparse);
            return &block.dense[offset];
        }
        if (block.sparse.size() == block.sparse.capacity())
        {
            std::vector<SparseEntry> staged;
            staged.reserve(std::min(kDenseThreshold, std::max<size_t>(1, block.sparse.size() * 2)));
            staged.insert(staged.end(), block.sparse.begin(), block.sparse.end());
            staged.insert(staged.begin() + index, {offset, Location()});
            block.sparse.swap(staged);
        }
        else
        {
            block.sparse.insert(entry, {offset, Location()});
        }
        return &block.sparse[index].location;
    }

    void checkDeltaWrite(int64_t id, const Location &loc, bool requireValidSource) const
    {
        checkId(id);
        if (!loc.valid()) throw std::invalid_argument("LocationTable: delta Location is invalid");
        checkFields(loc.deviceIndex(), loc.listId());
        const Location *p = slotPtr(id);
        if (p == nullptr) throw std::logic_error("LocationTable: delta destination slot not reserved");
        if (requireValidSource && !p->valid())
        {
            throw std::logic_error("LocationTable: moved source ID is not valid");
        }
    }

    bool deltaWriteValid(int64_t id, const Location &loc, bool requireValidSource) const noexcept
    {
        if (!idInRange(id) || !loc.valid() || !fieldsInRange(loc)) return false;
        const Location *p = slotPtr(id);
        return p != nullptr && (!requireValidSource || p->valid());
    }

    // 兼容直接 UT 调用的 checked helper；提交路径不调用此 throwing helper。
    void writeExisting(int64_t id, const Location &loc)
    {
        checkDeltaWrite(id, loc, false);
        writeExistingNoFail(id, loc);
    }

    void writeExistingNoFail(int64_t id, const Location &loc) noexcept { *mutableSlotPtr(id) = loc; }

   private:
    int64_t maxId_;
    size_t deviceCount_;
    size_t listCount_;
    std::unordered_map<BlockKey, Block> blocks_;
};
}  // namespace ascend
}  // namespace faiss
#endif
