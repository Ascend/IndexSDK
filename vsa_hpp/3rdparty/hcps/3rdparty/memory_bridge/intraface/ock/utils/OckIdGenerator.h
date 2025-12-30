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


#ifndef OCK_MEMORY_BRIDGE_OCK_ID_GENERATOR_H
#define OCK_MEMORY_BRIDGE_OCK_ID_GENERATOR_H
#include <cstdint>
#include <memory>
#include <limits>
#include <bitset>
namespace ock {
namespace utils {

template <uint32_t MaxCount>
class OckIdGenerator {
public:
    virtual ~OckIdGenerator() noexcept = default;
    OckIdGenerator(void) : curId(0)
    {}
    // 该函数有性能要求，需要做性能看护
    std::pair<bool, uint32_t> NewId(void)
    {
        const uint32_t maxTryTimes = 2;
        uint32_t tryTimes = 0;
        while (true) {
            if (curId == MaxCount) {
                curId = 0;
                tryTimes++;
                if (tryTimes >= maxTryTimes) {
                    return std::make_pair(false, curId);
                }
            }
            if (!datas.test(curId)) {
                datas.set(curId);
                return std::make_pair(true, curId++);
            }
            curId++;
        }
    }
    void DelId(uint32_t idValue)
    {
        datas.reset(idValue);
    }
    uint32_t UsedCount(void) const
    {
        return static_cast<uint32_t>(datas.count());
    }

private:
    uint32_t curId;
    std::bitset<MaxCount> datas{};
};
}  // namespace utils
}  // namespace ock
#endif