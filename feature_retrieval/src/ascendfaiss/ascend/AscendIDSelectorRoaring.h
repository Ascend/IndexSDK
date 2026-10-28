/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCEND_ID_SELECTOR_ROARING_INCLUDED
#define ASCEND_ID_SELECTOR_ROARING_INCLUDED

#include <faiss/impl/IDSelector.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss
{
namespace ascend
{

/**
 * IDSelector wrapping a CRoaring bitmap (uint32 IDs). Frozen bytes must be
 * roaring_bitmap_frozen_serialize of SDK CRoaring 5.0.0. Does not own the
 * caller's bitmap/frozen bytes. Unaligned frozen buffers are copied so
 * is_member is O(1). live is roaring_bitmap_t* as const void*.
 */
struct IDSelectorRoaring : IDSelector
{
    size_t n = 0;
    const uint8_t *frozen = nullptr;
    const void *live = nullptr;

    IDSelectorRoaring(size_t nBytes, const uint8_t *frozenBytes);
    explicit IDSelectorRoaring(const void *rb);
    ~IDSelectorRoaring() override;
    IDSelectorRoaring(const IDSelectorRoaring &) = delete;
    IDSelectorRoaring &operator=(const IDSelectorRoaring &) = delete;
    IDSelectorRoaring(IDSelectorRoaring &&) = delete;
    IDSelectorRoaring &operator=(IDSelectorRoaring &&) = delete;

    bool is_member(idx_t id) const override;

   private:
    void InitFrozenView();

    std::vector<uint8_t> frozenAligned_;
    const void *frozenView_ = nullptr;
};

}  // namespace ascend
}  // namespace faiss

#endif
