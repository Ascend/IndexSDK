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


#ifndef ASCEND_INDEX_BINARY_FLAT_INCLUDED
#define ASCEND_INDEX_BINARY_FLAT_INCLUDED

#include <faiss/MetaIndexes.h>
#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>

namespace faiss {
namespace ascend {
class AscendIndexBinaryFlatImpl;

constexpr int64_t BINARY_FLAT_DEFAULT_MEM = 0x40000000; /* 1GB */
struct AscendIndexBinaryFlatConfig {
    AscendIndexBinaryFlatConfig() = default;

    AscendIndexBinaryFlatConfig(std::initializer_list<int> devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources) {}

    AscendIndexBinaryFlatConfig(std::vector<int> devices, int64_t resources = BINARY_FLAT_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources) {}

    // Ascend devices mask on which the index is resident
    std::vector<int> deviceList { 0 };
    int64_t resourceSize = BINARY_FLAT_DEFAULT_MEM;
};

class AscendIndexBinaryFlat : public faiss::IndexBinary {
public:
    /* Construct from a pre-existing faiss::IndexBinaryFlat instance */
    AscendIndexBinaryFlat(const faiss::IndexBinaryFlat *index,
        AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);
    AscendIndexBinaryFlat(const faiss::IndexBinaryIDMap *index,
        AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(), bool usedFloat = false);

    /* Construct an empty instance that can be added to */
    AscendIndexBinaryFlat(int dims, AscendIndexBinaryFlatConfig config = AscendIndexBinaryFlatConfig(),
        bool usedFloat = false);

    /* * Add vectors with default ids to device
     *
     * @param x           input vectors to add, size n * dims / 8
     */
    void add(idx_t n, const uint8_t *x) override;

    void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids) override;

    size_t remove_ids(const faiss::IDSelector &sel) override;

    void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels,
        const SearchParameters *params = nullptr) const override;

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;
    void reset() override;

    /* Initialize ourselves from the given CPU index; will overwrite all data in ourselves */
    void copyFrom(const faiss::IndexBinaryFlat *index);
    void copyFrom(const faiss::IndexBinaryIDMap *index);

    /* Copy ourselves to the given CPU index; will overwrite all data in the index instance */
    void copyTo(faiss::IndexBinaryFlat *index) const;
    void copyTo(faiss::IndexBinaryIDMap *index) const;

    static void setRemoveFast(bool removeFast);

    /* AscendIndexBinaryFlat object is NON-copyable */
    AscendIndexBinaryFlat(const AscendIndexBinaryFlat &) = delete;
    AscendIndexBinaryFlat &operator = (const AscendIndexBinaryFlat &) = delete;

    virtual ~AscendIndexBinaryFlat() = default;

private:
    std::shared_ptr<AscendIndexBinaryFlatImpl> impl_; /* internal implementation */
};
} /* ascend */
} /* faiss */
#endif /* ASCEND_INDEX_BINARY_FLAT_INCLUDED */