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


#ifndef ASCEND_INDEX_IVF_INCLUDED
#define ASCEND_INDEX_IVF_INCLUDED

#include <faiss/Clustering.h>
#include "ascend/AscendIndex.h"

namespace faiss {
struct IndexIVF;
}  // namespace faiss

namespace faiss {
namespace ascend {
const int64_t IVF_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexIVFConfig : public AscendIndexConfig {
    inline AscendIndexIVFConfig() : AscendIndexConfig({ 0 }, IVF_DEFAULT_MEM), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline AscendIndexIVFConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline AscendIndexIVFConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize), useKmeansPP(false)
    {
        SetDefaultClusteringConfig();
    }

    inline void SetDefaultClusteringConfig()
    {
        // here we set a low # iterations because this is typically used
        // for large clusterings
        const int niter = 10;
        cp.niter = niter;
    }

    // Configuration for the coarse quantizer object
    AscendIndexConfig flatConfig;

    // whether to use kmeansPP
    bool useKmeansPP;
    
    // clustering parameters for trainQuantizer
    ClusteringParameters cp;
};

class AscendIndexIVFImpl;
class AscendIndexIVF : public AscendIndex {
public:

    AscendIndexIVF(int dims, faiss::MetricType metric, int nlist, AscendIndexIVFConfig config = AscendIndexIVFConfig());

    virtual ~AscendIndexIVF();

    // Returns the number of inverted lists we're managing
    int getNumLists() const;

    // Copy what we need from the CPU equivalent
    void copyFrom(const faiss::IndexIVF* index);

    // Copy what we have to the CPU equivalent
    void copyTo(faiss::IndexIVF* index) const;

    // Clears out all inverted lists, but retains the trained information
    void reset() override;

    // Sets the number of list probes per query
    virtual void setNumProbes(int nprobes);

    // Returns our current number of list probes per query
    int getNumProbes() const;

    // reserve memory for the database.
    void reserveMemory(size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;
    
    // return the list length of a particular list
    virtual uint32_t getListLength(int listId) const;

    // return the list codes of a particular list
    virtual void getListCodesAndIds(int listId, std::vector<uint8_t>& codes, std::vector<ascend_idx_t>& ids) const;

    // AscendIndex object is NON-copyable
    AscendIndexIVF(const AscendIndexIVF&) = delete;
    AscendIndexIVF& operator=(const AscendIndexIVF&) = delete;

protected:
    std::shared_ptr<AscendIndexIVFImpl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif
