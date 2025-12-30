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


#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>

#include "ascend/AscendIndexIVFSQ.h"
#include "ascend/custom/AscendIndexIVFSQT.h"
#include "ascend/AscendIndexFlat.h"
#include "ascend/AscendIndexSQ.h"
#include "ascend/AscendIndexInt8Flat.h"
#include "ascend/AscendCloner.h"

namespace faiss {
namespace ascend {
namespace {
// standard sq trained size is 2 * dim
const int SQ_VAL_SIZE = 2;
const size_t IVFSQ_FUZZY_PARAMS_COUNT = 3;
// Cloner specialized for Ascend -> CPU
struct ToCPUCloner : faiss::Cloner {
    Index *clone_Index(const Index *index) override;
    Index *clone_IndexInt8(const AscendIndexInt8 *index);
};

// Cloner specialized for CPU -> 1 Ascend
struct ToAscendCloner : faiss::Cloner, AscendClonerOptions {
    std::vector<int> devices;

    ToAscendCloner(std::initializer_list<int> devs, const AscendClonerOptions &options);
    ToAscendCloner(std::vector<int> devs, const AscendClonerOptions &options);
    Index *clone_Index(const Index *index);
    AscendIndexInt8 *clone_IndexInt8(const Index *index);

    Index *HandleIndexIDMap(const IndexIDMap *idm);
};

/**********************************************************
 * Cloning to CPU
 **********************************************************/
Index *ToCPUCloner::clone_Index(const Index *index)
{
    Index *res = nullptr;
    try {
        if (auto flat = dynamic_cast<const AscendIndexFlat *>(index)) {
            IndexFlat *flatCpu = new IndexFlat();
            auto tmp = new IndexIDMap(flatCpu);
            res = tmp;
            tmp->own_fields = true;
            flat->copyTo(tmp);
            return res;
        } else if (auto isqt = const_cast<AscendIndexIVFSQT *>(dynamic_cast<const AscendIndexIVFSQT *>(index))) {
            auto tmp = new IndexIVFScalarQuantizer();
            res = tmp;
            isqt->copyTo(tmp);
            return res;
        } else if (auto sq = dynamic_cast<const AscendIndexSQ *>(index)) {
            IndexScalarQuantizer *sqCpu = new IndexScalarQuantizer();
            auto tmp = new IndexIDMap(sqCpu);
            res = tmp;
            tmp->own_fields = true;
            sq->copyTo(tmp);
            return res;
        } else if (auto isq = dynamic_cast<const AscendIndexIVFSQ *>(index)) {
            auto tmp = new IndexIVFScalarQuantizer();
            res = tmp;
            isq->copyTo(tmp);
            return res;
        }
    } catch (const std::exception &e) {
        delete res;
        FAISS_THROW_FMT("clone from ascend to cpu failed, %s", e.what());
    }

    FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
}

Index *ToCPUCloner::clone_IndexInt8(const AscendIndexInt8 *index)
{
    if (auto int8flat = dynamic_cast<const AscendIndexInt8Flat *>(index)) {
        IndexScalarQuantizer *int8Cpu = new IndexScalarQuantizer();
        IndexIDMap *res = new IndexIDMap(int8Cpu);
        res->own_fields = true;
        try {
            int8flat->copyTo(res);
        } catch (const std::exception &e) {
            delete res;
            FAISS_THROW_FMT("clone from ascend to cpu failed, %s", e.what());
        }
        return res;
    }
    
    FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
}


/*
 * Cloning to Ascend
 */
ToAscendCloner::ToAscendCloner(std::initializer_list<int> devs, const AscendClonerOptions &options)
    : AscendClonerOptions(options), devices(devs)
{
    FAISS_THROW_IF_NOT_MSG(devs.size() != 0, "device list can not be empty!");
}

ToAscendCloner::ToAscendCloner(std::vector<int> devs, const AscendClonerOptions &options)
    : AscendClonerOptions(options), devices(devs)
{
    FAISS_THROW_IF_NOT_MSG(devs.size() != 0, "device list can not be empty!");
}

Index *ToAscendCloner::HandleIndexIDMap(const IndexIDMap *idm)
{
    if (dynamic_cast<const IndexFlat *>(idm->index)) {
        if (verbose) {
            printf("IndexIDMap/IndexFlat size %ld -> AscendIndexFlat, reserverVecs=%ld\n",
                idm->ntotal, reserveVecs);
        }

        AscendIndexFlatConfig config(devices, resourceSize);
        return new AscendIndexFlat(idm, config);
    } else if (dynamic_cast<const IndexScalarQuantizer *>(idm->index)) {
        if (verbose) {
            printf("IndexIDMap/IndexScalarQuantizer size %ld -> AscendIndexSQ, reserverVecs=%ld\n",
                idm->ntotal, reserveVecs);
        }

        AscendIndexSQConfig config(devices, resourceSize);
        config.slim = slim;
        config.filterable = filterable;
        config.dBlockSize = blockSize;
        return new AscendIndexSQ(idm, config);
    }
    
    FAISS_THROW_MSG("clone from cpu to ascend not supported for this type of Index");
}

Index *ToAscendCloner::clone_Index(const Index *index)
{
    if (auto flat = dynamic_cast<const IndexFlat *>(index)) {
        if (verbose) {
            printf("IndexFlat size %ld -> AscendIndexFlat, reserverVecs=%ld\n", flat->ntotal, reserveVecs);
        }

        AscendIndexFlatConfig config(devices, resourceSize);
        return new AscendIndexFlat(flat, config);
    } else if (auto sq = dynamic_cast<const IndexScalarQuantizer *>(index)) {
        if (verbose) {
            printf("IndexScalarQuantizer size %ld -> AscendIndexSQ, reserverVecs=%ld\n", sq->ntotal, reserveVecs);
        }

        AscendIndexSQConfig config(devices, resourceSize);
        config.slim = slim;
        config.filterable = filterable;
        config.dBlockSize = blockSize;
        return new AscendIndexSQ(sq, config);
    } else if (auto isq = dynamic_cast<const IndexIVFScalarQuantizer *>(index)) {
        if (verbose) {
            printf("IndexIVFScalarQuantizer size %ld -> AscendIndexIVFSQ, reserverVecs=%ld\n",
                isq->ntotal, reserveVecs);
        }
        size_t sqTrained = isq->sq.trained.size();
        size_t dim = isq->sq.d;

        if (sqTrained == dim * SQ_VAL_SIZE) {
            AscendIndexIVFSQConfig config(devices, resourceSize);
            return new AscendIndexIVFSQ(isq, config);
        }

        FAISS_THROW_IF_NOT_MSG(sqTrained >= IVFSQ_FUZZY_PARAMS_COUNT, "Invalid sqTrained.");
        int type = int(isq->sq.trained[sqTrained - IVFSQ_FUZZY_PARAMS_COUNT]);
        if (verbose) {
            printf("IndexIVFScalarQuantizer real Type is %d\n", type);
        }
        AscendIndexIVFSQTConfig config(devices, resourceSize);
        return new AscendIndexIVFSQT(isq, config);
    } else if (auto idm = dynamic_cast<const IndexIDMap *>(index)) {
        return HandleIndexIDMap(idm);
    }

    FAISS_THROW_MSG("clone from cpu to ascend not supported for this type of Index");
}

Int8IndexMode convertInt8IndexMode(uint32_t indexMode)
{
    if (indexMode == 1) {
        return Int8IndexMode::PIPE_SEARCH_MODE;
    } else if (indexMode == 2) { // 2 reps WITHOUT_NORM_MODE
        return Int8IndexMode::WITHOUT_NORM_MODE;
    }

    return Int8IndexMode::DEFAULT_MODE;
}

AscendIndexInt8 *ToAscendCloner::clone_IndexInt8(const Index *index)
{
    if (auto idm = dynamic_cast<const IndexIDMap *>(index)) {
        if (verbose) {
            printf("IndexIDMap/IndexScalarQuantizer size %ld -> AscendIndexInt8Flat, reserverVecs=%ld\n",
                idm->ntotal, reserveVecs);
        }

        AscendIndexInt8FlatConfig config;
        config.deviceList = devices;
        config.resourceSize = resourceSize;
        config.dIndexMode = convertInt8IndexMode(indexMode);
        return new AscendIndexInt8Flat(idm, config);
    }
    
    FAISS_THROW_MSG("clone from ascend to cpu not supported for this type of Index");
}
}
faiss::Index *index_ascend_to_cpu(const faiss::Index *ascendIndex)
{
    FAISS_THROW_IF_NOT_MSG(ascendIndex, "ascendIndex is nullptr.");
    ToCPUCloner cl;
    return cl.clone_Index(ascendIndex);
}

faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *ascendIndex)
{
    FAISS_THROW_IF_NOT_MSG(ascendIndex, "ascendIndex is nullptr.");
    ToCPUCloner cl;
    return cl.clone_IndexInt8(ascendIndex);
}

faiss::Index *index_cpu_to_ascend(std::initializer_list<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    FAISS_THROW_IF_NOT_MSG(index, "index is nullptr.");
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_Index(index);
}

faiss::Index *index_cpu_to_ascend(std::vector<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    FAISS_THROW_IF_NOT_MSG(index, "index is nullptr.");
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_Index(index);
}

AscendIndexInt8 *index_int8_cpu_to_ascend(
    std::initializer_list<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    FAISS_THROW_IF_NOT_MSG(index, "index is nullptr.");
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_IndexInt8(index);
}

AscendIndexInt8 *index_int8_cpu_to_ascend(
    std::vector<int> devices,
    const faiss::Index *index, const AscendClonerOptions *options)
{
    FAISS_THROW_IF_NOT_MSG(index, "index is nullptr.");
    AscendClonerOptions defaults;
    ToAscendCloner cl(devices, options ? *options : defaults);
    return cl.clone_IndexInt8(index);
}
}  // namespace ascend
}  // namespace faiss
