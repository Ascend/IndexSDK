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

#include "ascendsearch/ascend/rpc/AscendRpcIndexIVFSPSQ.h"

#include <algorithm>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "index_custom/IndexIVFSPSQL2Aicpu.h"
#include "rpc-local/RpcLocalSession.h"

using namespace ::ascendSearch;

namespace faiss {
    namespace ascendSearch {
        RpcError RpcCreateIndexIVFSPSQ(rpcContext ctx, int &indexId, const IndexIVFSPSQParameter &parameter)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            ::ascendSearch::Index* index = nullptr;
            if (!parameter.slim) {
                switch (parameter.metric) {
                    case MetricType::METRIC_L2:
                        index = new IndexIVFSPSQL2Aicpu(parameter.dim, parameter.dim2, parameter.k,
                                                   parameter.nlist, parameter.encodeResidual,
                                                   parameter.nprobe, parameter.searchListSize,
                                                   parameter.handleBatch, parameter.filterable,
                                                   parameter.resourceSize);
                        break;
                    default:
                        ASCEND_THROW_MSG("Unsupported metric type\n");
                }
            }
            auto ret = index->init();
            if (ret != APP_ERR_OK) {
                delete index;
            }
            APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to create index flat");
            indexId = session->AddIndex(index);
            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQTrainCodeBook(rpcContext ctx, int indexId,
                                              const RpcIndexCodeBookTrainerConfig &codeBookTrainerConfig,
                                              float *codebookPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            IVFSPCodeBookTrainerInitParam initParam;
            initParam.numIter = codeBookTrainerConfig.numIter;
            initParam.device = codeBookTrainerConfig.device;
            initParam.ratio = codeBookTrainerConfig.ratio;
            initParam.batchSize = codeBookTrainerConfig.batchSize;
            initParam.codeNum = codeBookTrainerConfig.codeNum;
            initParam.codeBookOutputDir = codeBookTrainerConfig.codeBookOutputDir;
            initParam.learnDataPath = codeBookTrainerConfig.learnDataPath;
            initParam.memLearnData = codeBookTrainerConfig.memLearnData;
            initParam.memLearnDataSize = codeBookTrainerConfig.memLearnDataSize;
            initParam.verbose = codeBookTrainerConfig.verbose;
            
            auto ret = pIndex->trainCodeBook(initParam, codebookPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQGetCodeWord(rpcContext ctx, int indexId, int n, int dim,
                                            const float *feature, uint16_t *codeWord,
                                            idx_t* labels)
        {
            APP_LOG_INFO("RpcIndexIVFSPSQGetCodeWord\n");
            VALUE_UNUSED(dim);
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            AscendTensor<float, DIMS_2> featureTensor({ n, dim });
            auto ret = aclrtMemcpy(featureTensor.data(), featureTensor.getSizeInBytes(),
                                   feature, n * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, RPC_ERROR_ERROR, "aclrtMemcpy error %d", ret);

            ret = pIndex->getCodeWord(n, featureTensor.data(), reinterpret_cast<float16_t *>(codeWord),
                                      reinterpret_cast<::ascendSearch::idx_t *>(labels));
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to getCodeWord index id: %d\n", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
            const char *dataPath, float *codebookPtr, float *spsqPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            auto ret = pIndex->loadDeviceAllData(dataPath, codebookPtr, spsqPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
            rpcContext ctxLoaded, int indexIdLoaded, const char *dataPath)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            auto *sessionLoaded = static_cast<RpcLocalSession *>(ctxLoaded);
            auto *indexLoaded = sessionLoaded->GetIndex(indexIdLoaded);
            auto *pIndexLoaded = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(indexLoaded);
            APPERR_RETURN_IF_NOT_FMT(pIndexLoaded, RPC_ERROR_ERROR,
                                     "Invalid loaded index id: %d\n", indexIdLoaded);
            auto ret = pIndex->loadDeviceAllData(dataPath, nullptr, nullptr, pIndexLoaded);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
            const uint8_t* data, size_t dataLen, float *codebookPtr, float *spsqPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            auto ret = pIndex->loadDeviceAllData(data, dataLen, codebookPtr, spsqPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQLoadAllData(rpcContext ctx, int indexId,
            rpcContext ctxLoaded, int indexIdLoaded, const uint8_t* data, size_t dataLen)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            auto *sessionLoaded = static_cast<RpcLocalSession *>(ctxLoaded);
            auto *indexLoaded = sessionLoaded->GetIndex(indexIdLoaded);
            auto *pIndexLoaded = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(indexLoaded);
            APPERR_RETURN_IF_NOT_FMT(pIndexLoaded, RPC_ERROR_ERROR,
                                     "Invalid loaded index id: %d\n", indexIdLoaded);
            auto ret = pIndex->loadDeviceAllData(data, dataLen, nullptr, nullptr, pIndexLoaded);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQSaveAllData(rpcContext ctx, int indexId,
            const char *dataPath, float *codebookPtr, float *spsqPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            auto ret = pIndex->saveDeviceAllData(dataPath, codebookPtr, spsqPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQSaveAllData(rpcContext ctx, int indexId,
            uint8_t* &data, size_t &dataLen, float *codebookPtr, float *spsqPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            auto ret = pIndex->saveDeviceAllData(data, dataLen, codebookPtr, spsqPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQSaveCodeBook(rpcContext ctx, int indexId,
            uint8_t* &data, size_t &dataLen, float *codebookPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            auto ret = pIndex->saveCodeBook(data, dataLen, codebookPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to save codebook for index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQLoadCodeBook(rpcContext ctx, int indexId,
            const uint8_t* data, size_t dataLen, float *codebookPtr)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            auto ret = pIndex->loadCodeBook(data, dataLen, codebookPtr);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to load codebook for index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQAddFinish(rpcContext ctx, int indexId)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            int ret = 0;
            try {
                pIndex->addFinishMerge();
            } catch (const std::exception &e) {
                ret = RPC_ERROR_ERROR;
            }
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR, "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQAddWithIds(rpcContext ctx, int indexId, int n, int,
                                           int listId, const uint8_t *data, const ascend_idx_t *ids,
                                           const float *precomputedVal, bool useNPU)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            auto ret = pIndex->addVectors(listId, n, data, ids, precomputedVal, useNPU);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, RPC_ERROR_ERROR,
                                     "Failed to add to index %d", indexId);

            return RPC_ERROR_NONE;
        }


        RpcError RpcIndexIVFSPSQAddCodeBook(rpcContext ctx, int indexId, int n, int dim,
            int, const uint16_t *data, idx_t *)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            AscendTensor<float16_t, DIMS_2> vec(const_cast<float16_t *>(data), { n, dim });
            pIndex->updateCoarseCentroidsData(vec);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQAddCodeBook(rpcContext ctx, int indexId, rpcContext ctxLoaded, int indexIdLoaded)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            auto *sessionLoaded = static_cast<RpcLocalSession *>(ctxLoaded);
            auto *indexLoaded = sessionLoaded->GetIndex(indexIdLoaded);
            auto *pIndexLoaded = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(indexLoaded);
            APPERR_RETURN_IF_NOT_FMT(pIndexLoaded, RPC_ERROR_ERROR,
                                     "Invalid loaded index id: %d\n", indexIdLoaded);
            pIndex->updateCoarseCentroidsData(pIndexLoaded);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQUpdateTrainedValue(rpcContext ctx, int indexId, int dim,
                                                   uint16_t *vmin, uint16_t *vdiff, bool)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);

            AscendTensor<float16_t, DIMS_1> vminTensor(reinterpret_cast<float16_t *>(vmin), { dim });
            AscendTensor<float16_t, DIMS_1> vdiffTensor(reinterpret_cast<float16_t *>(vdiff), { dim });
            pIndex->updateTrainedValue(vminTensor, vdiffTensor);
            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQGetBaseSize(rpcContext ctx, int indexId, uint32_t &size)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            size = static_cast<uint32_t>(pIndex->getSize());
            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQUpdateNprobe(rpcContext ctx, int indexId, int nprobe)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQL2Aicpu *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d", indexId);
            pIndex->setNumProbes(nprobe);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQRemoveIds(rpcContext ctx, int indexId, int n, ascend_idx_t *ids, uint32_t *numRemoved)
        {
            APP_LOG_INFO("remove %d vector(s) of index %d\n", n, indexId);
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR,
                                     "Invalid index id: %d\n", indexId);
            ::ascendSearch::IDSelectorBatch batch(n, ids);
            *numRemoved = index->removeIds(batch);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQRemoveRangeIds(rpcContext ctx, int indexId, ascend_idx_t min,
            ascend_idx_t max, uint32_t *numRemoved)
        {
            APP_LOG_INFO("remove vector(s) [%d, %d] of index %d\n", min, max, indexId);
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            APPERR_RETURN_IF_NOT_FMT(index, RPC_ERROR_ERROR, "Invalid index id: %d\n", indexId);

            ::ascendSearch::IDSelectorRange range(min, max);
            *numRemoved = index->removeIds(range);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexIVFSPSQGetListLength(rpcContext ctx, int indexId, int listId, uint32_t &len)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVF *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d", indexId);
            len = pIndex->getListLength(listId);

            return RPC_ERROR_NONE;
        }

        RpcError RpcIndexGetAddFinish(rpcContext ctx, int indexId, bool &addFinish)
        {
            auto *session = static_cast<RpcLocalSession *>(ctx);
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(session->deviceId), RPC_ERROR_ERROR);

            auto *index = session->GetIndex(indexId);
            auto *pIndex = dynamic_cast<::ascendSearch::IndexIVFSPSQ *>(index);
            APPERR_RETURN_IF_NOT_FMT(pIndex, RPC_ERROR_ERROR,
                                     "Invalid index id: %d", indexId);
            addFinish = pIndex->isAddFinish();

            return RPC_ERROR_NONE;
        }
    } // namespace ascendSearch
} // namespace faiss
