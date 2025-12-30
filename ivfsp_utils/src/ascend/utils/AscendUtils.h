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


#ifndef ASCEND_UTILS_INCLUDED
#define ASCEND_UTILS_INCLUDED

#include <future>

namespace faiss {
namespace ascendSearch {
#define CALL_PARALLEL_FUNCTOR(devices, threadPool, functor)                 \
    do {                                                                    \
        if ((devices) > 1) {                                                  \
            std::vector<std::future<void>> functorRets;                     \
            for (size_t i = 0; i < (devices); i++) {                          \
                functorRets.emplace_back((threadPool)->Enqueue(functor, i));  \
            }                                                               \
                                                                            \
            try {                                                           \
                for (auto & ret : functorRets) {                            \
                    ret.get();                                              \
                }                                                           \
            } catch (std::exception & e) {                                  \
                FAISS_THROW_MSG("wait for parallel future failed.");        \
            }                                                               \
        } else {                                                            \
            functor(0);                                                     \
        }                                                                   \
    } while (false)

#define CALL_PARALLEL_FUNCTOR_INDEXMAP(map, threadPool, functor)            \
    do {                                                                    \
        std::vector<std::future<void>> functorRets;                         \
        for (auto & index : (map)) {                                        \
            functorRets.emplace_back(                                       \
                (threadPool)->Enqueue(functor, index.first, index.second));   \
        }                                                                   \
                                                                            \
        try {                                                               \
            for (auto & ret : functorRets) {                                \
                ret.get();                                                  \
            }                                                               \
        } catch (std::exception & e) {                                      \
            FAISS_THROW_MSG("wait for indexmap parallel future failed.");   \
        }                                                                   \
    } while (false)
} // ascend
} // faiss
#endif
