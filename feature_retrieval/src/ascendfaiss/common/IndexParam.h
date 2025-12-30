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

#ifndef ASCENDHOST_SRC_ASCENDFAISS_COMMON_INDEXPARAM_H
#define ASCENDHOST_SRC_ASCENDFAISS_COMMON_INDEXPARAM_H

namespace faiss {
namespace ascend {

template<typename Q, typename D, typename L> class IndexParam {
public:
    IndexParam(int deviceId, int n, int dim, int k) : deviceId(deviceId), n(n), dim(dim), k(k) {}
    explicit IndexParam(int deviceId) : deviceId(deviceId) {}
    IndexParam() {}
    int deviceId = 0;
    int n = 0;
    int dim = 0;
    int k = 0;
    int listId = 0;
    const Q *query = nullptr;
    D *distance = nullptr;
    L *label = nullptr;
};
} // namespace ascend
} // namespace faiss
#endif // ASCENDHOST_SRC_ASCENDFAISS_COMMON_INDEXPARAM_H
