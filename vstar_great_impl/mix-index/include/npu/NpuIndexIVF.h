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


#ifndef IVFSP_INDEXIVF_H
#define IVFSP_INDEXIVF_H

#include "NpuIndex.h"
#include "npu/common/DeviceVector.h"

namespace ascendSearchacc {
class NpuIndexIVF : public NpuIndex {
public:
    NpuIndexIVF(int dim, int nList, MetricType metricType, NpuIndexConfig config);
    explicit NpuIndexIVF(NpuIndexConfig config);

    virtual ~NpuIndexIVF();

protected:
    int nList = 0;

private:
    /* *organization based on bucket ID, which are stored on device memory */
    std::vector<std::unique_ptr<DeviceVector<int64_t> > > ids;
    /* *organization based on bucket ID, which are stored on device memory */
    std::vector<std::unique_ptr<DeviceVector<unsigned char> > > baseShaped;
};
}  // namespace ascendSearchacc
#endif  // IVFSP_INDEXIVF_H
