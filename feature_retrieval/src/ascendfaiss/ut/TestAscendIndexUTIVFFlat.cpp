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

#include <string>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <faiss/IndexFlat.h>
#include "mockcpp/mockcpp.hpp"
#include "faiss/ascend/AscendIndexIVFFlat.h"

namespace ascend {

TEST(TestAscendIndexIVFFlat, Construct)
{
    int dim = 64;
    int ncentroids = 1024;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexIVFFlatConfig conf({ 0 });
    try {
        faiss::ascend::AscendIndexIVFFlat index(dim, type, ncentroids, conf);
    } catch(std::exception &e) {
        msg = e.what();
    }
    std::string str = "Unsupported dims";
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}
}
