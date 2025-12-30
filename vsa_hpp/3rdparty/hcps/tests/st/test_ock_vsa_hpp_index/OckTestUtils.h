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


#ifndef HCPS_PIER_TESTS_ST_TEST_OCK_VSA_HPP_INDEX_UTILS_H
#define HCPS_PIER_TESTS_ST_TEST_OCK_VSA_HPP_INDEX_UTILS_H
#include <cmath>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <unordered_map>
#include "securec.h"
#include "ock/utils/OckSafeUtils.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
class OckTestUtils {
public:
    static inline double GetMilliSecs()
    {
        struct timeval tv = { 0, 0 };
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
    }

    template <typename keyData, typename indexData>
    static std::unordered_map<keyData, indexData> VecToMap(const std::vector<keyData> &vec)
    {
        std::unordered_map<keyData, indexData> ret;
        for (uint32_t i = 0; i < vec.size(); ++i) {
            ret.insert(std::make_pair(vec.at(i), static_cast<indexData>(i)));
        }
        return ret;
    }

    static void PrintErrorDistance(const int64_t labelCur, const float distanceMindX, const float distanceHCP,
        const int index)
    {
        std::cout << "ERROR: mindx " << labelCur << " block[" << (labelCur / 262144ULL) << "] pos[" << index <<
                  "] MISS" << ", mindx distance is " << distanceMindX << ", same pos hcp distance is " <<
                  distanceHCP << std::endl;
    }

    // 召回率计算
    static float CalcRecallWithDist(const std::unordered_map<int64_t, uint32_t> &labelsHCP,
        const std::vector<int64_t> &labelsMindX, std::vector<float> &distancesHCP, std::vector<float> &distancesMindX,
        float errorRange, float minNNCosValue)
    {
        float recall = 0.0f;
        if (labelsMindX.size() != labelsHCP.size() || distancesHCP.size() != distancesMindX.size()) {
            std::cout << "The size of the output results is inconsistent! Labels in MindX = " <<
                labelsMindX.size() << ", in HCP = " << labelsHCP.size() << "; Distances in MindX = " <<
                distancesMindX.size() << ", in HCP = " << distancesHCP.size();
            return recall;
        }
        int totalNum = labelsMindX.size();
        int correctNum = 0;
        float minDistance = *(std::min_element(distancesMindX.begin(), distancesMindX.end()));
        for (int i = 0; i < totalNum; i++) {
            int64_t labelCur = labelsMindX[i];
            if (labelsHCP.find(labelCur) != labelsHCP.end()) { // HCP 中存在该 label
                correctNum++;
            } else if (utils::SafeFloatEqual(distancesMindX[i], minDistance, errorRange) &&
                utils::SafeFloatEqual(distancesHCP[i], minDistance, errorRange)) { // 是 HCP 结果中的最小距离
                correctNum++;
            } else if (distancesMindX[i] < minNNCosValue) { // 小于 minNNCosValue 的向量不考虑召回
                correctNum++;
                PrintErrorDistance(labelCur, distancesMindX[i], distancesHCP[i], i);
            } else {
                PrintErrorDistance(labelCur, distancesMindX[i], distancesHCP[i], i);
            }
        }
        recall = static_cast<float>(correctNum) / static_cast<float>(totalNum);
        std::cout << "recall = " << recall << ", mindx max dist = " << distancesMindX[0] << ", mindx min dist = " <<
            distancesMindX[distancesMindX.size() - 1] << std::endl;
        return recall;
    }

    // 距离结果的比对
    static void CalcErrorWithDist(const std::vector<float> &distHCP, const std::vector<float> &distMindX,
        const float curSearchRecall, const float errorRange)
    {
        bool flag = true;
        for (size_t i = 0; i < distHCP.size(); i++) {
            if (!utils::SafeFloatEqual(distMindX[i], distHCP[i], errorRange)) {
                std::cout << "DISERROR[" << i << "]ock[" << distHCP[i] << "]mindx[" << distMindX[i] << "]recallRate[" <<
                    curSearchRecall << "]" << std::endl;
                std::cout << "test failed" << std::endl;
                flag = false;
                break;
            }
        }
        if (flag) {
            std::cout << "Test success distances are same, can ignore some few difference of labels" << std::endl;
        }
    }

    // QPS计算
    static float GetQPS()
    {
        return 2.01; // 2.01 临时值；此函数待实现
    }

    // 环境变量设置函数
    template <typename DataTemp> static void setEnvValue(std::string valueName, DataTemp value)
    {
        std::string valueStr = std::to_string(value);
        std::string valueStrSET = valueName + "=" + valueStr;
        char *valueStrSETCHAR = new char[valueStrSET.length() + 1];
        errno_t ret = strcpy_s(valueStrSETCHAR, valueStrSET.length() + 1, valueStrSET.c_str());
        if (ret != EOK) {
            std::cout << "memcpy_s failed, ret = " << ret << std::endl;
        }
        int envRet = putenv(valueStrSETCHAR);
        if (envRet == 0) {
            std::cout << "set " << valueStrSET << " success!" << std::endl;
        }
    }

private:
    static void CompareIdSet() {}
};
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif // HCPS_PIER_TESTS_ST_TEST_OCK_VSA_HPP_INDEX_UTILS_H
