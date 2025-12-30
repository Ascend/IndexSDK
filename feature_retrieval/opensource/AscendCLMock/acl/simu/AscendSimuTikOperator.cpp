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

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "AscendSimuExecFlow.h"
#include "securec.h"
#include "../acl_base.h"

void WarpAffineU8Operator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t*)(opHandle.inputData[0].data);
    float *tilling = (float*)(opHandle.inputData[1].data);
    uint8_t *output = (uint8_t*)(opHandle.outputData[0].data);
    cv::Mat inverseMat(3, 3, CV_32FC1);
    inverseMat = (cv::Mat_<float>(3, 3) << tilling[0], tilling[1], tilling[2],
            tilling[3], tilling[4], tilling[5], 0, 0, 1);
    cv::Mat meinvMat = inverseMat.inv();
    float *meinv = (float *)meinvMat.data;
    cv::Mat warpPerspective_mat(2, 3, CV_32FC1);
    warpPerspective_mat = (cv::Mat_<float>(2, 3) << meinv[0], meinv[1], meinv[2], meinv[3], meinv[4], meinv[5]);
    cv::Mat src(tilling[15], tilling[16], CV_8UC3, input1);
    cv::Mat dst(tilling[6], tilling[7], CV_8UC3, output);
    cv::Scalar scalar(tilling[10], tilling[11], tilling[12], tilling[13]);
    cv::warpAffine(src, dst, warpPerspective_mat, dst.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, scalar);
}
void simuOpTikInstall()
{
    REG_OP("warp_affine_uint8", WarpAffineU8Operator);
}
void simuOpTikUninstall()
{
    UNREG_OP("warp_affine_uint8");
}