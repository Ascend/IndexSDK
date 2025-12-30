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

#pragma once

#include <limits>
#include <memory>

#include <malloc.h>
#include <cblas.h>

#include "CommonIncludes.h"
#include "Utils.h"

namespace math_utils {

float CalcDistance(float *vec1, float *vec2, size_t dim);

void ComputeVecsL2sq(float *vecsL2sq, float *data, const size_t numPoints, const size_t dim);

// Used internally by ComputeClosestCenters
void ComputeClosestCentersInBlock(const float *const data, const size_t numPoints, const size_t dim,
                                  const float *const centers, const size_t numCenters,
                                  const float *const docsL2sq, const float *const centersL2sq,
                                  uint32_t *centerIndex, float *const distMatrix, size_t k = 1);

void ComputeClosestCenters(float *data, size_t numPoints, size_t dim, float *pivotData, size_t numCenters,
                           size_t k, uint32_t *closestCentersIvf, std::vector<size_t> *invertedIndex = nullptr,
                           float *ptsNormsSquared = nullptr);

} // namespace math_utils

namespace kmeans {

float LloydsIter(float *data, size_t numPoints, size_t dim, float *centers, size_t numCenters, float *docsL2sq,
                 std::vector<size_t> *closestDocs, uint32_t *&closestCenter);

float RunLloyds(float *data, size_t numPoints, size_t dim, float *centers, const size_t numCenters,
                const size_t maxReps, std::vector<size_t> *closestDocs, uint32_t *closestCenter);

void SelectingPivots(float *data, size_t numPoints, size_t dim, float *pivotData, size_t numCenters);

void KmeansppSelectingPivots(float *data, size_t numPoints, size_t dim, float *pivotData, size_t numCenters);
} // namespace kmeans
