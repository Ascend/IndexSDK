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


#include "MathUtils.h"

namespace math_utils {

float CalcDistance(float *vec1, float *vec2, size_t dim)
{
    float dist = 0;
    for (size_t j = 0; j < dim; j++) {
        dist += (vec1[j] - vec2[j]) * (vec1[j] - vec2[j]);
    }
    return dist;
}

// compute l2-squared norms of data stored in row major numPoints * dim,
// needs
// to be pre-allocated
void ComputeVecsL2sq(float *vecsL2sq, float *data, const size_t numPoints, const size_t dim)
{
#pragma omp parallel for schedule(static, 8192)
    for (int64_t nIter = 0; nIter < static_cast<int64_t>(numPoints); nIter++) {
        vecsL2sq[nIter] = cblas_snrm2(static_cast<blasint>(dim), (data + (nIter * dim)), 1);
        vecsL2sq[nIter] *= vecsL2sq[nIter];
    }
}

// calculate k closest centers to data of numPoints * dim (row major)
// centers is numCenters * dim (row major)
// data_l2sq has pre-computed squared norms of data
// centersL2sq has pre-computed squared norms of centers
// pre-allocated centerIndex will contain id of nearest center
// pre-allocated distMatrix shound be numPoints * numCenters and contain
// squared distances
// Default value of k is 1

// Ideally used only by ComputeClosestCenters
void ComputeClosestCentersInBlock(const float *const data, const size_t numPoints, const size_t dim,
                                  const float *const centers, const size_t numCenters,
                                  const float *const docsL2sq, const float *const centersL2sq,
                                  uint32_t *centerIndex, float *const distMatrix, size_t k)
{
    DISK_THROW_IF_NOT_MSG(k <= numCenters, "k and numCenters should be equal.");
    auto onesCenters = std::make_unique<float[]>(numCenters);
    auto onesPoints = std::make_unique<float[]>(numPoints);

    for (size_t i = 0; i < numCenters; i++) {
        onesCenters[i] = 1.0;
    }
    for (size_t i = 0; i < numPoints; i++) {
        onesPoints[i] = 1.0;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<blasint>(numPoints),
                static_cast<blasint>(numCenters), static_cast<blasint>(1), 1.0f,
                docsL2sq, static_cast<blasint>(1), onesCenters.get(), static_cast<blasint>(1), 0.0f,
                distMatrix, static_cast<blasint>(numCenters));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<blasint>(numPoints),
                static_cast<blasint>(numCenters), static_cast<blasint>(1), 1.0f,
                onesPoints.get(), static_cast<blasint>(1), centersL2sq, static_cast<blasint>(1), 1.0f,
                distMatrix, static_cast<blasint>(numCenters));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<blasint>(numPoints),
                static_cast<blasint>(numCenters), static_cast<blasint>(dim), -2.0f,
                data, static_cast<blasint>(dim), centers, static_cast<blasint>(dim), 1.0f,
                distMatrix, static_cast<blasint>(numCenters));

    if (k == 1) {
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < static_cast<int64_t>(numPoints); i++) {
            float min = std::numeric_limits<float>::max();
            float *current = distMatrix + (i * numCenters);
            for (size_t j = 0; j < numCenters; j++) {
                if (current[j] < min) {
                    centerIndex[i] = static_cast<uint32_t>(j);
                    min = current[j];
                }
            }
        }
    } else {
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < static_cast<int64_t>(numPoints); i++) {
            std::priority_queue<PivotContainer> topKQueue;
            float *current = distMatrix + (i * numCenters);
            for (size_t j = 0; j < numCenters; j++) {
                PivotContainer thisPiv(j, current[j]);
                topKQueue.push(thisPiv);
            }
            for (size_t j = 0; j < k; j++) {
                PivotContainer thisPiv = topKQueue.top();
                centerIndex[i * k + j] = static_cast<uint32_t>(thisPiv.pivID);
                topKQueue.pop();
            }
        }
    }
}

// Given data in numPoints * new_dim row major
// Pivots stored in full_pivot_data as numCenters * new_dim row major
// Calculate the k closest pivot for each point and store it in vector
// closestCentersIvf (row major, numPoints*k) (which needs to be allocated
// outside) Additionally, if inverted index is not null (and pre-allocated),
// it
// will return inverted index for each center, assuming each of the inverted
// indices is an empty vector. Additionally, if ptsNormsSquared is not null,
// then it will assume that point norms are pre-computed and use those values

void ComputeClosestCenters(float *data, size_t numPoints, size_t dim, float *pivotData, size_t numCenters,
                           size_t k, uint32_t *closestCentersIvf, std::vector<size_t> *invertedIndex,
                           float *ptsNormsSquared)
{
    DISK_THROW_IF_NOT_MSG(k <= numCenters, "k and numCenters should be equal.");

    bool isNormGivenForPts = (ptsNormsSquared != nullptr);

    auto pivsNormsSquared = std::make_unique<float[]>(numCenters);
    std::unique_ptr<float[]> ptsNormsSquaredSharedPtr;
    if (!isNormGivenForPts) {
        ptsNormsSquaredSharedPtr = std::make_unique<float[]>(numPoints);
        ptsNormsSquared = ptsNormsSquaredSharedPtr.get();
    }

    size_t parBlockSize = numPoints;
    DISK_THROW_IF_NOT(numPoints > 0 && parBlockSize > 0);
    size_t nBlocks =
        (numPoints % parBlockSize) == 0 ? (numPoints / parBlockSize) : (numPoints / parBlockSize) + 1;

    if (!isNormGivenForPts) {
        math_utils::ComputeVecsL2sq(ptsNormsSquared, data, numPoints, dim);
    }
    math_utils::ComputeVecsL2sq(pivsNormsSquared.get(), pivotData, numCenters, dim);
    auto closestCenters = std::make_unique<uint32_t[]>(parBlockSize * k);
    auto distanceMatrix = std::make_unique<float[]>(numCenters * parBlockSize);

    for (size_t curBlk = 0; curBlk < nBlocks; curBlk++) {
        float *dataCurBlk = data + curBlk * parBlockSize * dim;
        size_t numPtsBlk = std::min(parBlockSize, numPoints - curBlk * parBlockSize);
        float *ptsNormsBlk = ptsNormsSquared + curBlk * parBlockSize;

        math_utils::ComputeClosestCentersInBlock(dataCurBlk, numPtsBlk, dim, pivotData, numCenters,
                                                 ptsNormsBlk, pivsNormsSquared.get(), closestCenters.get(),
                                                 distanceMatrix.get(), k);

        int64_t totalPointsToProcess = std::min(static_cast<int64_t>(numPoints),
                                                static_cast<int64_t>((curBlk + 1) * parBlockSize));
#pragma omp parallel for schedule(static, 1)
        for (int64_t j = curBlk * parBlockSize; j < totalPointsToProcess; j++) {
            for (size_t l = 0; l < k; l++) {
                size_t thisCenterID = closestCenters[(j - curBlk * parBlockSize) * k + l];
                closestCentersIvf[j * k + l] = static_cast<uint32_t>(thisCenterID);
                if (invertedIndex != nullptr) {
#pragma omp critical
                    invertedIndex[thisCenterID].push_back(j);
                }
            }
        }
    }
}

} // namespace math_utils

namespace kmeans {

// run Lloyds one iteration
// Given data in row major numPoints * dim, and centers in row major
// numCenters * dim And squared lengths of data points, output the closest
// center to each data point, update centers, and also return inverted index.
// If
// closestCenters == nullptr, will allocate memory and return. Similarly, if
// closestDocs == nullptr, will allocate memory and return.

float LloydsIter(float *data, size_t numPoints, size_t dim, float *centers, size_t numCenters, float *docsL2sq,
                 std::vector<size_t> *closestDocs, uint32_t *&closestCenter)
{
    const size_t MAX_POINTS = 1000000000;
    DISK_THROW_IF_NOT(numPoints > 0 && numPoints <= MAX_POINTS);
    for (size_t c = 0; c < numCenters; ++c) {
        closestDocs[c].clear();
    }
    math_utils::ComputeClosestCenters(data, numPoints, dim, centers, numCenters, 1, closestCenter, closestDocs,
                                      docsL2sq);
    std::fill(centers, centers + numCenters * dim, 0.0f);

#pragma omp parallel for schedule(static, 1)
    for (size_t c = 0; c < numCenters; ++c) {
        float *center = centers + c * dim;
        std::vector<double> clusterSum(dim, 0.0);
        for (size_t i = 0; i < closestDocs[c].size(); i++) {
            float *current = data + (closestDocs[c][i] * dim);
            for (size_t j = 0; j < dim; j++) {
                clusterSum[j] += static_cast<double>(current[j]);
            }
        }
        if (closestDocs[c].size() > 0) {
            for (size_t i = 0; i < dim; i++) {
                center[i] = static_cast<float>(clusterSum[i] / static_cast<double>(closestDocs[c].size()));
            }
        }
    }

    // compute residuals
    float residual = 0.0;
    size_t bufPad = 32;
    size_t chunkSize = 2 * 8192;
    size_t nchunks = numPoints / chunkSize + (numPoints % chunkSize == 0 ? 0 : 1);
    std::vector<float> residuals(nchunks * bufPad, 0.0);

#pragma omp parallel for schedule(static, 32)
    for (size_t chunk = 0; chunk < nchunks; ++chunk) {
        for (size_t d = chunk * chunkSize; d < numPoints && d < (chunk + 1) * chunkSize; ++d) {
            residuals[chunk * bufPad] += math_utils::CalcDistance(data + (d * dim),
                                                                  centers + static_cast<size_t>(closestCenter[d]) * dim,
                                                                  dim);
        }
    }

    for (size_t chunk = 0; chunk < nchunks; ++chunk) {
        residual += residuals[chunk * bufPad];
    }

    return residual;
}

// Run Lloyds until maxReps or stopping criterion
// If you pass nullptr for closestDocs and closestCenter, it will NOT return
// the
// results, else it will assume appriate allocation as closestDocs = new
// vector<size_t> [numCenters], and closestCenter = new size_t[numPoints]
// Final centers are output in centers as row major numCenters * dim
float RunLloyds(float *data, size_t numPoints, size_t dim, float *centers, const size_t numCenters,
                const size_t maxReps, std::vector<size_t> *closestDocs, uint32_t *closestCenter)
{
    float residual = std::numeric_limits<float>::max();
    std::unique_ptr<std::vector<size_t>[]> closestDocsSmartPtr;
    if (closestDocs == nullptr) {
        closestDocsSmartPtr = std::make_unique<std::vector<size_t>[]>(numCenters);
        closestDocs = closestDocsSmartPtr.get();
    }
    std::unique_ptr<uint32_t[]> closestCenterSmartPtr;
    if (closestCenter == nullptr) {
        closestCenterSmartPtr = std::make_unique<uint32_t[]>(numPoints);
        closestCenter = closestCenterSmartPtr.get();
    }

    auto docsL2sq = std::make_unique<float[]>(numPoints);
    math_utils::ComputeVecsL2sq(docsL2sq.get(), data, numPoints, dim);

    float oldResidual = 0;
    for (size_t i = 0; i < maxReps; ++i) {
        oldResidual = residual;
        residual = LloydsIter(data, numPoints, dim, centers, numCenters, docsL2sq.get(), closestDocs, closestCenter);
        if ((i != 0 && ((oldResidual - residual) / residual) < 0.00001) ||  // 判断变化率是否小于0.00001
            (residual < std::numeric_limits<float>::epsilon())) {
            break;
        }
    }
    return residual;
}

// assumes memory allocated for pivotData as new
// float[numCenters*dim]
// and select randomly numCenters points as pivots
void SelectingPivots(float *data, size_t numPoints, size_t dim, float *pivotData, size_t numCenters)
{
    DISK_THROW_IF_NOT_MSG(numCenters <= numPoints, "PQ centers should be <= numPoints.");

    std::vector<size_t> picked;
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator(x);
    std::uniform_int_distribution<size_t> distribution(0, numPoints - 1);

    size_t tmpPivot;
    for (size_t j = 0; j < numCenters; j++) {
        tmpPivot = distribution(generator);
        if (std::find(picked.begin(), picked.end(), tmpPivot) != picked.end()) {
            continue;
        }
        picked.push_back(tmpPivot);
        std::copy(data + tmpPivot * dim, data + tmpPivot * dim + dim, pivotData + j * dim);
    }
}

void KmeansppSelectingPivots(float *data, size_t numPoints, size_t dim, float *pivotData, size_t numCenters)
{
    if (numPoints == 0) {
        return;
    }
    if (numPoints > 1 << 23) {  // 2的23次方，即最大num_points是8388608, 此时不支持pp方法
        SelectingPivots(data, numPoints, dim, pivotData, numCenters);
        return;
    }

    std::vector<size_t> picked;
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator(x);
    std::uniform_real_distribution<> distribution(0, 1);
    std::uniform_int_distribution<size_t> intDist(0, numPoints - 1);
    size_t initID = intDist(generator);
    size_t numPicked = 1;

    picked.push_back(initID);
    std::copy(data + initID * dim, data + initID * dim + dim, pivotData);

    auto dist = std::make_unique<float[]>(numPoints);

#pragma omp parallel for schedule(static, 8192)
    for (int64_t i = 0; i < static_cast<int64_t>(numPoints); i++) {
        dist[i] = math_utils::CalcDistance(data + i * dim, data + initID * dim, dim);
    }

    double dartVal;
    size_t tmpPivot;
    bool sumFlag = false;

    while (numPicked < numCenters) {
        dartVal = distribution(generator);

        double sum = 0;
        for (size_t i = 0; i < numPoints; i++) {
            sum = sum + dist[i];
        }
        if (diskann_pro::FloatEqual(sum, 0)) {
            sumFlag = true;
        }

        dartVal *= sum;

        double prefixSum = 0;
        for (size_t i = 0; i < numPoints; i++) {
            tmpPivot = i;
            if (dartVal >= prefixSum && dartVal < prefixSum + dist[i]) {
                break;
            }
            prefixSum += dist[i];
        }

        if (std::find(picked.begin(), picked.end(), tmpPivot) != picked.end() && (sumFlag == false)) {
            continue;
        }
        picked.push_back(tmpPivot);
        std::copy(data + tmpPivot * dim, data + tmpPivot * dim + dim, pivotData + numPicked * dim);

#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < static_cast<int64_t>(numPoints); i++) {
            dist[i] = (std::min)(dist[i], math_utils::CalcDistance(data + i * dim, data + tmpPivot * dim, dim));
        }
        numPicked++;
    }
}

} // namespace kmeans
