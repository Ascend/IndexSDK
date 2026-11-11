/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

// SIFT1M end-to-end example using Init/Train/Search on a single index.
// Prepare graph-build and search models together from feature_retrieval/src/ascendfaiss/tools:
//   python3 cagra_generate_model.py -data_base 1000000 -d 128 -degree 64 -topK 32 -t Ascend950PR
//   export MX_INDEX_MODELPATH="$PWD/op_models"
// Run this binary with matching N/D/M/G/TopK and a SIFT directory:
//   export CAGRA_SIFT_DIR=/path/to/sift CAGRA_DEVICE_ID=0
//   ./TestAscendIndexCagra --gtest_filter=TestAscendIndexCagra.Sift1M
// Train builds the graph and quantizes the base; Add(n,data) is an alternative to Train.
// Search quantizes queries internally. No separate Builder or QuantizeData/QuantizeQueries call is needed.
#if __has_include(<faiss/ascend/AscendIndexCagra.h>)
#include <faiss/ascend/AscendIndexCagra.h>
#else
#include "index/AscendIndexCagra.h"
#endif
#include <faiss/IndexFlat.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
struct Config
{
    int dim = 128;
    int dataNum = 1000000;
    int intermediateDegree = 128;
    int graphDegree = 64;
    int topK = 32;
    int queryNum = 100;
    int graphAccuracySamples = 0;
    int deviceId = 0;
    int maxIterations = 20;
    int searchWarmups = 3;
    int searchLoops = 20;
    float terminationThreshold = 0.0001F;
    bool guaranteeConnectivity = true;
    bool verbose = true;
    bool runCpuGroundTruth = true;
    std::string siftDir;
};
int GetEnvInt(const char *name, int defaultValue, int minimum, int maximum)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return defaultValue;
    }
    const std::string text(value);
    size_t parsedLength = 0;
    const long long parsed = std::stoll(text, &parsedLength);
    if (parsedLength != text.size() || parsed < minimum || parsed > maximum)
    {
        throw std::runtime_error(std::string(name) + " must be in [" + std::to_string(minimum) + ", " +
                                 std::to_string(maximum) + "], got: " + text);
    }
    return static_cast<int>(parsed);
}

bool GetEnvBool(const char *name, bool defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return defaultValue;
    }
    const std::string text(value);
    if (text == "1" || text == "true" || text == "TRUE" || text == "on")
    {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off")
    {
        return false;
    }
    throw std::runtime_error(std::string(name) + " must be 0/1, true/false or on/off, got: " + text);
}

std::string GetEnvString(const char *name, const std::string &defaultValue)
{
    const char *value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? defaultValue : value;
}

int GetCompatibleEnvInt(const char *legacyName, const char *alias, int fallback, int minimum, int maximum)
{
    const char *value = std::getenv(legacyName);
    return GetEnvInt(value != nullptr && value[0] != '\0' ? legacyName : alias, fallback, minimum, maximum);
}

bool IsSupportedTopK(int topK)
{
    return topK == 32 || topK == 64 || topK == 128 || topK == 256 || topK == 512 || topK == 1024;
}

Config ReadConfig()
{
    Config config;
    config.deviceId = GetEnvInt("CAGRA_DEVICE_ID", 0, 0, 1024);
    config.dataNum = GetEnvInt("CAGRA_DATA_NUM", 1000000, 129, std::numeric_limits<int>::max());
    config.topK = GetEnvInt("CAGRA_TOPK", 32, 1, std::min(1024, config.dataNum));
    if (!IsSupportedTopK(config.topK)) throw std::runtime_error("CAGRA_TOPK must be in {32,64,128,256,512,1024}");
    config.queryNum = GetCompatibleEnvInt("CAGRA_QUERY_NUM", "CAGRA_SIFT_NQ", 100, 1, 4096);
    config.graphAccuracySamples =
        GetCompatibleEnvInt("CAGRA_GRAPH_ACCURACY_SAMPLES", "CAGRA_GRAPH_SAMPLES", 0, 0, config.dataNum);
    config.maxIterations = GetEnvInt("CAGRA_MAX_ITERATIONS", 20, 1, 1000);
    config.searchWarmups = GetCompatibleEnvInt("CAGRA_SEARCH_WARMUPS", "CAGRA_WARMUP", 3, 0, 1000);
    config.searchLoops = GetCompatibleEnvInt("CAGRA_SEARCH_LOOPS", "CAGRA_REPEAT", 20, 1, 10000);
    config.verbose = GetEnvBool("CAGRA_VERBOSE", true);
    config.runCpuGroundTruth = GetEnvBool("CAGRA_RUN_CPU_GT", true);
    config.siftDir = GetEnvString("CAGRA_SIFT_DIR", "");
    return config;
}

double ElapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string JoinPath(const std::string &directory, const std::string &fileName)
{
    if (directory.empty() || directory == ".")
    {
        return directory.empty() ? fileName : directory + "/" + fileName;
    }
    const char last = directory.back();
    return last == '/' || last == '\\' ? directory + fileName : directory + "/" + fileName;
}

std::vector<float> ReadFvecs(const std::string &path, int expectedDim, int requestedRows)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("failed to open fvecs file: " + path);
    }
    const std::streamoff fileBytes = input.tellg();
    const uint64_t recordBytes = static_cast<uint64_t>(expectedDim + 1) * sizeof(float);
    if (fileBytes <= 0 || static_cast<uint64_t>(fileBytes) % recordBytes != 0)
    {
        throw std::runtime_error("invalid fvecs file size for D=" + std::to_string(expectedDim) + ": " + path);
    }
    const uint64_t availableRows = static_cast<uint64_t>(fileBytes) / recordBytes;
    if (availableRows < static_cast<uint64_t>(requestedRows))
    {
        throw std::runtime_error("fvecs file has only " + std::to_string(availableRows) + " rows, but " +
                                 std::to_string(requestedRows) + " are required: " + path);
    }

    input.seekg(0, std::ios::beg);
    std::vector<float> result(static_cast<size_t>(requestedRows) * expectedDim);
    constexpr int batchRows = 4096;
    std::vector<char> buffer(static_cast<size_t>(batchRows) * recordBytes);
    for (int firstRow = 0; firstRow < requestedRows; firstRow += batchRows)
    {
        const int currentRows = std::min(batchRows, requestedRows - firstRow);
        const size_t currentBytes = static_cast<size_t>(currentRows) * recordBytes;
        if (!input.read(buffer.data(), static_cast<std::streamsize>(currentBytes)))
        {
            throw std::runtime_error("failed to read fvecs records from: " + path);
        }
        for (int offset = 0; offset < currentRows; ++offset)
        {
            const char *record = buffer.data() + static_cast<size_t>(offset) * recordBytes;
            int32_t rowDim = 0;
            std::memcpy(&rowDim, record, sizeof(rowDim));
            if (rowDim != expectedDim)
            {
                throw std::runtime_error("fvecs row " + std::to_string(firstRow + offset) +
                                         " has D=" + std::to_string(rowDim) +
                                         ", expected D=" + std::to_string(expectedDim) + ": " + path);
            }
            std::memcpy(result.data() + static_cast<size_t>(firstRow + offset) * expectedDim, record + sizeof(rowDim),
                        static_cast<size_t>(expectedDim) * sizeof(float));
        }
    }
    std::cout << "input_file path=" << path << " available_rows=" << availableRows << " loaded_rows=" << requestedRows
              << " dim=" << expectedDim << std::endl;
    return result;
}

class DisjointSet
{
   public:
    explicit DisjointSet(size_t size) : parents(size), ranks(size, 0) { std::iota(parents.begin(), parents.end(), 0U); }

    uint32_t Find(uint32_t node)
    {
        uint32_t root = node;
        while (parents[root] != root)
        {
            root = parents[root];
        }
        while (parents[node] != node)
        {
            const uint32_t parent = parents[node];
            parents[node] = root;
            node = parent;
        }
        return root;
    }

    void Unite(uint32_t left, uint32_t right)
    {
        left = Find(left);
        right = Find(right);
        if (left == right)
        {
            return;
        }
        if (ranks[left] < ranks[right])
        {
            std::swap(left, right);
        }
        parents[right] = left;
        if (ranks[left] == ranks[right])
        {
            ++ranks[left];
        }
    }

   private:
    std::vector<uint32_t> parents;
    std::vector<uint8_t> ranks;
};

struct GraphDiagnostics
{
    uint32_t weakComponents = 0;
    uint32_t zeroIndegreeNodes = 0;
    double largestWeakComponentRatio = 0.0;
    double validationMs = 0.0;
};

GraphDiagnostics ValidateAndAnalyzeGraph(const Config &config, const std::vector<uint32_t> &graph)
{
    if (graph.size() != static_cast<size_t>(config.dataNum) * config.graphDegree)
        throw std::runtime_error("graph size must equal dataNum * graphDegree");
    const auto start = Clock::now();
    DisjointSet components(config.dataNum);
    std::vector<uint32_t> indegrees(config.dataNum, 0);
    std::vector<uint32_t> seenInRow(config.dataNum, std::numeric_limits<uint32_t>::max());
    for (uint32_t row = 0; row < static_cast<uint32_t>(config.dataNum); ++row)
    {
        const size_t rowOffset = static_cast<size_t>(row) * config.graphDegree;
        for (int rank = 0; rank < config.graphDegree; ++rank)
        {
            const uint32_t neighbor = graph[rowOffset + rank];
            if (neighbor >= static_cast<uint32_t>(config.dataNum))
            {
                throw std::runtime_error("graph contains an out-of-range node at row " + std::to_string(row));
            }
            if (neighbor == row)
            {
                throw std::runtime_error("graph contains a self edge at row " + std::to_string(row));
            }
            if (seenInRow[neighbor] == row)
            {
                throw std::runtime_error("graph contains a duplicate edge at row " + std::to_string(row));
            }
            seenInRow[neighbor] = row;
            ++indegrees[neighbor];
            components.Unite(row, neighbor);
        }
    }

    std::vector<uint32_t> componentSizes(config.dataNum, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(config.dataNum); ++row)
    {
        ++componentSizes[components.Find(row)];
    }
    GraphDiagnostics result;
    uint32_t largestComponent = 0;
    for (uint32_t size : componentSizes)
    {
        if (size != 0)
        {
            ++result.weakComponents;
            largestComponent = std::max(largestComponent, size);
        }
    }
    result.zeroIndegreeNodes = static_cast<uint32_t>(std::count(indegrees.begin(), indegrees.end(), 0U));
    result.largestWeakComponentRatio = static_cast<double>(largestComponent) / config.dataNum;
    result.validationMs = ElapsedMs(start, Clock::now());
    return result;
}

struct ExactReferences
{
    std::vector<uint32_t> queryGroundTruth;
    std::vector<uint32_t> graphGroundTruth;
    std::vector<uint32_t> graphSampleRows;
    int graphGroundTruthWidth = 0;
    double indexAddMs = 0.0;
    double queryGroundTruthMs = 0.0;
    double graphGroundTruthMs = 0.0;
};

ExactReferences BuildExactReferences(const Config &config, const std::vector<float> &base,
                                     const std::vector<float> &queries)
{
    ExactReferences result;
    std::cout << "ground_truth phase=index_add start N=" << config.dataNum << " D=" << config.dim << std::endl;
    faiss::IndexFlatL2 exactIndex(config.dim);
    auto start = Clock::now();
    exactIndex.add(config.dataNum, base.data());
    result.indexAddMs = ElapsedMs(start, Clock::now());
    std::cout << "ground_truth phase=index_add done elapsed_ms=" << result.indexAddMs << std::endl;

    std::cout << "ground_truth phase=query_search start Q=" << config.queryNum << " TopK=" << config.topK << std::endl;
    std::vector<float> queryDistances(static_cast<size_t>(config.queryNum) * config.topK);
    std::vector<faiss::idx_t> queryLabels(static_cast<size_t>(config.queryNum) * config.topK);
    start = Clock::now();
    exactIndex.search(config.queryNum, queries.data(), config.topK, queryDistances.data(), queryLabels.data());
    result.queryGroundTruthMs = ElapsedMs(start, Clock::now());
    std::cout << "ground_truth phase=query_search done elapsed_ms=" << result.queryGroundTruthMs << std::endl;
    result.queryGroundTruth.resize(queryLabels.size());
    std::transform(queryLabels.begin(), queryLabels.end(), result.queryGroundTruth.begin(),
                   [](faiss::idx_t label) { return static_cast<uint32_t>(label); });

    if (config.graphAccuracySamples == 0)
    {
        return result;
    }

    result.graphSampleRows.resize(config.graphAccuracySamples);
    std::vector<float> sampleVectors(static_cast<size_t>(config.graphAccuracySamples) * config.dim);
    for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
    {
        const uint32_t row =
            static_cast<uint32_t>((static_cast<uint64_t>(sample) * config.dataNum) / config.graphAccuracySamples);
        result.graphSampleRows[sample] = row;
        std::copy_n(base.data() + static_cast<size_t>(row) * config.dim, config.dim,
                    sampleVectors.data() + static_cast<size_t>(sample) * config.dim);
    }

    result.graphGroundTruthWidth = std::min(32, config.dataNum - 1);
    const int candidates = result.graphGroundTruthWidth + 1;
    std::cout << "ground_truth phase=graph_sample_search start samples=" << config.graphAccuracySamples
              << " TopK=" << candidates << std::endl;
    std::vector<float> sampleDistances(static_cast<size_t>(config.graphAccuracySamples) * candidates);
    std::vector<faiss::idx_t> sampleLabels(static_cast<size_t>(config.graphAccuracySamples) * candidates);
    start = Clock::now();
    exactIndex.search(config.graphAccuracySamples, sampleVectors.data(), candidates, sampleDistances.data(),
                      sampleLabels.data());
    result.graphGroundTruthMs = ElapsedMs(start, Clock::now());
    std::cout << "ground_truth phase=graph_sample_search done elapsed_ms=" << result.graphGroundTruthMs << std::endl;
    result.graphGroundTruth.resize(static_cast<size_t>(config.graphAccuracySamples) * result.graphGroundTruthWidth);
    for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
    {
        int outputRank = 0;
        for (int rank = 0; rank < candidates && outputRank < result.graphGroundTruthWidth; ++rank)
        {
            const faiss::idx_t candidate = sampleLabels[static_cast<size_t>(sample) * candidates + rank];
            if (candidate < 0 || candidate >= config.dataNum ||
                candidate == static_cast<faiss::idx_t>(result.graphSampleRows[sample]))
            {
                continue;
            }
            result.graphGroundTruth[static_cast<size_t>(sample) * result.graphGroundTruthWidth + outputRank] =
                static_cast<uint32_t>(candidate);
            ++outputRank;
        }
        if (outputRank != result.graphGroundTruthWidth)
        {
            throw std::runtime_error("exact graph ground truth returned an incomplete row");
        }
    }
    return result;
}

double RecallAtK(const std::vector<uint32_t> &expected, const std::vector<uint32_t> &actual, int rows, int resultWidth,
                 int k)
{
    uint64_t hits = 0;
    for (int row = 0; row < rows; ++row)
    {
        std::unordered_set<uint32_t> truth(expected.begin() + static_cast<size_t>(row) * resultWidth,
                                           expected.begin() + static_cast<size_t>(row) * resultWidth + k);
        for (int rank = 0; rank < k; ++rank)
        {
            hits += truth.count(actual[static_cast<size_t>(row) * resultWidth + rank]);
        }
    }
    return static_cast<double>(hits) / (static_cast<uint64_t>(rows) * k);
}

struct SearchDiagnostics
{
    double validRatio = 0.0;
    double uniqueRatio = 0.0;
};

SearchDiagnostics AnalyzeSearch(const Config &config, const std::vector<uint32_t> &labels)
{
    uint64_t valid = 0;
    uint64_t unique = 0;
    for (int query = 0; query < config.queryNum; ++query)
    {
        std::unordered_set<uint32_t> seen;
        for (int rank = 0; rank < config.topK; ++rank)
        {
            const uint32_t label = labels[static_cast<size_t>(query) * config.topK + rank];
            if (label >= static_cast<uint32_t>(config.dataNum))
            {
                continue;
            }
            ++valid;
            unique += seen.insert(label).second;
        }
    }
    const double total = static_cast<double>(static_cast<uint64_t>(config.queryNum) * config.topK);
    SearchDiagnostics result;
    result.validRatio = valid / total;
    result.uniqueRatio = unique / total;
    return result;
}

double GraphNeighborRecallAtK(const Config &config, const ExactReferences &reference,
                              const std::vector<uint32_t> &graph, int k, bool scanFullDegree)
{
    if (graph.size() != static_cast<size_t>(config.dataNum) * config.graphDegree)
        throw std::runtime_error("graph size must equal dataNum * graphDegree");
    uint64_t hits = 0;
    const int graphRanks = scanFullDegree ? config.graphDegree : k;
    for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
    {
        std::unordered_set<uint32_t> truth(
            reference.graphGroundTruth.begin() + static_cast<size_t>(sample) * reference.graphGroundTruthWidth,
            reference.graphGroundTruth.begin() + static_cast<size_t>(sample) * reference.graphGroundTruthWidth + k);
        const size_t graphOffset = static_cast<size_t>(reference.graphSampleRows[sample]) * config.graphDegree;
        for (int rank = 0; rank < graphRanks; ++rank)
        {
            hits += truth.count(graph[graphOffset + rank]);
        }
    }
    return static_cast<double>(hits) / (static_cast<uint64_t>(config.graphAccuracySamples) * k);
}

struct SearchResult
{
    std::vector<uint32_t> labels;
    double warmupMs = 0.0;
    double averageSearchMs = 0.0;
    double qps = 0.0;
};

SearchResult RunNpuSearch(const Config &config, const std::vector<float> &queries,
                          faiss::ascend::AscendIndexCagra &index)
{
    SearchResult result;
    int ret = 0;
    std::vector<float> distances(static_cast<size_t>(config.queryNum) * config.topK,
                                 std::numeric_limits<float>::quiet_NaN());
    result.labels.assign(static_cast<size_t>(config.queryNum) * config.topK, std::numeric_limits<uint32_t>::max());
    auto start = Clock::now();
    for (int loop = 0; loop < config.searchWarmups; ++loop)
    {
        ret = index.Search(config.queryNum, queries.data(), config.topK, distances.data(), result.labels.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.Search warmup failed: " + std::to_string(ret));
        }
    }
    result.warmupMs = ElapsedMs(start, Clock::now());

    start = Clock::now();
    for (int loop = 0; loop < config.searchLoops; ++loop)
    {
        ret = index.Search(config.queryNum, queries.data(), config.topK, distances.data(), result.labels.data());
        if (ret != 0)
        {
            throw std::runtime_error("index.Search failed: " + std::to_string(ret));
        }
    }
    const double searchMs = ElapsedMs(start, Clock::now());
    if (searchMs <= 0) throw std::runtime_error("non-positive measured search interval");
    result.averageSearchMs = searchMs / config.searchLoops;
    result.qps = 1000.0 * config.queryNum * config.searchLoops / searchMs;
    const SearchDiagnostics diagnostics = AnalyzeSearch(config, result.labels);
    if (diagnostics.validRatio != 1.0 || diagnostics.uniqueRatio != 1.0)
    {
        throw std::runtime_error("NPU search returned invalid or duplicate IDs");
    }
    for (int query = 0; query < config.queryNum; ++query)
    {
        for (int rank = 0; rank < config.topK; ++rank)
        {
            const size_t offset = static_cast<size_t>(query) * config.topK + rank;
            if (!std::isfinite(distances[offset]) || (rank > 0 && distances[offset] < distances[offset - 1]))
            {
                throw std::runtime_error("NPU search returned non-finite or unsorted distances");
            }
        }
    }
    return result;
}

void RunSift1M(const Config &config)
{
    std::cout << std::fixed << std::setprecision(4) << "IndexSDK CAGRA SIFT1M N=" << config.dataNum
              << " D=" << config.dim << " M=" << config.intermediateDegree << " G=" << config.graphDegree
              << " Q=" << config.queryNum << " topK=" << config.topK << " device=" << config.deviceId
              << " graph_samples=" << config.graphAccuracySamples << " max_iterations=" << config.maxIterations
              << " warmup=" << config.searchWarmups << " repeat=" << config.searchLoops
              << " api=Init/Train/Search gt=" << (config.runCpuGroundTruth ? "CPU_IndexFlatL2" : "skipped")
              << std::endl;
    const char *modelPath = std::getenv("MX_INDEX_MODELPATH");
    std::cout << "MX_INDEX_MODELPATH=" << (modelPath == nullptr ? "<unset>" : modelPath) << std::endl;
    std::cout << "cagra_phase data_loading start" << std::endl;
    auto start = Clock::now();
    const auto base = ReadFvecs(JoinPath(config.siftDir, "sift_base.fvecs"), config.dim, config.dataNum);
    const auto queries = ReadFvecs(JoinPath(config.siftDir, "sift_query.fvecs"), config.dim, config.queryNum);
    if (!std::all_of(base.begin(), base.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(queries.begin(), queries.end(), [](float x) { return std::isfinite(x); }))
    {
        throw std::runtime_error("SIFT input contains non-finite values");
    }
    std::cout << "cagra_phase data_loading done elapsed_ms=" << ElapsedMs(start, Clock::now()) << std::endl;
    std::cout << "cagra_phase train start" << std::endl;
    faiss::ascend::AscendIndexCagra index;
    faiss::ascend::AscendIndexCagraConfig buildConfig;
    buildConfig.graphDegree = config.graphDegree;
    buildConfig.intermediateDegree = config.intermediateDegree;
    buildConfig.maxIterations = config.maxIterations;
    buildConfig.terminationThreshold = config.terminationThreshold;
    buildConfig.guaranteeConnectivity = config.guaranteeConnectivity;
    buildConfig.verbose = config.verbose;
    int ret = index.Init(config.dim, {config.deviceId}, buildConfig);
    if (ret != 0) throw std::runtime_error("index.Init failed: " + std::to_string(ret));
    start = Clock::now();
    ret = index.Train(config.dataNum, base.data());
    const double trainMs = ElapsedMs(start, Clock::now());
    if (ret != 0) throw std::runtime_error("index.Train failed: " + std::to_string(ret));
    std::cout << "cagra_phase train done" << std::endl;
    std::vector<uint32_t> graph;
    if (index.GetKnnGraph(graph) != 0) throw std::runtime_error("index.GetKnnGraph failed");
    std::cout << "train_performance train_total_ms=" << trainMs
              << " scope=graph_build_plus_load_plus_base_quantization vectors_per_s="
              << 1000.0 * config.dataNum / trainMs << std::endl;
    const auto graphDiagnostics = ValidateAndAnalyzeGraph(config, graph);
    std::cout << "graph_validation edges=" << graph.size() << " invalid=0 self_loops=0 duplicate_rows=0"
              << " weak_components=" << graphDiagnostics.weakComponents
              << " largest_weak_component_ratio=" << graphDiagnostics.largestWeakComponentRatio
              << " zero_indegree_nodes=" << graphDiagnostics.zeroIndegreeNodes << std::endl;
    ExactReferences exact;
    if (config.runCpuGroundTruth)
    {
        std::cout << "cagra_phase cpu_exact_ground_truth start" << std::endl;
        exact = BuildExactReferences(config, base, queries);
        std::cout << "ground_truth exact_index_add_ms=" << exact.indexAddMs
                  << " query_gt_ms=" << exact.queryGroundTruthMs << " graph_sample_gt_ms=" << exact.graphGroundTruthMs
                  << std::endl;
    }
    else
    {
        std::cout << "cagra_phase cpu_exact_ground_truth skipped reason=CAGRA_RUN_CPU_GT=0" << std::endl;
    }
    if (config.runCpuGroundTruth && config.graphAccuracySamples > 0)
    {
        std::cout << "graph_accuracy samples=" << config.graphAccuracySamples << " sampling=even";
        for (const int k : {1, 10, 32})
        {
            std::cout << " prefix_recall@" << k << '=' << GraphNeighborRecallAtK(config, exact, graph, k, false);
        }
        for (const int k : {1, 10, 32})
        {
            std::cout << " degree_coverage@" << k << '=' << GraphNeighborRecallAtK(config, exact, graph, k, true);
        }
        std::cout << std::endl;
    }
    std::cout << "cagra_phase search start" << std::endl;
    const auto search = RunNpuSearch(config, queries, index);
    const auto diagnostics = AnalyzeSearch(config, search.labels);
    std::cout << "cagra_phase search done" << std::endl;
    std::cout << "search_warmup warmup_total_ms=" << search.warmupMs << std::endl;
    std::cout << "search_performance scope=query_quantization_plus_search average_batch_ms=" << search.averageSearchMs
              << " qps=" << search.qps << " topK=" << config.topK << " checked_results=" << search.labels.size();
    if (config.runCpuGroundTruth)
    {
        for (const int k : {1, 10, 32})
        {
            if (k <= config.topK)
            {
                std::cout << " recall@" << k << '='
                          << RecallAtK(exact.queryGroundTruth, search.labels, config.queryNum, config.topK, k);
            }
        }
        if (config.topK != 1 && config.topK != 10 && config.topK != 32)
        {
            std::cout << " recall@" << config.topK << '='
                      << RecallAtK(exact.queryGroundTruth, search.labels, config.queryNum, config.topK, config.topK);
        }
    }
    else
    {
        std::cout << " recall=not_evaluated";
    }
    std::cout << " valid_ratio=" << diagnostics.validRatio << " unique_ratio=" << diagnostics.uniqueRatio << std::endl;
    std::cout << "PASS: SIFT1M NPU build, quantization and search completed"
              << (config.runCpuGroundTruth ? "" : " (Recall not evaluated)") << std::endl;
}

TEST(TestAscendIndexCagra, RejectsInvalidGraphSize)
{
    Config config;
    config.dataNum = 129;
    const size_t expected = static_cast<size_t>(config.dataNum) * config.graphDegree;
    for (size_t edges : {size_t{0}, expected - 1, expected + 1})
    {
        const std::vector<uint32_t> graph(edges);
        EXPECT_THROW(ValidateAndAnalyzeGraph(config, graph), std::runtime_error);
        EXPECT_THROW(GraphNeighborRecallAtK(config, ExactReferences{}, graph, 1, false), std::runtime_error);
    }
}

TEST(TestAscendIndexCagra, CpuExactGroundTruth)
{
    // Exercise the linked CPU Faiss/BLAS independently, without initializing an NPU index.
    Config config;
    config.dataNum = 1024;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> random(0.0F, 1.0F);
    std::vector<float> base(static_cast<size_t>(config.dataNum) * config.dim);
    for (auto &value : base) value = random(rng);
    for (const int queryNum : {1, 100})
    {
        config.queryNum = queryNum;
        config.graphAccuracySamples = queryNum == 1 ? 0 : 100;
        std::vector<float> queries(base.begin(), base.begin() + static_cast<size_t>(queryNum) * config.dim);
        const auto reference = BuildExactReferences(config, base, queries);
        ASSERT_EQ(reference.queryGroundTruth.size(), static_cast<size_t>(queryNum) * config.topK);
        for (int query = 0; query < queryNum; ++query)
        {
            EXPECT_EQ(reference.queryGroundTruth[static_cast<size_t>(query) * config.topK],
                      static_cast<uint32_t>(query));
        }
        ASSERT_EQ(reference.graphGroundTruth.size(),
                  static_cast<size_t>(config.graphAccuracySamples) * reference.graphGroundTruthWidth);
        for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
        {
            const auto first =
                reference.graphGroundTruth.begin() + static_cast<size_t>(sample) * reference.graphGroundTruthWidth;
            std::unordered_set<uint32_t> seen;
            for (int rank = 0; rank < reference.graphGroundTruthWidth; ++rank)
            {
                EXPECT_LT(first[rank], static_cast<uint32_t>(config.dataNum));
                EXPECT_NE(first[rank], reference.graphSampleRows[sample]);
                EXPECT_TRUE(seen.insert(first[rank]).second);
            }
        }
    }
}

TEST(TestAscendIndexCagra, FullTopKDoesNotChangeGraphGroundTruthWidth)
{
    for (int count : {256, 512, 1024})
    {
        Config config;
        config.dim = 64;
        config.dataNum = count;
        config.queryNum = 2;
        config.topK = count;
        config.graphAccuracySamples = 3;
        std::vector<float> base(static_cast<size_t>(count) * config.dim, 0.0F);
        for (int row = 0; row < count; ++row) base[static_cast<size_t>(row) * config.dim] = row;
        std::vector<float> queries(static_cast<size_t>(config.queryNum) * config.dim);
        std::copy_n(base.data(), config.dim, queries.data());
        std::copy_n(base.data() + static_cast<size_t>(count - 1) * config.dim, config.dim, queries.data() + config.dim);
        const auto exact = BuildExactReferences(config, base, queries);
        ASSERT_EQ(exact.queryGroundTruth.size(), static_cast<size_t>(config.queryNum) * count);
        ASSERT_EQ(exact.graphGroundTruthWidth, 32);
        ASSERT_EQ(exact.graphGroundTruth.size(), static_cast<size_t>(config.graphAccuracySamples) * 32);
        for (int query = 0; query < config.queryNum; ++query)
        {
            const auto begin = exact.queryGroundTruth.begin() + static_cast<size_t>(query) * count;
            const std::unordered_set<uint32_t> row(begin, begin + count);
            ASSERT_EQ(row.size(), static_cast<size_t>(count));
            for (uint32_t node : row) ASSERT_TRUE(node < static_cast<uint32_t>(count));
        }
        std::vector<uint32_t> graph(static_cast<size_t>(count) * config.graphDegree, 0);
        for (int sample = 0; sample < config.graphAccuracySamples; ++sample)
        {
            const auto begin = exact.graphGroundTruth.begin() + static_cast<size_t>(sample) * 32;
            const std::unordered_set<uint32_t> row(begin, begin + 32);
            ASSERT_EQ(row.size(), 32U);
            ASSERT_TRUE(row.count(exact.graphSampleRows[sample]) == 0);
            std::copy_n(begin, 32,
                        graph.begin() + static_cast<size_t>(exact.graphSampleRows[sample]) * config.graphDegree);
        }
        EXPECT_FLOAT_EQ(GraphNeighborRecallAtK(config, exact, graph, 32, false), 1.0F);
    }
}

TEST(TestAscendIndexCagra, Sift1M)
{
    if (GetEnvString("CAGRA_SIFT_DIR", "").empty())
    {
        GTEST_SKIP() << "Set CAGRA_SIFT_DIR to run the SIFT1M end-to-end test";
    }
    try
    {
        RunSift1M(ReadConfig());
    }
    catch (const std::exception &error)
    {
        FAIL() << error.what();
    }
}
TEST(TestAscendIndexCagra, UnifiedLifecycle)
{
    using faiss::ascend::AscendIndexCagra;
    using faiss::ascend::AscendIndexCagraConfig;
    const char *enabled = std::getenv("CAGRA_RUN_UNIFIED_UT");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        GTEST_SKIP() << "Set CAGRA_RUN_UNIFIED_UT=1 after generating matching build/search models";
    }
    auto envInt = [](const char *name, int fallback)
    {
        const char *value = std::getenv(name);
        return value == nullptr ? fallback : std::stoi(value);
    };
    const int dim = envInt("CAGRA_TEST_DIM", 128);
    const int count = envInt("CAGRA_TEST_N", 10000);
    const int topk = envInt("CAGRA_TEST_TOPK", 32);
    const int secondTopk = envInt("CAGRA_TEST_SECOND_TOPK", 0);
    ASSERT_GT(count, 128);
    ASSERT_GT(topk, 0);
    ASSERT_LE(topk, std::min(1024, count));
    ASSERT_TRUE(IsSupportedTopK(topk));
    ASSERT_TRUE(secondTopk == 0 || (IsSupportedTopK(secondTopk) && secondTopk <= count));
    AscendIndexCagraConfig config;
    config.graphDegree = envInt("CAGRA_TEST_GRAPH_DEGREE", 64);
    AscendIndexCagra index;
    ASSERT_EQ(index.Init(dim, {envInt("CAGRA_DEVICE_ID", 0)}, config), 0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> random(0.0F, 1.0F);
    std::vector<float> data(static_cast<size_t>(count) * dim);
    for (auto &v : data) v = random(rng);
    ASSERT_EQ(index.Train(count, data.data()), 0);
    EXPECT_TRUE(index.IsTrained());
    EXPECT_EQ(index.GetSize(), count);
    std::vector<uint32_t> graph;
    ASSERT_EQ(index.GetKnnGraph(graph), 0);
    EXPECT_EQ(graph.size(), static_cast<size_t>(count) * config.graphDegree);
    // Optional artifact for file export validation; keep the file for inspection.
    const char *graphFile = std::getenv("CAGRA_TEST_GRAPH_FILE");
    if (graphFile != nullptr && graphFile[0] != '\0')
    {
        ASSERT_EQ(index.SaveGraphToFile(graphFile), 0);
        std::ifstream input(graphFile, std::ios::binary | std::ios::ate);
        ASSERT_TRUE(input.is_open());
        ASSERT_EQ(input.tellg(), static_cast<std::streamoff>(graph.size() * sizeof(uint32_t)));
        input.seekg(0);
        std::vector<uint32_t> saved(graph.size());
        ASSERT_TRUE(input.read(reinterpret_cast<char *>(saved.data()), saved.size() * sizeof(uint32_t)));
        EXPECT_EQ(saved, graph);
    }
    auto checkSearch = [&](int k)
    {
        constexpr int queries = 100;  // Exercises 64+32+4 query pagination.
        std::vector<float> distances(queries * k, std::numeric_limits<float>::quiet_NaN());
        std::vector<uint32_t> labels(queries * k, std::numeric_limits<uint32_t>::max());
        ASSERT_EQ(index.Search(queries, data.data(), k, distances.data(), labels.data()), 0);
        for (int q = 0; q < queries; ++q)
        {
            std::vector<uint32_t> row(labels.begin() + q * k, labels.begin() + (q + 1) * k);
            for (int rank = 0; rank < k; ++rank)
            {
                ASSERT_TRUE(row[rank] < static_cast<uint32_t>(count));
                ASSERT_TRUE(std::isfinite(distances[q * k + rank]));
                if (rank > 0) ASSERT_LE(distances[q * k + rank - 1], distances[q * k + rank]);
            }
            std::sort(row.begin(), row.end());
            ASSERT_EQ(std::adjacent_find(row.begin(), row.end()), row.end());
        }
        std::cout << "topk_validation topK=" << k << " queries=" << queries << " checked_results=" << labels.size()
                  << " valid_unique_finite_sorted=true" << std::endl;
    };
    checkSearch(topk);
    if (secondTopk > 0 && secondTopk <= std::min(1024, count))
    {
        checkSearch(secondTopk);
        checkSearch(topk);  // Switch back on the same trained index before Reset.
    }
    EXPECT_NE(index.Add(count, data.data()), 0);
    EXPECT_TRUE(index.IsTrained());
    EXPECT_EQ(index.GetSize(), count);
    ASSERT_EQ(index.Reset(), 0);
    EXPECT_FALSE(index.IsTrained());
    EXPECT_EQ(index.GetSize(), 0);
    ASSERT_EQ(index.Add(count, data.data()), 0);  // Add(n,data) also builds and quantizes after Reset.
    checkSearch(topk);
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
