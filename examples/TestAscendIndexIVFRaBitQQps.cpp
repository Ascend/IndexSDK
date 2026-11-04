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

/*
 * IVFRaBitQ 多比特、多卡 QPS 与召回率测试，以下命令均在仓库根目录执行。
 * 前置条件：安装配套 host/ops 库及匹配 dim、nlist、quant-bits 的算子模型；
 * A3 需要 aicpu / ivfflat / ivfrabitq 算子，编译需要 Faiss 1.14.1。
 *
 * 编译：bash examples/build.sh TestAscendIndexIVFRaBitQQps
 * 参数帮助：./examples/build/TestAscendIndexIVFRaBitQQps --help
 *
 * 示例：两卡分片，依次测试 1/2/4-bit；首次建库并保存索引和 GT，后续运行复用。
 * for bits in 1 2 4; do
 *   ./examples/build/TestAscendIndexIVFRaBitQQps \
 *     --devices 0,1 --search-mode shard --quant-bits "$bits" --refine none \
 *     --dim 128 --ntotal 100000 --nlist 1024 --nprobe 32 --nq 1,8,32 \
 *     --candidate-topk 300 --target-topk 100 --recall-nq 10 --warmup 5 --loops 20 \
 *     --index-path "./rabitq_${bits}bit.index" --gt-path ./rabitq_100k.gt \
 *     --csv "./rabitq_${bits}bit.csv"
 * done
 * 注意：每次运行前需准备好当前 bits 对应的模型，不同位数使用不同索引文件。
 *
 * 多卡：shard 将底库分布到多卡，每批查询广播到各卡并合并结果；
 *       replica 每卡存完整底库，将一批 nq 分配到各卡，无需合并（nq=1 仅用一卡）。
 * 计时：wait 按 --loops 执行；sustain 按 --duration 秒连续执行同步 search。
 *       例如在上方命令追加 --qps-mode sustain --duration 60。
 * 精排：--refine none 测粗排；npu 使用 SDK 精排，cpu 使用本工具的 Host 精排。
 *       npu/cpu 均需 --src-bin；原文为无文件头的 float32 [ntotal, dim] 数据。
 *       原文存在时用其建库，不存在时建库阶段生成并保存；加载索引时精排原文必须存在。
 *       npu 精排不支持进程内 nprobe/候选 TopK 矩阵；--npu-k 控制 CPU 精排候选数，
 *       SDK 精排的候选扩展由 --refine-alpha 控制（默认按 npu-k / candidate-topk 计算）。
 * 参数：--nprobe 和 --candidate-topk 可用逗号列表；2-bit dim<=512，4-bit dim<=256，均为 L2。
 * 复用：加载已有索引时必须提供匹配的 --gt-path；--rebuild 1 强制重建索引。
 *       同一底库可用 --reuse-gt 1 复用 GT，但 ntotal、dim、seed、原文和召回参数须保持一致。
 * 大库：5 亿底库可设 --ntotal 500000000 --nlist 32768；同时调整模型、资源及分块大小。
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/ascend/AscendIndexIVFRaBitQ.h>
#include <faiss/index_io.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int kDefaultDim = 128;
constexpr size_t kDefaultNtotal = 500000000;
constexpr int kDefaultNlist = 32768;
constexpr int kDefaultNprobe = 200;
constexpr int kDefaultK = 300;
constexpr int kDefaultNpuK = 5000;
constexpr int kDefaultWarmup = 5;
constexpr int kDefaultLoops = 20;
constexpr int kDefaultResourceMb = 4096;
constexpr int kDefaultAddChunk = 1000000;
constexpr int kDefaultRecallNq = 100;
constexpr int kTrainMultiplier = 40;
constexpr size_t kMaxTrainNum = 7000000;
constexpr uint32_t kDefaultSeed = 5678;
constexpr uint32_t kQuerySeed = 9012;
constexpr int kMilliSecond = 1000;
constexpr double kNpuHostOverhead = 1.15;
constexpr int kMemGb = 1024 * 1024;

enum class RefineMode
{
    None = 0,
    Npu = 1,
    Cpu = 2
};

std::vector<int> g_devices = {0};
int g_dim = kDefaultDim;
size_t g_ntotal = kDefaultNtotal;
int g_nlist = kDefaultNlist;
int g_nprobe = kDefaultNprobe;
int g_k = kDefaultK;
std::vector<int> g_nprobes = {kDefaultNprobe};
std::vector<int> g_candidateTopks = {kDefaultK};
int g_npuK = kDefaultNpuK;
int g_quantBits = 1;
int g_recallK = 0;
int g_warmup = kDefaultWarmup;
int g_loops = kDefaultLoops;
int g_durationSec = 0;
int g_resourceMb = kDefaultResourceMb;
int g_addChunk = kDefaultAddChunk;
int g_recallNq = kDefaultRecallNq;
int g_useKmeansPp = 1;
int g_niter = 10;
int g_trainSlice = 0;
int g_searchReplica = 0;
int g_attachSrc = 0;  // 1=replica 共享同一份精排原文（attachRefineSources 零拷贝），0=每副本深拷贝
int g_qpsSustain = 0;
uint32_t g_seed = kDefaultSeed;
float g_refineAlpha = -1.0f;
std::vector<int> g_nq = {1, 8, 16, 32};
std::vector<RefineMode> g_refine = {RefineMode::None, RefineMode::Npu};
std::string g_indexPath = "./ivfrabitq.index";
std::string g_srcBin;
std::string g_gtPath;
std::string g_csvPath;
int g_rebuild = 0;
int g_reuseGt = 0;

const int kValidNlists[] = {1024, 2048, 4096, 8192, 10048, 16384, 32768};
const int kValidDims[] = {64, 128, 256, 384, 512, 768, 1024};

inline double GetMillisecs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

std::string DevicesJson();

void PrintUsage(const char *prog)
{
    printf(
        "Usage: %s [--devices 0] [--dim 128] [--ntotal 500000000] [--nlist 32768] [--nprobe 200,400]\n"
        "  [--candidate-topk 300,512] [--target-topk 100] [--npu-k 5000] [--quant-bits 1]\n"
        "  [--nq 1,8,16,32] [--recall-nq 100] [--warmup 5] [--loops 20]\n"
        "  [--resource-mb 4096] [--add-chunk 1000000] [--index-path ./ivfrabitq.index]\n"
        "  [--src-bin path] [--gt-path path] [--rebuild 0] [--reuse-gt 0] [--csv path]\n"
        "  [--seed 5678] [--use-kmeans-pp 1] [--niter 10] [--train-mode replica|slice]\n"
        "  [--search-mode shard|replica] [--qps-mode wait|sustain] [--duration 60] [gtest flags]\n",
        prog);
    printf("  none: NPU search k (needRefine=false)\n");
    printf("  npu:  search k with SDK refine (needs --src-bin for srcIndexes)\n");
    printf("  cpu:  NPU coarse search k=npu-k, host-side refine on --src-bin (mmap, multi-thread)\n");
    printf("  Existing --src-bin is raw headerless float32 input in row-major [ntotal, dim] layout.\n");
    printf("  Comma-separated --nprobe/--candidate-topk run a matrix after one index build/load.\n");
    printf("  If --index-path exists and --rebuild=0, skip train/add and load the index.\n");
    printf(
        "  --reuse-gt 1: reuse an existing GT file while rebuilding an index; only use with identical data "
        "parameters.\n");
    printf("  --niter: coarse k-means iterations (default 10). --train-mode only affects train.\n");
    printf("  --search-mode shard: one index, SDK shards data and broadcasts queries (default).\n");
    printf("  --search-mode replica: one full copy per device; split nq across copies (no merge).\n");
    printf("  --qps-mode wait: stop-and-wait, --loops batches (default).\n");
    printf("  --qps-mode sustain: back-to-back search for --duration seconds (default 60).\n");
    printf("  --candidate-topk: approximate candidates returned by RaBitQ (--k is a legacy alias).\n");
    printf("  --target-topk: exact GT cutoff; candidate coverage equals Recall@target after exact rerank\n");
    printf("                 (--recall-k is a legacy alias).\n");
    printf("  Multi-bit POC: A3 L2, 2-bit dim <= 512, 4-bit dim <= 256, refine=none.\n");
}

bool ParseLong(const char *text, long &out)
{
    if (text == nullptr)
    {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
    {
        return false;
    }
    out = value;
    return true;
}

bool ParseUll(const char *text, unsigned long long &out)
{
    if (text == nullptr || text[0] == '-')
    {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
    {
        return false;
    }
    out = value;
    return true;
}

bool ParseIntList(const std::string &text, std::vector<int> &out, bool allowZero)
{
    std::istringstream input(text);
    std::string token;
    std::vector<int> values;
    while (std::getline(input, token, ','))
    {
        long value = 0;
        if (!ParseLong(token.c_str(), value) || value > INT_MAX || value < (allowZero ? 0 : 1))
        {
            printf("invalid list value: %s\n", token.c_str());
            return false;
        }
        values.push_back(static_cast<int>(value));
    }
    if (values.empty() || text.back() == ',')
    {
        return false;
    }
    out = values;
    return true;
}

bool ParseRefineList(const std::string &text, std::vector<RefineMode> &out)
{
    std::istringstream input(text);
    std::string token;
    std::vector<RefineMode> values;
    while (std::getline(input, token, ','))
    {
        if (token == "none" || token == "0")
        {
            values.push_back(RefineMode::None);
        }
        else if (token == "npu" || token == "1")
        {
            values.push_back(RefineMode::Npu);
        }
        else if (token == "cpu" || token == "2")
        {
            values.push_back(RefineMode::Cpu);
        }
        else
        {
            printf("invalid refine token: %s (use none,npu,cpu)\n", token.c_str());
            return false;
        }
    }
    if (values.empty() || text.back() == ',')
    {
        return false;
    }
    out.swap(values);
    return true;
}

bool HasMode(RefineMode mode) { return std::find(g_refine.begin(), g_refine.end(), mode) != g_refine.end(); }

bool InSet(int value, const int *arr, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (arr[i] == value)
        {
            return true;
        }
    }
    return false;
}

bool ParseCli(int *argc, char **argv)
{
    const std::unordered_map<std::string, int *> integers = {{"--dim", &g_dim},
                                                             {"--nlist", &g_nlist},
                                                             {"--npu-k", &g_npuK},
                                                             {"--quant-bits", &g_quantBits},
                                                             {"--recall-k", &g_recallK},
                                                             {"--target-topk", &g_recallK},
                                                             {"--recall-nq", &g_recallNq},
                                                             {"--warmup", &g_warmup},
                                                             {"--loops", &g_loops},
                                                             {"--duration", &g_durationSec},
                                                             {"--resource-mb", &g_resourceMb},
                                                             {"--add-chunk", &g_addChunk},
                                                             {"--rebuild", &g_rebuild},
                                                             {"--reuse-gt", &g_reuseGt},
                                                             {"--niter", &g_niter},
                                                             {"--use-kmeans-pp", &g_useKmeansPp}};
    const std::unordered_map<std::string, std::string *> strings = {
        {"--index-path", &g_indexPath}, {"--src-bin", &g_srcBin}, {"--gt-path", &g_gtPath}, {"--csv", &g_csvPath}};
    const std::unordered_map<std::string, std::vector<int> *> lists = {{"--devices", &g_devices},
                                                                       {"--nq", &g_nq},
                                                                       {"--nprobe", &g_nprobes},
                                                                       {"--k", &g_candidateTopks},
                                                                       {"--candidate-topk", &g_candidateTopks}};
    std::vector<char *> kept{argv[0]};
    for (int i = 1; i < *argc; ++i)
    {
        const std::string argument(argv[i]);
        if (argument == "--help")
        {
            PrintUsage(argv[0]);
            return false;
        }
        if (argument.compare(0, 8, "--gtest_") == 0)
        {
            kept.push_back(argv[i]);
            continue;
        }
        const size_t equal = argument.find('=');
        const std::string flag = argument.substr(0, equal);
        const auto integer = integers.find(flag);
        const auto string = strings.find(flag);
        const auto list = lists.find(flag);
        const bool isMode = flag == "--train-mode" || flag == "--search-mode" || flag == "--qps-mode";
        if (integer == integers.end() && string == strings.end() && list == lists.end() && !isMode &&
            flag != "--refine" && flag != "--refine-alpha" && flag != "--ntotal" && flag != "--seed" &&
            flag != "--attach-src")
        {
            printf("unknown argument: %s\n", argument.c_str());
            PrintUsage(argv[0]);
            return false;
        }
        if (equal == std::string::npos && i + 1 >= *argc)
        {
            printf("missing value for %s\n", flag.c_str());
            PrintUsage(argv[0]);
            return false;
        }
        const std::string value = equal == std::string::npos ? argv[++i] : argument.substr(equal + 1);
        if (string != strings.end())
        {
            *string->second = value;
        }
        else if (list != lists.end())
        {
            if (!ParseIntList(value, *list->second, flag == "--devices"))
            {
                PrintUsage(argv[0]);
                return false;
            }
            g_nprobe = g_nprobes.front();
            g_k = g_candidateTopks.front();
        }
        else if (flag == "--refine")
        {
            if (!ParseRefineList(value, g_refine))
            {
                PrintUsage(argv[0]);
                return false;
            }
        }
        else if (isMode)
        {
            int *target = flag == "--train-mode"    ? &g_trainSlice
                          : flag == "--search-mode" ? &g_searchReplica
                                                    : &g_qpsSustain;
            const char *zero = flag == "--train-mode" ? "replica" : flag == "--search-mode" ? "shard" : "wait";
            const char *one = flag == "--train-mode" ? "slice" : flag == "--search-mode" ? "replica" : "sustain";
            if (value == zero || value == "0" || (flag == "--qps-mode" && value == "stop"))
            {
                *target = 0;
            }
            else if (value == one || value == "1")
            {
                *target = 1;
            }
            else
            {
                printf("%s must be %s or %s\n", flag.c_str() + 2, zero, one);
                return false;
            }
        }
        else if (flag == "--refine-alpha")
        {
            char *end = nullptr;
            errno = 0;
            const float alpha = strtof(value.c_str(), &end);
            if (errno != 0 || end == value.c_str() || *end != '\0' || alpha < 1.0f)
            {
                printf("refine-alpha must be >= 1\n");
                return false;
            }
            g_refineAlpha = alpha;
        }
        else if (flag == "--ntotal")
        {
            unsigned long long count = 0;
            if (!ParseUll(value.c_str(), count) || count == 0)
            {
                printf("ntotal must be > 0\n");
                return false;
            }
            g_ntotal = static_cast<size_t>(count);
        }
        else
        {
            long number = 0;
            if (!ParseLong(value.c_str(), number) || number < 0 || number > INT_MAX ||
                (flag == "--attach-src" && number > 1))
            {
                printf("invalid value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            if (flag == "--seed")
            {
                g_seed = static_cast<uint32_t>(number);
            }
            else if (flag == "--attach-src")
            {
                g_attachSrc = static_cast<int>(number);
            }
            else
            {
                *integer->second = static_cast<int>(number);
            }
        }
    }
    std::copy(kept.begin(), kept.end(), argv);
    *argc = static_cast<int>(kept.size());
    argv[*argc] = nullptr;
    return true;
}

bool ValidateCli()
{
    if (g_devices.empty())
    {
        printf("devices must not be empty\n");
        return false;
    }
    if (!InSet(g_dim, kValidDims, static_cast<int>(sizeof(kValidDims) / sizeof(kValidDims[0]))))
    {
        printf("unsupported dim %d\n", g_dim);
        return false;
    }
    if (!InSet(g_nlist, kValidNlists, static_cast<int>(sizeof(kValidNlists) / sizeof(kValidNlists[0]))))
    {
        printf("unsupported nlist %d\n", g_nlist);
        return false;
    }
    if (std::any_of(g_nprobes.begin(), g_nprobes.end(), [](int value) { return value <= 0 || value > g_nlist; }))
    {
        printf("every nprobe must be in (0, nlist]\n");
        return false;
    }
    if (std::any_of(g_candidateTopks.begin(), g_candidateTopks.end(), [](int value) { return value <= 0; }) ||
        g_npuK <= 0)
    {
        printf("k and npu-k must be > 0\n");
        return false;
    }
    if (g_quantBits != 1 && g_quantBits != 2 && g_quantBits != 4)
    {
        printf("quant-bits must be 1, 2, or 4\n");
        return false;
    }
    if (g_recallK == 0)
    {
        g_recallK = *std::min_element(g_candidateTopks.begin(), g_candidateTopks.end());
    }
    const int minCandidateTopk = *std::min_element(g_candidateTopks.begin(), g_candidateTopks.end());
    const int maxCandidateTopk = *std::max_element(g_candidateTopks.begin(), g_candidateTopks.end());
    if (g_recallK < 0 || g_recallK > minCandidateTopk)
    {
        printf("target-topk (%d) must be in (0, minimum candidate-topk=%d]\n", g_recallK, minCandidateTopk);
        return false;
    }
    if (g_quantBits == 2 && g_dim > 512)
    {
        printf("2-bit POC only supports dim <= 512\n");
        return false;
    }
    if (g_quantBits == 4 && g_dim > 256)
    {
        printf("4-bit POC only supports dim <= 256\n");
        return false;
    }
    // multi-bit + npu 精排可用：SDK refine 用原文（srcIndexes）重排，与 quantBits 无关；
    // multi-bit 的真实约束（L2、dim 上限）由 Impl 层校验。
    if (HasMode(RefineMode::Npu) && (g_nprobes.size() > 1 || g_candidateTopks.size() > 1))
    {
        printf("npu refine does not support an in-process nprobe/TopK matrix; use --refine none\n");
        return false;
    }
    if (g_npuK < maxCandidateTopk)
    {
        printf("npu-k (%d) must be >= maximum candidate-topk (%d)\n", g_npuK, maxCandidateTopk);
        return false;
    }
    if (g_addChunk <= 0 || g_recallNq <= 0 || g_loops <= 0)
    {
        printf("add-chunk, recall-nq and loops must be > 0\n");
        return false;
    }
    if (g_warmup < 0)
    {
        printf("warmup must be >= 0\n");
        return false;
    }
    if (g_durationSec < 0)
    {
        printf("duration must be >= 0\n");
        return false;
    }
    if (g_qpsSustain != 0 && g_durationSec <= 0)
    {
        g_durationSec = 60;
        printf("qps-mode=sustain: --duration defaulted to 60s\n");
    }
    if (g_qpsSustain == 0 && g_durationSec > 0)
    {
        printf("qps-mode=wait: --duration ignored, using --loops=%d\n", g_loops);
    }
    const int maxNq = *std::max_element(g_nq.begin(), g_nq.end());
    if (static_cast<size_t>(maxNq) > g_ntotal || static_cast<size_t>(g_recallNq) > g_ntotal)
    {
        printf("nq and recall-nq must be <= ntotal\n");
        return false;
    }
    if (HasMode(RefineMode::Npu) && g_srcBin.empty())
    {
        printf("npu refine requires --src-bin\n");
        return false;
    }
    if (HasMode(RefineMode::Cpu) && g_srcBin.empty())
    {
        printf("cpu refine requires --src-bin\n");
        return false;
    }
    if (HasMode(RefineMode::Cpu) && g_npuK > 5000)
    {
        printf("cpu refine requires --npu-k <= 5000 (NPU coarse search MAX_K)\n");
        return false;
    }
    // --src-bin 与 refine 档位解耦：存在则读取复用，不存在则建库时落盘；仅 npu 档加载时强依赖（见 LoadSrcBin）
    if (g_indexPath.empty())
    {
        printf("index-path must not be empty\n");
        return false;
    }
    if (g_refineAlpha < 0.0f)
    {
        // 矩阵模式下 candidate-topk 有多个取值，用最大值计算默认 alpha，避免候选扩展被低估
        g_refineAlpha = static_cast<float>(g_npuK) / static_cast<float>(maxCandidateTopk);
    }
    if (g_niter <= 0 || g_niter > 65536)
    {
        printf("niter must be in (0, 65536]\n");
        return false;
    }
    if (g_reuseGt != 0 && g_reuseGt != 1)
    {
        printf("reuse-gt must be 0 or 1\n");
        return false;
    }
    if (g_trainSlice != 0 && g_devices.size() <= 1)
    {
        printf("train-mode=slice needs more than one --devices; falling back to replica at train time\n");
    }
    if (g_searchReplica != 0 && g_devices.size() <= 1)
    {
        printf("search-mode=replica with one device is equivalent to single-card search\n");
    }
    if (g_searchReplica != 0 && g_trainSlice != 0)
    {
        printf("search-mode=replica builds on device0 only; train-mode=slice falls back to replica train\n");
    }
    return true;
}

uint32_t MixSeed(uint32_t seed, uint64_t id)
{
    uint64_t x = (id + 1ULL) * 0x9E3779B97F4A7C15ULL;
    x ^= static_cast<uint64_t>(seed);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<uint32_t>(x);
}

void NormOne(float *vec, int dim)
{
    float l2norm = 0.0f;
    for (int j = 0; j < dim; ++j)
    {
        l2norm += vec[j] * vec[j];
    }
    l2norm = std::sqrt(l2norm);
    if (std::fabs(l2norm) < FLT_EPSILON)
    {
        const float fill = 1.0f / std::sqrt(static_cast<float>(dim));
        for (int j = 0; j < dim; ++j)
        {
            vec[j] = fill;
        }
        return;
    }
    for (int j = 0; j < dim; ++j)
    {
        vec[j] /= l2norm;
    }
}

void FillRange(float *data, size_t startId, size_t n, int dim, uint32_t seed)
{
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i)
    {
        const uint64_t id = startId + static_cast<size_t>(i);
        std::mt19937 gen(MixSeed(seed, id));
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float *row = data + static_cast<size_t>(i) * static_cast<size_t>(dim);
        for (int d = 0; d < dim; ++d)
        {
            row[d] = dis(gen);
        }
        NormOne(row, dim);
    }
}

float L2sqr(const float *a, const float *b, int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

using DistId = std::pair<float, faiss::idx_t>;

void HeapPush(std::vector<DistId> &heap, int k, float dist, faiss::idx_t id)
{
    if (static_cast<int>(heap.size()) < k)
    {
        heap.emplace_back(dist, id);
        std::push_heap(heap.begin(), heap.end());
        return;
    }
    if (dist < heap.front().first)
    {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = DistId(dist, id);
        std::push_heap(heap.begin(), heap.end());
    }
}

void HeapToSorted(std::vector<DistId> &heap)
{
    if (heap.empty())
    {
        return;
    }
    std::sort_heap(heap.begin(), heap.end());
}

void UpdateGt(std::vector<std::vector<DistId>> &gt, const float *queries, int recallNq, const float *chunk,
              size_t startId, size_t n, int dim, int k)
{
#pragma omp parallel for
    for (int q = 0; q < recallNq; ++q)
    {
        const float *query = queries + static_cast<size_t>(q) * static_cast<size_t>(dim);
        for (size_t i = 0; i < n; ++i)
        {
            const float dist = L2sqr(query, chunk + i * static_cast<size_t>(dim), dim);
            HeapPush(gt[static_cast<size_t>(q)], k, dist, static_cast<faiss::idx_t>(startId + i));
        }
    }
}

class BaseBinWriter
{
   public:
    BaseBinWriter() = default;
    BaseBinWriter(const BaseBinWriter &) = delete;
    BaseBinWriter &operator=(const BaseBinWriter &) = delete;

    bool Open(const std::string &path, size_t ntotal, int dim)
    {
        Close();
        dim_ = dim;
        fp_ = std::fopen(path.c_str(), "wb");
        if (fp_ == nullptr)
        {
            printf("failed to open --src-bin for write: %s\n", path.c_str());
            return false;
        }
        const double gb =
            static_cast<double>(ntotal) * static_cast<double>(dim) * sizeof(float) / (1024.0 * 1024.0 * 1024.0);
        printf("writing --src-bin %s (%.2f GB, %zu vectors)\n", path.c_str(), gb, ntotal);
        return true;
    }

    bool Write(const float *x, size_t n)
    {
        if (fp_ == nullptr)
        {
            printf("--src-bin is not open for write\n");
            return false;
        }
        const size_t count = n * static_cast<size_t>(dim_);
        if (std::fwrite(x, sizeof(float), count, fp_) != count)
        {
            printf("fwrite --src-bin failed after %zu vectors\n", written_);
            return false;
        }
        written_ += n;
        return true;
    }

    void Close()
    {
        if (fp_ != nullptr)
        {
            std::fclose(fp_);
            fp_ = nullptr;
        }
    }

    ~BaseBinWriter() { Close(); }

   private:
    FILE *fp_ = nullptr;
    int dim_ = 0;
    size_t written_ = 0;
};

bool FileExists(const std::string &path) { return access(path.c_str(), F_OK) == 0; }

bool ValidateSrcBin(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        printf("failed to stat --src-bin %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    const uint64_t expected = static_cast<uint64_t>(g_ntotal) * static_cast<uint64_t>(g_dim) * sizeof(float);
    const uint64_t actual = static_cast<uint64_t>(info.st_size);
    if (actual != expected)
    {
        printf("src-bin byte size mismatch: actual=%llu expected=%llu (ntotal=%zu dim=%d)\n",
               static_cast<unsigned long long>(actual), static_cast<unsigned long long>(expected), g_ntotal, g_dim);
        return false;
    }
    printf("use existing src-bin %s bytes=%llu vectors=%zu dim=%d\n", path.c_str(),
           static_cast<unsigned long long>(actual), g_ntotal, g_dim);
    return true;
}

bool LoadTrainFromSrcBin(const std::string &path, size_t trainNum, std::vector<float> &train)
{
    train.resize(trainNum * static_cast<size_t>(g_dim));
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("failed to open --src-bin for train: %s\n", path.c_str());
        return false;
    }
    const size_t got = std::fread(train.data(), sizeof(float), train.size(), fp);
    std::fclose(fp);
    if (got != train.size())
    {
        printf("failed to read train vectors from --src-bin: got=%zu expected=%zu floats\n", got, train.size());
        return false;
    }
    printf("loaded %zu train vectors from src-bin\n", trainNum);
    return true;
}

void SaveIndex(const faiss::ascend::AscendIndexIVFRaBitQ &index, const std::string &path)
{
    faiss::IndexFlatL2 quantizer(g_dim);
    faiss::IndexIVFRaBitQ cpu(&quantizer, g_dim, static_cast<size_t>(g_nlist), faiss::METRIC_L2);
    index.copyTo(&cpu);
    printf("write index %s ntotal=%ld quant_bits=%d code_size=%zu\n", path.c_str(), static_cast<long>(cpu.ntotal),
           g_quantBits, cpu.code_size);
    faiss::write_index(&cpu, path.c_str());
}

void LoadIndex(faiss::ascend::AscendIndexIVFRaBitQ &index, const std::string &path)
{
    const double loadT0 = GetMillisecs();
    printf("read index %s\n", path.c_str());
    std::unique_ptr<faiss::Index> raw(faiss::read_index(path.c_str()));
    const double readMs = GetMillisecs() - loadT0;
    auto *cpu = dynamic_cast<faiss::IndexIVFRaBitQ *>(raw.get());
    if (cpu == nullptr)
    {
        throw std::runtime_error("index-path is not IndexIVFRaBitQ");
    }
    printf("loaded CPU carrier ntotal=%ld code_size=%zu, requested quant_bits=%d\n", static_cast<long>(cpu->ntotal),
           cpu->code_size, g_quantBits);
    const double copyT0 = GetMillisecs();
    index.copyFrom(cpu);
    const double copyFromMs = GetMillisecs() - copyT0;
    index.setNumProbes(g_nprobe);
    printf("loaded ntotal=%ld nlist=%ld\n", static_cast<long>(index.ntotal), static_cast<long>(index.getNumLists()));
    printf("INDEX_LOAD quant_bits=%d devices=%s read_ms=%.3f copyfrom_ms=%.3f total_ms=%.3f path=%s\n", g_quantBits,
           DevicesJson().c_str(), readMs, copyFromMs, GetMillisecs() - loadT0, path.c_str());
    fflush(stdout);
}

bool SaveGt(const std::string &path, const std::vector<std::vector<DistId>> &gt)
{
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr)
    {
        printf("failed to open --gt-path for write: %s\n", path.c_str());
        return false;
    }
    const int32_t magic = 0x47545251;
    const int32_t nq = static_cast<int32_t>(gt.size());
    const int32_t k = g_recallK;
    if (std::fwrite(&magic, sizeof(magic), 1, fp) != 1 || std::fwrite(&nq, sizeof(nq), 1, fp) != 1 ||
        std::fwrite(&k, sizeof(k), 1, fp) != 1)
    {
        std::fclose(fp);
        return false;
    }
    for (auto heap : gt)
    {
        HeapToSorted(heap);
        const int32_t n = static_cast<int32_t>(heap.size());
        if (std::fwrite(&n, sizeof(n), 1, fp) != 1)
        {
            std::fclose(fp);
            return false;
        }
        for (const DistId &item : heap)
        {
            if (std::fwrite(&item.first, sizeof(float), 1, fp) != 1 ||
                std::fwrite(&item.second, sizeof(faiss::idx_t), 1, fp) != 1)
            {
                std::fclose(fp);
                return false;
            }
        }
    }
    std::fclose(fp);
    printf("wrote gt %s queries=%d\n", path.c_str(), nq);
    return true;
}

bool LoadGt(const std::string &path, std::vector<std::vector<DistId>> &gt)
{
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("failed to open --gt-path for read: %s\n", path.c_str());
        return false;
    }
    int32_t magic = 0;
    int32_t nq = 0;
    int32_t k = 0;
    if (std::fread(&magic, sizeof(magic), 1, fp) != 1 || magic != 0x47545251 ||
        std::fread(&nq, sizeof(nq), 1, fp) != 1 || std::fread(&k, sizeof(k), 1, fp) != 1)
    {
        std::fclose(fp);
        return false;
    }
    if (nq != g_recallNq || k != g_recallK)
    {
        printf("gt shape mismatch: file nq=%d k=%d, expected nq=%d k=%d\n", nq, k, g_recallNq, g_recallK);
        std::fclose(fp);
        return false;
    }
    gt.assign(static_cast<size_t>(nq), {});
    for (int32_t q = 0; q < nq; ++q)
    {
        int32_t n = 0;
        if (std::fread(&n, sizeof(n), 1, fp) != 1 || n < 0 || n > k)
        {
            printf("gt file corrupt at query %d: n=%d (k=%d)\n", q, n, k);
            std::fclose(fp);
            return false;
        }
        gt[static_cast<size_t>(q)].resize(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i)
        {
            DistId item;
            if (std::fread(&item.first, sizeof(float), 1, fp) != 1 ||
                std::fread(&item.second, sizeof(faiss::idx_t), 1, fp) != 1)
            {
                std::fclose(fp);
                return false;
            }
            gt[static_cast<size_t>(q)][static_cast<size_t>(i)] = item;
        }
    }
    std::fclose(fp);
    printf("loaded gt %s queries=%d\n", path.c_str(), nq);
    return true;
}

bool LoadSrcBin(const std::string &path, std::vector<float> &src)
{
    const size_t need = g_ntotal * static_cast<size_t>(g_dim);
    src.resize(need);
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("failed to open --src-bin for read: %s\n", path.c_str());
        return false;
    }
    const size_t got = std::fread(src.data(), sizeof(float), need, fp);
    std::fclose(fp);
    if (got != need)
    {
        printf("src-bin size mismatch: got %zu floats, expect %zu\n", got, need);
        return false;
    }
    printf("loaded src-bin %s vectors=%zu\n", path.c_str(), g_ntotal);
    return true;
}

struct RecallReport
{
    float r1At1 = 0.0f;
    float r1At10 = 0.0f;
    float r1At100 = 0.0f;
    float r1At300 = 0.0f;
    float rAtK = 0.0f;
    float candCov = 0.0f;
};

RecallReport ComputeRecall(const std::vector<std::vector<DistId>> &gt, const faiss::idx_t *ann, int nq, int annK,
                           int finalK, const faiss::idx_t *cand, int candK)
{
    RecallReport out;
    if (nq <= 0)
    {
        return out;
    }
    int hit1 = 0;
    int hit10 = 0;
    int hit100 = 0;
    int hit300 = 0;
    double rAt = 0.0;
    double cov = 0.0;
    for (int q = 0; q < nq; ++q)
    {
        const faiss::idx_t nn = gt[static_cast<size_t>(q)].empty() ? static_cast<faiss::idx_t>(-1)
                                                                   : gt[static_cast<size_t>(q)].front().second;
        std::set<faiss::idx_t> annSet;
        int foundAt = -1;
        for (int j = 0; j < finalK && j < annK; ++j)
        {
            const faiss::idx_t id = ann[static_cast<size_t>(q) * static_cast<size_t>(annK) + static_cast<size_t>(j)];
            if (id >= 0)
            {
                annSet.insert(id);
            }
            if (foundAt < 0 && id == nn)
            {
                foundAt = j;
            }
        }
        if (foundAt == 0)
        {
            ++hit1;
        }
        if (foundAt >= 0 && foundAt < 10)
        {
            ++hit10;
        }
        if (foundAt >= 0 && foundAt < 100)
        {
            ++hit100;
        }
        if (foundAt >= 0 && foundAt < 300)
        {
            ++hit300;
        }
        std::set<faiss::idx_t> gtSet;
        const int gk = std::min(finalK, static_cast<int>(gt[static_cast<size_t>(q)].size()));
        for (int j = 0; j < gk; ++j)
        {
            gtSet.insert(gt[static_cast<size_t>(q)][static_cast<size_t>(j)].second);
        }
        int inter = 0;
        for (faiss::idx_t id : annSet)
        {
            if (gtSet.count(id) != 0)
            {
                ++inter;
            }
        }
        rAt += (gk > 0) ? (static_cast<double>(inter) / static_cast<double>(gk)) : 0.0;
        if (cand != nullptr && candK > 0)
        {
            std::set<faiss::idx_t> candSet;
            for (int j = 0; j < candK; ++j)
            {
                const faiss::idx_t id =
                    cand[static_cast<size_t>(q) * static_cast<size_t>(candK) + static_cast<size_t>(j)];
                if (id >= 0)
                {
                    candSet.insert(id);
                }
            }
            int cover = 0;
            for (faiss::idx_t id : gtSet)
            {
                if (candSet.count(id) != 0)
                {
                    ++cover;
                }
            }
            cov += (gk > 0) ? (static_cast<double>(cover) / static_cast<double>(gk)) : 0.0;
        }
    }
    const float inv = 100.0f / static_cast<float>(nq);
    out.r1At1 = static_cast<float>(hit1) * inv;
    out.r1At10 = static_cast<float>(hit10) * inv;
    out.r1At100 = static_cast<float>(hit100) * inv;
    out.r1At300 = static_cast<float>(hit300) * inv;
    out.rAtK = static_cast<float>(rAt / nq * 100.0);
    out.candCov = (cand != nullptr) ? static_cast<float>(cov / nq * 100.0) : 0.0f;
    return out;
}
