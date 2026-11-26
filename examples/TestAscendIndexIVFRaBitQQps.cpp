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
 * 需要 aicpu / ivfflat / ivfrabitq 算子，编译需要 Faiss 1.14.1。
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
 *       --npu-k 是粗排阶段 NPU 输出的 k，--candidate-topk 是精排最终输出的 k。
 *       cpu 档会用 --npu-k 列表 × --candidate-topk 列表跑二维矩阵，每个组合单独计算召回与 QPS。
 *       例如 --npu-k 2000,5000 --candidate-topk 100,300 会测试 4 种粗排/精排组合。
 *       none 档在 --npu-k 给出多值时按其列表扫描 NPU 输出 k，否则沿用 --candidate-topk。
 *       原文存在时用其建库，不存在时建库阶段生成并保存；加载索引时精排原文必须存在。
 *       npu 精排不支持进程内 nprobe/候选 TopK/npu-k 矩阵；SDK 精排的候选扩展由 --refine-alpha
 *       控制（默认按最大 npu-k / 最大 candidate-topk 计算）。
 * 参数：--nprobe、--candidate-topk 和 --npu-k 可用逗号列表；2-bit dim<=512，4-bit dim<=256，支持 L2 和 IP。
 * 复用：加载已有索引时必须提供匹配的 --gt-path；--rebuild 1 强制重建索引。
 *       同一底库可用 --reuse-gt 1 复用 GT，但 ntotal、dim、seed、原文和召回参数须保持一致。
 * 大库：5 亿底库可设 --ntotal 500000000 --nlist 32768；同时调整模型、资源及分块大小。
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendIndexIVFRaBitQ.h>
#include <faiss/index_io.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
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

// 建库 / 训练 / GT 数据来源：随机生成、float32 src-bin、开源 uint8 u8bin（如 SIFT1B base.1B.u8bin）
enum class DataSrc
{
    Generate = 0,
    SrcBin = 1,
    BaseBin = 2
};

std::vector<int> g_devices = {0};
int g_dim = kDefaultDim;
size_t g_ntotal = kDefaultNtotal;
int g_nlist = kDefaultNlist;
int g_nprobe = kDefaultNprobe;
int g_k = kDefaultK;
std::vector<int> g_nprobes = {kDefaultNprobe};
std::vector<int> g_candidateTopks = {kDefaultK};
// 粗排阶段 NPU 输出的 k 列表；cpu 档与 g_candidateTopks 组成二维矩阵。
int g_npuK = kDefaultNpuK;
std::vector<int> g_npuKs = {kDefaultNpuK};
bool g_npuKExplicit = false;
int g_quantBits = 1;
faiss::MetricType g_metric = faiss::METRIC_L2;
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
int g_attachSrc = 0;  // 1=共享同一份精排原文（attachRefineSources 零拷贝），0=深拷贝
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
// 开源数据集（SIFT1B 等 fbin 格式：int32 n、int32 d、uint8 向量行优先）路径
std::string g_baseBin;
std::string g_queryBin;
// 1=用 NPU Flat 按所选度量暴搜生成 GT（可脱离索引重建单独生成）；0=建库时 host 暴搜
int g_gtNpu = 0;
uint64_t g_gtSignature = 0;

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
        "  [--candidate-topk 300,512] [--target-topk 100] [--npu-k 2000,5000] [--quant-bits 1]\n"
        "  [--nq 1,8,16,32] [--recall-nq 100] [--warmup 5] [--loops 20]\n"
        "  [--resource-mb 4096] [--add-chunk 1000000] [--index-path ./ivfrabitq.index]\n"
        "  [--src-bin path] [--gt-path path] [--rebuild 0] [--reuse-gt 0] [--csv path]\n"
        "  [--base-bin path] [--query-bin path] [--gt-npu 0]\n"
        "  [--seed 5678] [--use-kmeans-pp 1] [--niter 10] [--train-mode replica|slice]\n"
        "  [--search-mode shard|replica] [--qps-mode wait|sustain] [--duration 60] [gtest flags]\n",
        prog);
    printf("  none: NPU search k (needRefine=false)\n");
    printf("  npu:  search k with SDK refine (needs --src-bin for srcIndexes)\n");
    printf("  cpu:  NPU coarse search k=npu-k, host-side refine on --src-bin (mmap, multi-thread)\n");
    printf("  Existing --src-bin is raw headerless float32 input in row-major [ntotal, dim] layout.\n");
    printf("  --base-bin reads an open-source uint8 u8bin dataset (e.g. SIFT1B base.1B.u8bin: int32 n, int32 d,\n");
    printf("        uint8 row-major); --query-bin is the matching query u8bin (e.g. query.public.10K.u8bin).\n");
    printf("  u8bin train/base/query/GT all use float32(u8)/255, without per-vector normalization.\n");
    printf("  --gt-npu 1 builds chunked NPU Flat GT for the selected metric (FP16 precision); default is host FP32.\n");
    printf("  Comma-separated --nprobe/--candidate-topk/--npu-k run a matrix after one index build/load.\n");
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
    printf("  --npu-k: coarse-stage NPU output k. In cpu refine it is scanned together with\n");
    printf("           --candidate-topk (final k) to form a 2-D matrix; npu refine keeps it single-valued.\n");
    printf("  --target-topk: exact GT cutoff; candidate coverage equals Recall@target after exact rerank\n");
    printf("                 (--recall-k is a legacy alias).\n");
    printf(
        "  Multi-bit POC: SIMT on A5, Cube otherwise, --metric L2|IP (default L2), 2-bit dim <= 512, 4-bit dim <= "
        "256.\n");
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
        {"--index-path", &g_indexPath}, {"--src-bin", &g_srcBin},   {"--gt-path", &g_gtPath},
        {"--csv", &g_csvPath},          {"--base-bin", &g_baseBin}, {"--query-bin", &g_queryBin}};
    const std::unordered_map<std::string, std::vector<int> *> lists = {{"--devices", &g_devices},
                                                                       {"--nq", &g_nq},
                                                                       {"--nprobe", &g_nprobes},
                                                                       {"--k", &g_candidateTopks},
                                                                       {"--candidate-topk", &g_candidateTopks},
                                                                       {"--npu-k", &g_npuKs}};
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
            flag != "--attach-src" && flag != "--gt-npu" && flag != "--metric")
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
            g_npuK = g_npuKs.front();
            if (flag == "--npu-k")
            {
                g_npuKExplicit = true;
            }
        }
        else if (flag == "--refine")
        {
            if (!ParseRefineList(value, g_refine))
            {
                PrintUsage(argv[0]);
                return false;
            }
        }
        else if (flag == "--metric")
        {
            if (value != "L2" && value != "IP")
            {
                printf("--metric must be L2 or IP\n");
                return false;
            }
            g_metric = value == "IP" ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
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
                (flag == "--attach-src" && number > 1) || (flag == "--gt-npu" && number > 1))
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
            else if (flag == "--gt-npu")
            {
                g_gtNpu = static_cast<int>(number);
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
    if (!g_baseBin.empty() && g_queryBin.empty())
    {
        printf("--base-bin requires the matching --query-bin; random queries are not used\n");
        return false;
    }
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
        std::any_of(g_npuKs.begin(), g_npuKs.end(), [](int value) { return value <= 0; }))
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
    if (g_ntotal < static_cast<size_t>(g_nlist) || g_recallK < 0 || g_ntotal < static_cast<size_t>(g_recallK))
    {
        printf("ntotal must be >= nlist and target-topk\n");
        return false;
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
    // multi-bit 的设备和维度限制由 Impl 层校验。
    if (HasMode(RefineMode::Npu) && (g_nprobes.size() > 1 || g_candidateTopks.size() > 1 || g_npuKs.size() > 1))
    {
        printf("npu refine does not support an in-process nprobe/TopK/npu-k matrix; use --refine none or cpu\n");
        return false;
    }
    const int minNpuK = *std::min_element(g_npuKs.begin(), g_npuKs.end());
    const int maxNpuK = *std::max_element(g_npuKs.begin(), g_npuKs.end());
    if (minNpuK < maxCandidateTopk)
    {
        printf("every npu-k (minimum %d) must be >= maximum candidate-topk (%d)\n", minNpuK, maxCandidateTopk);
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
    if (HasMode(RefineMode::Cpu) && maxNpuK > 5000)
    {
        printf("cpu refine requires every --npu-k <= 5000 (NPU coarse search MAX_K); got max %d\n", maxNpuK);
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
        // 矩阵模式下 candidate-topk / npu-k 均有多个取值，分别用最大值计算默认 alpha，避免候选扩展被低估
        g_refineAlpha = static_cast<float>(maxNpuK) / static_cast<float>(maxCandidateTopk);
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

float ExactScore(const float *a, const float *b, int dim)
{
    if (g_metric == faiss::METRIC_INNER_PRODUCT)
    {
        return std::inner_product(a, a + dim, b, 0.0f);
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

using DistId = std::pair<float, faiss::idx_t>;

bool BetterResult(const DistId &a, const DistId &b)
{
    if (a.first == b.first) return a.second < b.second;
    return g_metric == faiss::METRIC_INNER_PRODUCT ? a.first > b.first : a.first < b.first;
}

void HeapPush(std::vector<DistId> &heap, int k, float dist, faiss::idx_t id)
{
    if (static_cast<int>(heap.size()) < k)
    {
        heap.emplace_back(dist, id);
        std::push_heap(heap.begin(), heap.end(), BetterResult);
        return;
    }
    if (BetterResult({dist, id}, heap.front()))
    {
        std::pop_heap(heap.begin(), heap.end(), BetterResult);
        heap.back() = DistId(dist, id);
        std::push_heap(heap.begin(), heap.end(), BetterResult);
    }
}

void HeapToSorted(std::vector<DistId> &heap)
{
    if (heap.empty())
    {
        return;
    }
    std::sort(heap.begin(), heap.end(), BetterResult);
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
            const float dist = ExactScore(query, chunk + i * static_cast<size_t>(dim), dim);
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

bool SaveGt(const std::string &path, const std::vector<std::vector<DistId>> &gt);
size_t TrainNum();
bool BuildGtOnHost(const std::string &path, std::vector<std::vector<DistId>> &gt, DataSrc dataSrc,
                   const float *queries);

// u8bin（SIFT1B 等）uint8 原始值域 0~255，直接转 float 后 FP16 距离计算会溢出（255^2*128 > 65504），
// 统一除以 255 压到 [0,1]；均匀缩放不改变 L2 排序，召回结果与未缩放一致
inline float U8ToF32(uint8_t v) { return static_cast<float>(v) * (1.0f / 255.0f); }

// 当前建库 / 训练 / GT 使用的实际数据来源：base-bin > src-bin > 随机生成
DataSrc ResolveDataSrc()
{
    if (!g_baseBin.empty())
    {
        return DataSrc::BaseBin;
    }
    if (!g_srcBin.empty() && FileExists(g_srcBin))
    {
        return DataSrc::SrcBin;
    }
    return DataSrc::Generate;
}

// 校验 fbin 头（int32 n、int32 d），n 取实际文件大小，dim 必须与 --dim 一致
bool ValidateBaseBin(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        printf("failed to stat --base-bin %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("failed to open --base-bin %s\n", path.c_str());
        return false;
    }
    int32_t header[2] = {0, 0};
    const bool headerOk = std::fread(header, sizeof(int32_t), 2, fp) == 2;
    std::fclose(fp);
    if (!headerOk)
    {
        printf("--base-bin %s is too small to hold an fbin header\n", path.c_str());
        return false;
    }
    const int32_t n = header[0];
    const int32_t d = header[1];
    if (n <= 0 || d <= 0)
    {
        printf("--base-bin %s has invalid fbin header n=%d d=%d\n", path.c_str(), n, d);
        return false;
    }
    if (d != g_dim)
    {
        printf("--base-bin dim=%d does not match --dim %d\n", d, g_dim);
        return false;
    }
    if (static_cast<size_t>(n) < g_ntotal)
    {
        printf("--base-bin has only %d vectors, less than --ntotal %zu\n", n, g_ntotal);
        return false;
    }
    const uint64_t expected =
        sizeof(int32_t) * 2ULL + static_cast<uint64_t>(n) * static_cast<uint64_t>(d) * sizeof(uint8_t);
    const uint64_t actual = static_cast<uint64_t>(info.st_size);
    if (actual != expected)
    {
        printf("--base-bin %s size mismatch: bytes=%llu expected=%llu (n=%d d=%d)\n", path.c_str(),
               static_cast<unsigned long long>(actual), static_cast<unsigned long long>(expected), n, d);
        return false;
    }
    printf("use --base-bin %s header_n=%d dim=%d bytes=%llu (ntotal=%zu)\n", path.c_str(), n, d,
           static_cast<unsigned long long>(actual), g_ntotal);
    return true;
}

// 读取 fbin 查询文件（int32 n、int32 d、uint8 行优先），uint8 转 float 后返回
// nqIn/out：调用方传入请求条数（如 recallNq）；nqReadOut 返回文件实际条数（<= 请求值）。
bool LoadQueriesFromFbin(const std::string &path, size_t requestNq, std::vector<float> &queries, size_t &nqReadOut)
{
    nqReadOut = 0;
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("failed to open query u8bin %s\n", path.c_str());
        return false;
    }
    int32_t header[2] = {0, 0};
    if (std::fread(header, sizeof(int32_t), 2, fp) != 2)
    {
        printf("failed to read fbin header from %s\n", path.c_str());
        std::fclose(fp);
        return false;
    }
    const int32_t n = header[0];
    const int32_t d = header[1];
    if (n <= 0 || d != g_dim)
    {
        printf("query u8bin %s header mismatch: n=%d d=%d expected dim=%d\n", path.c_str(), n, d, g_dim);
        std::fclose(fp);
        return false;
    }
    struct stat info;
    const uint64_t expected = 8ULL + static_cast<uint64_t>(n) * static_cast<uint64_t>(d);
    if (fstat(fileno(fp), &info) != 0 || static_cast<uint64_t>(info.st_size) != expected)
    {
        printf("query u8bin size mismatch: %s (expected %llu bytes)\n", path.c_str(),
               static_cast<unsigned long long>(expected));
        std::fclose(fp);
        return false;
    }
    const size_t nq = std::min(static_cast<size_t>(n), requestNq);
    if (nq == 0)
    {
        std::fclose(fp);
        return true;
    }
    queries.assign(nq * static_cast<size_t>(d), 0.0f);
    std::vector<uint8_t> row(static_cast<size_t>(d));
    for (size_t q = 0; q < nq; ++q)
    {
        if (std::fread(row.data(), sizeof(uint8_t), row.size(), fp) != row.size())
        {
            printf("short read in query u8bin %s at query %zu\n", path.c_str(), q);
            std::fclose(fp);
            return false;
        }
        float *out = queries.data() + q * static_cast<size_t>(d);
        for (int j = 0; j < d; ++j)
        {
            out[j] = U8ToF32(row[static_cast<size_t>(j)]);
        }
    }
    std::fclose(fp);
    nqReadOut = nq;
    printf("loaded %zu queries from %s (file has %d, dim=%d)\n", nq, path.c_str(), n, d);
    return true;
}

// 用 NPU Flat 按所选度量为 queries 生成 TopK GT，暴搜时将 FP32 转为 FP16。
// 底库从 resolveDataSrc 指向的来源分块读取；g_recallK 截断后按 SaveGt 格式落盘。
bool BuildGtOnNpu(const std::string &path, std::vector<std::vector<DistId>> &gt, DataSrc dataSrc, const float *queries)
{
    const size_t dim = static_cast<size_t>(g_dim);
    const size_t k = static_cast<size_t>(g_recallK);
    if (g_recallNq <= 0 || k == 0)
    {
        printf("gt-npu: recall-nq and target-topk must be > 0\n");
        return false;
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> input(nullptr, &std::fclose);
    if (dataSrc != DataSrc::Generate)
    {
        input.reset(std::fopen(dataSrc == DataSrc::BaseBin ? g_baseBin.c_str() : g_srcBin.c_str(), "rb"));
        if (!input)
        {
            printf("gt-npu: failed to open source file for read\n");
            return false;
        }
    }
    gt.assign(static_cast<size_t>(g_recallNq), {});
    for (auto &heap : gt)
    {
        heap.reserve(k);
    }
    const size_t nqRead = static_cast<size_t>(g_recallNq);
    try
    {
        faiss::ascend::AscendIndexFlat flat(g_dim, g_metric,
                                            faiss::ascend::AscendIndexFlatConfig(g_devices, 1024LL * 1024 * 512));
        const size_t chunk = std::max(static_cast<size_t>(g_addChunk), static_cast<size_t>(100000));
        std::vector<float> buf;
        std::vector<float> dist(nqRead * k);
        std::vector<faiss::idx_t> label(nqRead * k);
        size_t read = 0;
        const double t0 = GetMillisecs();
        while (read < g_ntotal)
        {
            const size_t n = std::min(chunk, g_ntotal - read);
            buf.resize(n * dim);
            if (input)
            {
                if (dataSrc == DataSrc::BaseBin)
                {
                    const size_t skip = sizeof(int32_t) * 2ULL + read * dim;
                    if (fseeko(input.get(), static_cast<off_t>(skip), SEEK_SET) != 0)
                    {
                        throw std::runtime_error("gt-npu: failed to seek base-bin");
                    }
                    std::vector<uint8_t> raw(n * dim);
                    if (std::fread(raw.data(), sizeof(uint8_t), raw.size(), input.get()) != raw.size())
                    {
                        throw std::runtime_error("gt-npu: short read in base-bin");
                    }
                    for (size_t i = 0; i < n * dim; ++i)
                    {
                        buf[i] = U8ToF32(raw[i]);
                    }
                }
                else
                {
                    if (std::fread(buf.data(), sizeof(float), buf.size(), input.get()) != buf.size())
                    {
                        throw std::runtime_error("gt-npu: short read in src-bin");
                    }
                }
            }
            else
            {
                FillRange(buf.data(), read, n, g_dim, g_seed);
            }
            flat.reset();
            flat.add(static_cast<faiss::idx_t>(n), buf.data());
            const int chunkK = static_cast<int>(std::min(n, k));
            flat.search(static_cast<faiss::idx_t>(nqRead), queries, chunkK, dist.data(), label.data());
            for (size_t q = 0; q < nqRead; ++q)
            {
                for (int j = 0; j < chunkK; ++j)
                {
                    const size_t pos = q * static_cast<size_t>(chunkK) + static_cast<size_t>(j);
                    if (label[pos] >= 0 && static_cast<size_t>(label[pos]) < n)
                    {
                        HeapPush(gt[q], g_recallK, dist[pos], static_cast<faiss::idx_t>(read) + label[pos]);
                    }
                }
            }
            read += n;
            const double ms = GetMillisecs() - t0;
            printf("gt-npu add progress %zu / %zu (%.1f vec/s)\n", read, g_ntotal,
                   ms > 0.0 ? (read * 1000.0 / ms) : 0.0);
            fflush(stdout);
        }
    }
    catch (const std::exception &e)
    {
        printf("gt-npu failed: %s; falling back to host brute-force gt\n", e.what());
        return BuildGtOnHost(path, gt, dataSrc, queries);
    }
    for (auto &heap : gt)
    {
        HeapToSorted(heap);
    }
    if (!path.empty())
    {
        return SaveGt(path, gt);
    }
    printf("gt-npu built in-memory gt queries=%zu k=%zu\n", nqRead, k);
    return true;
}

// Host 多线程暴搜生成 GT：与 gt-npu 同一数据源与读取逻辑，NPU Flat 不可用时兜底
bool BuildGtOnHost(const std::string &path, std::vector<std::vector<DistId>> &gt, DataSrc dataSrc, const float *queries)
{
    const size_t dim = static_cast<size_t>(g_dim);
    const size_t k = static_cast<size_t>(g_recallK);
    const size_t nqRead = static_cast<size_t>(g_recallNq);
    std::unique_ptr<FILE, decltype(&std::fclose)> input(nullptr, &std::fclose);
    if (dataSrc != DataSrc::Generate)
    {
        input.reset(std::fopen(dataSrc == DataSrc::BaseBin ? g_baseBin.c_str() : g_srcBin.c_str(), "rb"));
        if (!input)
        {
            printf("gt-host: failed to open source file for read\n");
            return false;
        }
    }
    gt.assign(static_cast<size_t>(g_recallNq), {});
    for (auto &heap : gt)
    {
        heap.reserve(k);
    }
    const size_t chunk = std::max(static_cast<size_t>(g_addChunk), static_cast<size_t>(100000));
    std::vector<float> buf;
    size_t read = 0;
    const double t0 = GetMillisecs();
    while (read < g_ntotal)
    {
        const size_t n = std::min(chunk, g_ntotal - read);
        buf.resize(n * dim);
        if (input)
        {
            if (dataSrc == DataSrc::BaseBin)
            {
                const size_t skip = sizeof(int32_t) * 2ULL + read * dim;
                if (fseeko(input.get(), static_cast<off_t>(skip), SEEK_SET) != 0)
                {
                    printf("gt-host: failed to seek base-bin\n");
                    return false;
                }
                std::vector<uint8_t> raw(n * dim);
                if (std::fread(raw.data(), sizeof(uint8_t), raw.size(), input.get()) != raw.size())
                {
                    printf("gt-host: short read in base-bin\n");
                    return false;
                }
                for (size_t i = 0; i < n * dim; ++i)
                {
                    buf[i] = U8ToF32(raw[i]);
                }
            }
            else
            {
                if (std::fread(buf.data(), sizeof(float), buf.size(), input.get()) != buf.size())
                {
                    printf("gt-host: short read in src-bin\n");
                    return false;
                }
            }
        }
        else
        {
            FillRange(buf.data(), read, n, g_dim, g_seed);
        }
        UpdateGt(gt, queries, static_cast<int>(nqRead), buf.data(), read, n, g_dim, static_cast<int>(k));
        read += n;
        const double ms = GetMillisecs() - t0;
        printf("gt-host progress %zu / %zu (%.1f vec/s)\n", read, g_ntotal, ms > 0.0 ? (read * 1000.0 / ms) : 0.0);
        fflush(stdout);
    }
    for (auto &heap : gt)
    {
        HeapToSorted(heap);
    }
    if (!path.empty())
    {
        return SaveGt(path, gt);
    }
    printf("gt-host built in-memory gt queries=%zu k=%zu\n", nqRead, k);
    return true;
}

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

// 从 fbin（int32 n、int32 d、uint8 行优先）读取前 trainNum 条向量做训练，uint8 转 float
bool LoadTrainFromFbin(const std::string &path, size_t trainNum, std::vector<float> &train)
{
    train.resize(trainNum * static_cast<size_t>(g_dim));
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("failed to open --base-bin for train: %s\n", path.c_str());
        return false;
    }
    int32_t header[2] = {0, 0};
    if (std::fread(header, sizeof(int32_t), 2, fp) != 2)
    {
        printf("failed to read fbin header from %s\n", path.c_str());
        std::fclose(fp);
        return false;
    }
    if (header[0] <= 0 || header[1] != g_dim || static_cast<size_t>(header[0]) < trainNum)
    {
        printf("--base-bin has only %d vectors, need %zu for train\n", header[0], trainNum);
        std::fclose(fp);
        return false;
    }
    std::vector<uint8_t> raw(train.size());
    const size_t got = std::fread(raw.data(), sizeof(uint8_t), raw.size(), fp);
    std::fclose(fp);
    if (got != raw.size())
    {
        printf("failed to read train vectors from --base-bin: got=%zu expected=%zu\n", got, raw.size());
        return false;
    }
    for (size_t i = 0; i < train.size(); ++i)
    {
        train[i] = U8ToF32(raw[i]);
    }
    printf("loaded %zu train vectors from base-bin %s\n", trainNum, path.c_str());
    return true;
}

void SaveIndex(const faiss::ascend::AscendIndexIVFRaBitQ &index, const std::string &path)
{
    faiss::IndexIVFRaBitQ cpu;
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

int32_t GroundTruthMagic() { return g_metric == faiss::METRIC_INNER_PRODUCT ? 0x47544932 : 0x47545232; }

// Cache identity includes the query contents, conversion version and source file identity.
// This avoids scanning the full 128 GB source just to validate a cache; it is not a content checksum.
uint64_t GtSignature(DataSrc source, const float *queries)
{
    uint64_t hash = 14695981039346656037ULL;
    auto append = [&](const void *data, size_t n)
    {
        const auto *bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < n; ++i)
        {
            hash = (hash ^ bytes[i]) * 1099511628211ULL;
        }
    };
    const uint64_t config[] = {2,      static_cast<uint64_t>(source), static_cast<uint64_t>(g_dim), g_ntotal,
                               g_seed, static_cast<uint64_t>(g_gtNpu)};
    append(config, sizeof(config));
    append(queries, static_cast<size_t>(g_recallNq) * g_dim * sizeof(float));
    const std::string &path = source == DataSrc::BaseBin ? g_baseBin : g_srcBin;
    if (source != DataSrc::Generate)
    {
        struct stat info;
        if (stat(path.c_str(), &info) != 0)
        {
            throw std::runtime_error("failed to stat GT source: " + path);
        }
        append(path.data(), path.size());
        append(&info.st_size, sizeof(info.st_size));
        append(&info.st_mtime, sizeof(info.st_mtime));
    }
    return hash;
}

bool SaveGt(const std::string &path, const std::vector<std::vector<DistId>> &gt)
{
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr)
    {
        printf("failed to open --gt-path for write: %s\n", path.c_str());
        return false;
    }
    const int32_t magic = GroundTruthMagic();
    const int32_t nq = static_cast<int32_t>(gt.size());
    const int32_t k = g_recallK;
    if (std::fwrite(&magic, sizeof(magic), 1, fp) != 1 || std::fwrite(&nq, sizeof(nq), 1, fp) != 1 ||
        std::fwrite(&k, sizeof(k), 1, fp) != 1 || std::fwrite(&g_gtSignature, sizeof(g_gtSignature), 1, fp) != 1)
    {
        std::fclose(fp);
        return false;
    }
    for (const auto &heap : gt)
    {
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
    uint64_t signature = 0;
    int32_t magic = 0;
    int32_t nq = 0;
    int32_t k = 0;
    if (std::fread(&magic, sizeof(magic), 1, fp) != 1 || magic != GroundTruthMagic() ||
        std::fread(&nq, sizeof(nq), 1, fp) != 1 || std::fread(&k, sizeof(k), 1, fp) != 1 ||
        std::fread(&signature, sizeof(signature), 1, fp) != 1 || signature != g_gtSignature)
    {
        printf(
            "GT cache is outdated or has a different dataset/query/conversion. "
            "Regenerate with --rebuild 1 --reuse-gt 0.\n");
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
        if (std::fread(&n, sizeof(n), 1, fp) != 1 || n != k)
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
            if (!std::isfinite(item.first) || item.second < 0 || static_cast<uint64_t>(item.second) >= g_ntotal)
            {
                printf("invalid GT distance/ID at query %d position %d\n", q, i);
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

size_t TrainNum()
{
    const size_t cap = static_cast<size_t>(g_nlist) * static_cast<size_t>(kTrainMultiplier);
    size_t n = std::min(g_ntotal, cap);
    if (n > kMaxTrainNum)
    {
        n = kMaxTrainNum;
    }
    return n;
}

long MemAvailableKb()
{
    FILE *fp = std::fopen("/proc/meminfo", "r");
    if (fp == nullptr)
    {
        return -1;
    }
    char key[64];
    long value = 0;
    char unit[32];
    long avail = -1;
    while (std::fscanf(fp, "%63s %ld %31s", key, &value, unit) == 3)
    {
        if (std::strcmp(key, "MemAvailable:") == 0)
        {
            avail = value;
            break;
        }
    }
    std::fclose(fp);
    return avail;
}

int SearchReplicaCount() { return (g_searchReplica != 0) ? static_cast<int>(g_devices.size()) : 1; }

const char *SearchModeName() { return (g_searchReplica != 0) ? "replica" : "shard"; }

const char *QpsModeName() { return (g_qpsSustain != 0) ? "sustain" : "wait"; }

bool HostRamEnoughForNpu()
{
    // attach 模式共享 src 向量；深拷贝模式还需计入加载时的 src 向量
    const int copies = (g_attachSrc != 0) ? 1 : 2;
    const double needGb = static_cast<double>(g_ntotal) * static_cast<double>(g_dim) * sizeof(float) /
                          (1024.0 * 1024.0 * 1024.0) * static_cast<double>(copies);
    const long availKb = MemAvailableKb();
    printf("npu refine srcIndexes estimate=%.2f GB host (copies=%d), MemAvailable=%.2f GB\n", needGb, copies,
           availKb < 0 ? -1.0 : static_cast<double>(availKb) / kMemGb);
    fflush(stdout);
    if (availKb < 0)
    {
        return true;
    }
    const double availGb = static_cast<double>(availKb) / kMemGb;
    return availGb >= needGb * kNpuHostOverhead;
}

std::string DevicesJson()
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < g_devices.size(); ++i)
    {
        if (i != 0)
        {
            out << ",";
        }
        out << g_devices[i];
    }
    out << "]";
    return out.str();
}

std::unique_ptr<faiss::ascend::AscendIndexIVFRaBitQ> MakeIndex(bool needRefine, const std::vector<int> &devices)
{
    const int64_t resourceSize = static_cast<int64_t>(g_resourceMb) * 1024 * 1024;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf(devices, true, needRefine, 12345, g_refineAlpha, resourceSize);
    conf.useKmeansPP = (g_useKmeansPp != 0);
    conf.niter = g_niter;
    conf.cp.niter = g_niter;
    conf.quantBits = g_quantBits;
    conf.storageMode = (g_searchReplica != 0) ? faiss::ascend::AscendIndexIVFRaBitQConfig::StorageMode::Replica
                                              : faiss::ascend::AscendIndexIVFRaBitQConfig::StorageMode::Shard;
    conf.trainMode = (g_trainSlice != 0) ? faiss::ascend::AscendIndexIVFRaBitQConfig::TrainMode::Slice
                                         : faiss::ascend::AscendIndexIVFRaBitQConfig::TrainMode::Replica;
    auto index = std::unique_ptr<faiss::ascend::AscendIndexIVFRaBitQ>(
        new faiss::ascend::AscendIndexIVFRaBitQ(g_dim, g_metric, g_nlist, conf));
    index->verbose = true;
    index->setNumProbes(g_nprobe);
    return index;
}

std::unique_ptr<faiss::ascend::AscendIndexIVFRaBitQ> MakeIndex(bool needRefine)
{
    return MakeIndex(needRefine, g_devices);
}

using IndexPtr = std::unique_ptr<faiss::ascend::AscendIndexIVFRaBitQ>;

// Storage replication and query distribution are handled inside the multi-device index.
void LoadSearchIndex(std::vector<IndexPtr> &indexes, bool needRefine, const float *src)
{
    indexes.clear();
    indexes.push_back(MakeIndex(needRefine));
    LoadIndex(*indexes.back(), g_indexPath);
    if (src == nullptr) return;
    if (g_attachSrc != 0)
    {
        indexes.back()->attachRefineSources(src, static_cast<faiss::idx_t>(g_ntotal));
    }
    else
    {
        indexes.back()->loadRefineSources(src, static_cast<faiss::idx_t>(g_ntotal));
    }
}

void AddChunk(faiss::ascend::AscendIndexIVFRaBitQ &index, const float *x, size_t startId, size_t n)
{
    std::vector<faiss::idx_t> ids(n);
    std::iota(ids.begin(), ids.end(), static_cast<faiss::idx_t>(startId));
    index.add_with_ids(static_cast<faiss::idx_t>(n), x, ids.data());
}

struct StreamAddOptions
{
    bool readSrc = false;
    bool readFbin = false;  // 从 --base-bin（uint8 u8bin）分块读取
    bool updateGt = false;
    BaseBinWriter *writer = nullptr;
};

void StreamAdd(faiss::ascend::AscendIndexIVFRaBitQ &index, const float *train, size_t trainNum,
               std::vector<std::vector<DistId>> *gt, const float *queries, const StreamAddOptions &opt)
{
    if (opt.updateGt && (gt == nullptr || queries == nullptr))
    {
        throw std::runtime_error("updateGt requires GT storage and queries");
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> input(nullptr, &std::fclose);
    if (opt.readSrc || opt.readFbin)
    {
        input.reset(std::fopen(opt.readFbin ? g_baseBin.c_str() : g_srcBin.c_str(), "rb"));
        if (!input)
        {
            throw std::runtime_error("failed to open source bin for add");
        }
        // 训练向量取自数据源前 trainNum 条，add 从其后继续
        const uint64_t skip = opt.readFbin ? sizeof(int32_t) * 2ULL + static_cast<uint64_t>(trainNum) * g_dim
                                           : static_cast<uint64_t>(trainNum) * g_dim * sizeof(float);
        if (fseeko(input.get(), static_cast<off_t>(skip), SEEK_SET) != 0)
        {
            throw std::runtime_error("failed to seek source bin after train vectors");
        }
    }
    const size_t chunk = static_cast<size_t>(g_addChunk);
    size_t added = 0;
    const double t0 = GetMillisecs();
    auto handle = [&](const float *x, size_t start, size_t n)
    {
        if (opt.updateGt)
        {
            UpdateGt(*gt, queries, g_recallNq, x, start, n, g_dim, g_recallK);
        }
        if (opt.writer != nullptr && !opt.writer->Write(x, n))
        {
            throw std::runtime_error("write --src-bin failed");
        }
        AddChunk(index, x, start, n);
        added += n;
        if (added % (chunk * 10) == 0 || added == g_ntotal)
        {
            const double ms = GetMillisecs() - t0;
            printf("%sadd progress %zu / %zu (%.1f vec/s)\n", opt.readSrc ? "src-bin " : "", added, g_ntotal,
                   ms > 0.0 ? (added * 1000.0 / ms) : 0.0);
            fflush(stdout);
        }
    };
    handle(train, 0, trainNum);
    std::vector<float> buf;
    while (added < g_ntotal)
    {
        const size_t n = std::min(chunk, g_ntotal - added);
        buf.resize(n * static_cast<size_t>(g_dim));
        if (input)
        {
            if (opt.readFbin)
            {
                std::vector<uint8_t> raw(buf.size());
                if (std::fread(raw.data(), sizeof(uint8_t), raw.size(), input.get()) != raw.size())
                {
                    throw std::runtime_error("short read while streaming base-bin");
                }
                for (size_t i = 0; i < buf.size(); ++i)
                {
                    buf[i] = U8ToF32(raw[i]);
                }
            }
            else
            {
                if (std::fread(buf.data(), sizeof(float), buf.size(), input.get()) != buf.size())
                {
                    throw std::runtime_error("short read while streaming --src-bin");
                }
            }
        }
        else
        {
            FillRange(buf.data(), added, n, g_dim, g_seed);
        }
        handle(buf.data(), added, n);
    }
}

struct BenchRow
{
    const char *refine = "none";
    int nprobe = 0;
    int nq = 0;
    int npuK = 0;
    long long batches = 0;
    double durationS = 0.0;
    double npuQps = 0.0;
    double npuMs = 0.0;
    double e2eQps = 0.0;
    double e2eMs = 0.0;
    RecallReport recall;
};

void PrintRow(const BenchRow &row)
{
    printf(
        "ROW quant_bits=%d refine=%s search_mode=%s qps_mode=%s replicas=%d nprobe=%d nq=%d ntotal=%zu npu_k=%d "
        "candidate_topk=%d target_topk=%d "
        "duration_s=%.3f batches=%lld npu_QPS=%.4f npu_ms=%.4f "
        "e2e_QPS=%.4f e2e_ms=%.4f "
        "r1@1=%.2f r1@10=%.2f r1@100=%.2f r1@300=%.2f R@%d=%.2f rerank_recall=%.2f\n",
        g_quantBits, row.refine, SearchModeName(), QpsModeName(), SearchReplicaCount(), row.nprobe, row.nq, g_ntotal,
        row.npuK, g_k, g_recallK, row.durationS, row.batches, row.npuQps, row.npuMs, row.e2eQps, row.e2eMs,
        row.recall.r1At1, row.recall.r1At10, row.recall.r1At100, row.recall.r1At300, g_recallK, row.recall.rAtK,
        row.recall.candCov);
    fflush(stdout);
}

void WriteCsvHeader(FILE *fp)
{
    std::fprintf(fp,
                 "quant_bits,refine,search_mode,qps_mode,replicas,nprobe,nq,ntotal,npu_k,k,recall_k,duration_s,batches,"
                 "npu_qps,npu_ms,e2e_"
                 "qps,e2e_ms,"
                 "r1_at1,r1_at10,r1_at100,r1_at300,r_at_k,cand_cov\n");
}

void WriteCsvRow(FILE *fp, const BenchRow &row)
{
    std::fprintf(fp,
                 "%d,%s,%s,%s,%d,%d,%d,%zu,%d,%d,%d,%.6f,%lld,%.6f,%.6f,%.6f,%.6f,"
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                 g_quantBits, row.refine, SearchModeName(), QpsModeName(), SearchReplicaCount(), row.nprobe, row.nq,
                 g_ntotal, row.npuK, g_k, g_recallK, row.durationS, row.batches, row.npuQps, row.npuMs, row.e2eQps,
                 row.e2eMs, row.recall.r1At1, row.recall.r1At10, row.recall.r1At100, row.recall.r1At300,
                 row.recall.rAtK, row.recall.candCov);
}

void SearchNpu(std::vector<IndexPtr> &indexes, int nq, int k, const float *query, float *dist, faiss::idx_t *label)
{
    if (indexes.size() != 1)
    {
        throw std::runtime_error("expected one multi-device search index");
    }
    indexes.front()->search(nq, query, k, dist, label);
}

double TimeLoops(int loops, const std::function<void()> &fn, long long *batchesOut)
{
    const double ts = GetMillisecs();
    for (int i = 0; i < loops; ++i)
    {
        fn();
    }
    if (batchesOut != nullptr)
    {
        *batchesOut = loops;
    }
    return (GetMillisecs() - ts) / loops;
}

double TimeSustain(double durationSec, int nq, const std::function<void()> &fn, long long *batchesOut)
{
    const double budgetMs = durationSec * static_cast<double>(kMilliSecond);
    const double ts = GetMillisecs();
    const double deadline = ts + budgetMs;
    long long batches = 0;
    double lastLog = ts;
    while (GetMillisecs() < deadline)
    {
        fn();
        ++batches;
        const double now = GetMillisecs();
        if (now - lastLog >= 5000.0)
        {
            const double elapsed = now - ts;
            const double qps = elapsed > 0.0 ? (kMilliSecond * batches * nq / elapsed) : 0.0;
            printf("sustain nq=%d elapsed=%.1fs batches=%lld qps=%.2f\n", nq, elapsed / kMilliSecond, batches, qps);
            fflush(stdout);
            lastLog = now;
        }
    }
    const double elapsed = GetMillisecs() - ts;
    if (batchesOut != nullptr)
    {
        *batchesOut = batches;
    }
    if (batches <= 0)
    {
        throw std::runtime_error("sustain completed 0 batches; increase --duration");
    }
    return elapsed / static_cast<double>(batches);
}

double TimeSearch(int nq, const std::function<void()> &fn, long long *batchesOut)
{
    if (g_qpsSustain != 0)
    {
        return TimeSustain(static_cast<double>(g_durationSec), nq, fn, batchesOut);
    }
    return TimeLoops(g_loops, fn, batchesOut);
}

RecallReport RecallSearch(std::vector<IndexPtr> &indexes, const float *query,
                          const std::vector<std::vector<DistId>> &gt, bool includeCandidates)
{
    const size_t buf = static_cast<size_t>(g_recallNq) * static_cast<size_t>(g_k);
    std::vector<float> dist(buf, 0.0f);
    std::vector<faiss::idx_t> label(buf, -1);
    SearchNpu(indexes, g_recallNq, g_k, query, dist.data(), label.data());
    return ComputeRecall(gt, label.data(), g_recallNq, g_k, g_recallK, includeCandidates ? label.data() : nullptr,
                         includeCandidates ? g_k : 0);
}

BenchRow BenchSearch(std::vector<IndexPtr> &indexes, int nq, const float *query, const RecallReport &recall,
                     const char *refine)
{
    const size_t buf = static_cast<size_t>(nq) * static_cast<size_t>(g_k);
    std::vector<float> dist(buf, 0.0f);
    std::vector<faiss::idx_t> label(buf, -1);
    for (int i = 0; i < g_warmup; ++i)
    {
        SearchNpu(indexes, nq, g_k, query, dist.data(), label.data());
    }
    long long batches = 0;
    const double avgMs =
        TimeSearch(nq, [&]() { SearchNpu(indexes, nq, g_k, query, dist.data(), label.data()); }, &batches);
    BenchRow row;
    row.refine = refine;
    row.nprobe = g_nprobe;
    row.nq = nq;
    row.npuK = g_k;
    row.batches = batches;
    row.durationS = avgMs * static_cast<double>(batches) / kMilliSecond;
    row.npuMs = avgMs;
    row.e2eMs = avgMs;
    row.npuQps = kMilliSecond * nq / avgMs;
    row.e2eQps = row.npuQps;
    row.recall = recall;
    return row;
}

// ---------- cpu 档：NPU 粗排 k=npu-k 回传 Host，Host 用 src-bin 原文精排 ----------

class MmapSrc
{
   public:
    MmapSrc(const std::string &path, size_t ntotal, int dim)
    {
        const uint64_t needBytes = static_cast<uint64_t>(ntotal) * static_cast<uint64_t>(dim) * sizeof(float);
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            throw std::runtime_error("cpu refine: failed to open --src-bin: " + path);
        }
        struct stat info;
        if (fstat(fd_, &info) != 0 || static_cast<uint64_t>(info.st_size) < needBytes)
        {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("cpu refine: --src-bin too small: " + path);
        }
        data_ = static_cast<const float *>(mmap(nullptr, needBytes, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED)
        {
            close(fd_);
            fd_ = -1;
            data_ = nullptr;
            throw std::runtime_error("cpu refine: mmap --src-bin failed: " + path);
        }
        bytes_ = needBytes;
    }
    ~MmapSrc()
    {
        if (data_ != nullptr)
        {
            munmap(const_cast<float *>(data_), bytes_);
        }
        if (fd_ >= 0)
        {
            close(fd_);
        }
    }
    MmapSrc(const MmapSrc &) = delete;
    MmapSrc &operator=(const MmapSrc &) = delete;
    const float *Get() const { return data_; }

   private:
    int fd_ = -1;
    const float *data_ = nullptr;
    uint64_t bytes_ = 0;
};

// Re-score candidates against the original vectors using the selected metric.
void RefineOneQuery(const float *src, int dim, const float *query, const faiss::idx_t *candLabel, int candK, int finalK,
                    float *outDist, faiss::idx_t *outLabel)
{
    std::vector<DistId> heap;
    heap.reserve(static_cast<size_t>(finalK));
    for (int j = 0; j < candK; ++j)
    {
        const faiss::idx_t id = candLabel[j];
        if (id < 0 || static_cast<uint64_t>(id) >= g_ntotal)
        {
            continue;
        }
        const float *vec = src + static_cast<uint64_t>(id) * static_cast<uint64_t>(dim);
        HeapPush(heap, finalK, ExactScore(query, vec, dim), id);
    }
    HeapToSorted(heap);
    const int got = static_cast<int>(heap.size());
    for (int j = 0; j < finalK; ++j)
    {
        if (j < got)
        {
            outDist[j] = heap[static_cast<size_t>(j)].first;
            outLabel[j] = heap[static_cast<size_t>(j)].second;
        }
        else
        {
            outDist[j] = 0.0f;
            outLabel[j] = -1;
        }
    }
}

// 多线程对 nq 条查询做 Host 精排（供 recall 与 bench 共用）
void RefineHost(const float *src, const float *query, const faiss::idx_t *candLabel, int candK, int nq, int finalK,
                float *outDist, faiss::idx_t *outLabel)
{
#pragma omp parallel for num_threads(std::min(nq, omp_get_max_threads()))
    for (int q = 0; q < nq; ++q)
    {
        RefineOneQuery(src, g_dim, query + static_cast<uint64_t>(q) * static_cast<uint64_t>(g_dim),
                       candLabel + static_cast<uint64_t>(q) * static_cast<uint64_t>(candK), candK, finalK,
                       outDist + static_cast<uint64_t>(q) * static_cast<uint64_t>(finalK),
                       outLabel + static_cast<uint64_t>(q) * static_cast<uint64_t>(finalK));
    }
}

RecallReport RecallCpuRefine(std::vector<IndexPtr> &indexes, const float *query, const MmapSrc &src,
                             const std::vector<std::vector<DistId>> &gt)
{
    const size_t candBuf = static_cast<size_t>(g_recallNq) * static_cast<size_t>(g_npuK);
    std::vector<float> candDist(candBuf, 0.0f);
    std::vector<faiss::idx_t> candLabel(candBuf, -1);
    SearchNpu(indexes, g_recallNq, g_npuK, query, candDist.data(), candLabel.data());
    const size_t buf = static_cast<size_t>(g_recallNq) * static_cast<size_t>(g_k);
    std::vector<float> dist(buf, 0.0f);
    std::vector<faiss::idx_t> label(buf, -1);
    RefineHost(src.Get(), query, candLabel.data(), g_npuK, g_recallNq, g_k, dist.data(), label.data());
    // cand_cov：粗排候选对 GT 的覆盖率（npu 档拿不到候选，cpu 档独有）
    return ComputeRecall(gt, label.data(), g_recallNq, g_k, g_recallK, candLabel.data(), g_npuK);
}

BenchRow BenchCpuRefine(std::vector<IndexPtr> &indexes, int nq, const float *query, const MmapSrc &src,
                        const RecallReport &recall)
{
    const size_t candBuf = static_cast<size_t>(nq) * static_cast<size_t>(g_npuK);
    std::vector<float> candDist(candBuf, 0.0f);
    std::vector<faiss::idx_t> candLabel(candBuf, -1);
    const size_t buf = static_cast<size_t>(nq) * static_cast<size_t>(g_k);
    std::vector<float> dist(buf, 0.0f);
    std::vector<faiss::idx_t> label(buf, -1);
    double npuTotalMs = 0.0;
    auto benchOnce = [&]()
    {
        const double npuStart = GetMillisecs();
        SearchNpu(indexes, nq, g_npuK, query, candDist.data(), candLabel.data());
        npuTotalMs += GetMillisecs() - npuStart;
        RefineHost(src.Get(), query, candLabel.data(), g_npuK, nq, g_k, dist.data(), label.data());
    };
    for (int i = 0; i < g_warmup; ++i)
    {
        benchOnce();
    }
    npuTotalMs = 0.0;
    long long batches = 0;
    const double avgMs = TimeSearch(nq, benchOnce, &batches);
    BenchRow row;
    row.refine = "cpu";
    row.nprobe = g_nprobe;
    row.nq = nq;
    row.npuK = g_npuK;
    row.batches = batches;
    row.durationS = avgMs * static_cast<double>(batches) / kMilliSecond;
    // e2e 含粗排+回传+Host 精排
    row.npuMs = npuTotalMs / static_cast<double>(batches);
    row.e2eMs = avgMs;
    row.npuQps = kMilliSecond * nq / row.npuMs;
    row.e2eQps = kMilliSecond * nq / avgMs;
    row.recall = recall;
    return row;
}

TEST(TestAscendIndexIVFRaBitQ, LargeScaleQpsRecall)
{
    try
    {
        const size_t trainNum = TrainNum();
        const int maxNq = std::max(*std::max_element(g_nq.begin(), g_nq.end()), g_recallNq);
        const bool loadExisting = (g_rebuild == 0) && FileExists(g_indexPath);
        const DataSrc dataSrc = ResolveDataSrc();
        const bool useBaseBin = dataSrc == DataSrc::BaseBin;
        if (useBaseBin)
        {
            ASSERT_TRUE(ValidateBaseBin(g_baseBin));
        }
        else if (dataSrc == DataSrc::SrcBin)
        {
            ASSERT_TRUE(ValidateSrcBin(g_srcBin));
        }
        printf(
            "IVFRaBitQ QPS recall: metric=%s devices=%s dim=%d ntotal=%zu nlist=%d nprobe=%d quant_bits=%d "
            "candidate_topk=%d "
            "target_topk=%d npu_k=%d "
            "refine-alpha=%.6f trainNum=%zu recall_nq=%d resource_mb=%d niter=%d train-mode=%s "
            "search-mode=%s qps-mode=%s replicas=%d duration=%d loops=%d index=%s load=%d reuse_gt=%d "
            "data_src=%s gt_npu=%d\n",
            g_metric == faiss::METRIC_INNER_PRODUCT ? "IP" : "L2", DevicesJson().c_str(), g_dim, g_ntotal, g_nlist,
            g_nprobe, g_quantBits, g_k, g_recallK, g_npuK, g_refineAlpha, trainNum, g_recallNq, g_resourceMb, g_niter,
            g_trainSlice != 0 ? "slice" : "replica", SearchModeName(), QpsModeName(), SearchReplicaCount(),
            g_durationSec, g_loops, g_indexPath.c_str(), loadExisting ? 1 : 0, g_reuseGt,
            useBaseBin                   ? "base-bin"
            : dataSrc == DataSrc::SrcBin ? "src-bin"
                                         : "generate",
            g_gtNpu);
        fflush(stdout);

        // 查询向量：提供 --query-bin（SIFT1B query.public.10K.u8bin）时读取真实查询，否则随机生成
        std::vector<float> queries;
        if (!g_queryBin.empty())
        {
            size_t nqRead = 0;
            ASSERT_TRUE(LoadQueriesFromFbin(g_queryBin, static_cast<size_t>(maxNq), queries, nqRead));
            if (nqRead < static_cast<size_t>(maxNq))
            {
                FAIL() << "query-bin only has " << nqRead << " queries, need " << maxNq;
            }
        }
        else
        {
            queries.resize(static_cast<size_t>(maxNq) * static_cast<size_t>(g_dim));
            FillRange(queries.data(), 0, static_cast<size_t>(maxNq), g_dim, kQuerySeed);
        }

        g_gtSignature = GtSignature(dataSrc, queries.data());
        std::vector<std::vector<DistId>> gt(static_cast<size_t>(g_recallNq));
        bool gtReady = false;
        if (!g_gtPath.empty() && FileExists(g_gtPath) && (g_rebuild == 0 || g_reuseGt != 0))
        {
            ASSERT_TRUE(LoadGt(g_gtPath, gt));
            gtReady = true;
        }
        // 独立生成 GT：用 NPU Flat 暴搜，可先于索引建库单独执行（--gt-npu 1）
        if (!gtReady && (g_gtNpu != 0 || loadExisting))
        {
            ASSERT_TRUE(g_gtNpu != 0 ? BuildGtOnNpu(g_gtPath, gt, dataSrc, queries.data())
                                     : BuildGtOnHost(g_gtPath, gt, dataSrc, queries.data()));
            gtReady = true;
        }

        std::unique_ptr<faiss::ascend::AscendIndexIVFRaBitQ> stored;
        if (loadExisting)
        {
            stored = MakeIndex(false);
            LoadIndex(*stored, g_indexPath);
        }
        else
        {
            const bool useSrcInput = dataSrc != DataSrc::Generate;
            std::vector<float> train(trainNum * static_cast<size_t>(g_dim));
            if (useBaseBin)
            {
                ASSERT_TRUE(LoadTrainFromFbin(g_baseBin, trainNum, train));
            }
            else if (useSrcInput)
            {
                ASSERT_TRUE(LoadTrainFromSrcBin(g_srcBin, trainNum, train));
            }
            else
            {
                printf("generate train\n");
                FillRange(train.data(), 0, trainNum, g_dim, g_seed);
            }
            stored = MakeIndex(false);
            printf("start train\n");
            stored->train(static_cast<faiss::idx_t>(trainNum), train.data());
            BaseBinWriter writer;
            StreamAddOptions opt;
            opt.readSrc = dataSrc == DataSrc::SrcBin;
            opt.readFbin = useBaseBin;
            opt.updateGt = !gtReady;
            // 精排（npu/cpu）与加载复用都依赖 float32 原文 --src-bin：
            // base-bin 建库时若 src-bin 不存在，顺带把 uint8→float 转换后的数据落盘一份；
            // src-bin 输入本身已存在，无需重写。
            const bool writeSrc = !g_srcBin.empty() && !opt.readSrc && !FileExists(g_srcBin);
            if (!g_srcBin.empty() && !writeSrc && useBaseBin && FileExists(g_srcBin))
            {
                printf(
                    "warning: --src-bin exists and will be reused for refine; make sure it was converted from the "
                    "same --base-bin dataset\n");
            }
            if (writeSrc)
            {
                ASSERT_TRUE(writer.Open(g_srcBin, g_ntotal, g_dim));
                opt.writer = &writer;
            }
            printf("start add %s needRefine=false\n", useBaseBin    ? "from base-bin (sift1b)"
                                                      : useSrcInput ? "from existing src-bin"
                                                                    : "generated vectors");
            StreamAdd(*stored, train.data(), trainNum, &gt, queries.data(), opt);
            writer.Close();
            if (!gtReady)
            {
                for (auto &h : gt)
                {
                    HeapToSorted(h);
                }
                gtReady = true;
                if (!g_gtPath.empty())
                {
                    ASSERT_TRUE(SaveGt(g_gtPath, gt));
                }
            }
            SaveIndex(*stored, g_indexPath);
        }
        ASSERT_TRUE(gtReady);
        ASSERT_EQ(static_cast<size_t>(stored->ntotal), g_ntotal);

        std::vector<IndexPtr> searchers;
        searchers.push_back(std::move(stored));

        std::vector<BenchRow> rows;
        FILE *csv = nullptr;
        if (!g_csvPath.empty())
        {
            csv = std::fopen(g_csvPath.c_str(), "w");
            if (csv != nullptr)
            {
                WriteCsvHeader(csv);
            }
        }

        if (HasMode(RefineMode::None))
        {
            // 无精排时 NPU 输出 k 就是最终 k：--npu-k 给出多值时扫它，否则沿用 --candidate-topk（兼容旧命令）。
            const std::vector<int> &noneKs = (g_npuKExplicit && g_npuKs.size() > 1) ? g_npuKs : g_candidateTopks;
            for (int nprobe : g_nprobes)
            {
                g_nprobe = nprobe;
                for (auto &index : searchers)
                {
                    index->setNumProbes(g_nprobe);
                }
                for (int npuK : noneKs)
                {
                    g_k = npuK;
                    g_npuK = npuK;
                    printf(
                        "measure matrix refine=none quant_bits=%d nprobe=%d npu_k=%d candidate_topk=%d target_topk=%d "
                        "recall_nq=%d\n",
                        g_quantBits, g_nprobe, g_npuK, g_k, g_recallK, g_recallNq);
                    const RecallReport recNone = RecallSearch(searchers, queries.data(), gt, true);
                    for (int nq : g_nq)
                    {
                        const BenchRow row = BenchSearch(searchers, nq, queries.data(), recNone, "none");
                        PrintRow(row);
                        if (csv != nullptr)
                        {
                            WriteCsvRow(csv, row);
                        }
                        rows.push_back(row);
                    }
                }
            }
        }

        if (HasMode(RefineMode::Cpu))
        {
            try
            {
                // 复用 needRefine=false 索引（与 none 档同库），粗排 k=npu-k 回传 Host 后本地精排。
                // 必须在 npu 档之前跑：npu 块会 searchers.clear() 并重建 needRefine=true 索引。
                MmapSrc src(g_srcBin, g_ntotal, g_dim);
                // 粗排 NPU 输出 k × 精排最终 k 二维矩阵，每个组合单独算召回再测 QPS。
                for (int npuK : g_npuKs)
                {
                    g_npuK = npuK;
                    for (int candidateTopk : g_candidateTopks)
                    {
                        g_k = candidateTopk;
                        printf("measure recall cpu, nprobe=%d recall_nq=%d npu_k=%d candidate_topk=%d target_topk=%d\n",
                               g_nprobe, g_recallNq, g_npuK, g_k, g_recallK);
                        const RecallReport recCpu = RecallCpuRefine(searchers, queries.data(), src, gt);
                        for (int nq : g_nq)
                        {
                            BenchRow row = BenchCpuRefine(searchers, nq, queries.data(), src, recCpu);
                            PrintRow(row);
                            if (csv != nullptr)
                            {
                                WriteCsvRow(csv, row);
                            }
                            rows.push_back(std::move(row));
                        }
                    }
                }
            }
            catch (std::exception &e)
            {
                printf("skip cpu refine: %s\n", e.what());
                if (!HasMode(RefineMode::None) && !HasMode(RefineMode::Npu))
                {
                    throw;
                }
            }
        }

        if (HasMode(RefineMode::Npu))
        {
            if (!HostRamEnoughForNpu())
            {
                printf("skip npu refine: host RAM not enough for srcIndexes\n");
                fflush(stdout);
                if (!HasMode(RefineMode::None))
                {
                    FAIL() << "npu refine requested but host RAM is insufficient";
                }
            }
            else
            {
                searchers.clear();
                printf("create index needRefine=true refineAlpha=%.6f internal_k=%d\n", g_refineAlpha,
                       static_cast<int>(g_k * g_refineAlpha));
                std::vector<float> src;
                ASSERT_TRUE(LoadSrcBin(g_srcBin, src));
                LoadSearchIndex(searchers, true, src.data());
                printf("measure recall npu, recall_nq=%d\n", g_recallNq);
                const RecallReport recNpu = RecallSearch(searchers, queries.data(), gt, false);
                for (int nq : g_nq)
                {
                    const BenchRow row = BenchSearch(searchers, nq, queries.data(), recNpu, "npu");
                    PrintRow(row);
                    if (csv != nullptr)
                    {
                        WriteCsvRow(csv, row);
                    }
                    rows.push_back(row);
                }
            }
        }

        printf(
            "summary bits refine search qps reps nprobe nq batches candidate_topk target_topk npu_QPS e2e_QPS r1@1 "
            "r1@10 "
            "r1@100 r1@300 R@target_topk rerank_recall\n");
        for (const BenchRow &row : rows)
        {
            printf(
                "%4d %-5s %-7s %-7s %4d %6d %4d %7lld %11d %8d %10.2f %10.2f %7.2f %7.2f %8.2f %8.2f "
                "%12.2f %8.2f\n",
                g_quantBits, row.refine, SearchModeName(), QpsModeName(), SearchReplicaCount(), row.nprobe, row.nq,
                row.batches, row.npuK, g_recallK, row.npuQps, row.e2eQps, row.recall.r1At1, row.recall.r1At10,
                row.recall.r1At100, row.recall.r1At300, row.recall.rAtK, row.recall.candCov);
        }
        fflush(stdout);
        if (csv != nullptr)
        {
            std::fclose(csv);
        }
        ASSERT_FALSE(rows.empty());
    }
    catch (std::exception &e)
    {
        FAIL() << "Exception occurred: " << e.what();
    }
}

}  // namespace

int main(int argc, char **argv)
{
    if (!ParseCli(&argc, argv) || !ValidateCli())
    {
        return 1;
    }
    printf(
        "TestAscendIndexIVFRaBitQQps: devices=%s dim=%d ntotal=%zu nlist=%d nprobe=%d quant-bits=%d candidate-topk=%d "
        "target-topk=%d npu-k=%d "
        "search-mode=%s qps-mode=%s replicas=%d duration=%d loops=%d\n",
        DevicesJson().c_str(), g_dim, g_ntotal, g_nlist, g_nprobe, g_quantBits, g_k, g_recallK, g_npuK,
        SearchModeName(), QpsModeName(), SearchReplicaCount(), g_durationSec, g_loops);
    fflush(stdout);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
