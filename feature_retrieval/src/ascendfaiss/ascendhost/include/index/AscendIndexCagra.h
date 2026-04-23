#ifndef ASCEND_INDEX_CAGRA_H
#define ASCEND_INDEX_CAGRA_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>

namespace faiss {
namespace ascend {
using APP_ERROR = int;
class AscendIndexCagraImpl;
// ==============================================
// 【CAGRA 初始化参数】：建图/建库时固定，建图后不可变
// ==============================================
constexpr int64_t DEFAULT_MEM = 0x8000000;
struct IndexCagraInitParams {
    IndexCagraInitParams() = default;

    IndexCagraInitParams(int dim, int graph_degree, const std::vector<int>& deviceList,
                         int64_t ascendResourceSize)
        : dim(dim),
          graph_degree(graph_degree),
          deviceList(deviceList),
          ascendResourceSize(ascendResourceSize)
    {
    }

    int dim = 128;
    int graph_degree = 64;
    std::vector<int> deviceList = {0};
    int64_t ascendResourceSize = DEFAULT_MEM;
};

// ==============================================
// 【CAGRA 检索超参】：运行时可动态修改（唯一接口）
// ==============================================
struct IndexCagraSearchParams {
    IndexCagraSearchParams() = default;

    IndexCagraSearchParams(int topk, size_t dataNum, int hashBitlen)
        : topk(topk),
          dataNum(dataNum),
          hashBitlen(hashBitlen)
    {
    }

    int topk = 32;
    size_t dataNum = 1000000;
    int hashBitlen = 8;
};

// ==============================================
// 【CAGRA 构图配置】：构图时的参数配置
// ==============================================
struct IndexCagraBuildConfig {
    IndexCagraBuildConfig() = default;

    IndexCagraBuildConfig(uint32_t graph_degree, int64_t data_size)
        : graphDegree(graph_degree),
          dataSize(data_size)
    {
    }

    uint32_t graphDegree = 32;           // 图的度数（K值）
    int64_t dataSize = 0;                // 数据大小（数据点数量）
};

// ==============================================
// CAGRA 索引类
// ==============================================
class AscendIndexCagra {
public:
    AscendIndexCagra();

    virtual ~AscendIndexCagra();

    APP_ERROR Init(const IndexCagraInitParams& params, const IndexCagraSearchParams& searchParams);

    // ====================== 图结构 + 向量库 ======================
    APP_ERROR AddGraph(const std::vector<uint32_t>& graphData, const std::string& saveBinPath);

    // ====================== 构图接口 ======================
    APP_ERROR BuildGraph(int64_t n, const float* data, const std::string& graphFilePath, 
                         const IndexCagraBuildConfig& buildConfig);

    // ====================== 检索接口 ======================
    APP_ERROR Search(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
        const float* data, float* dists, uint32_t* labels);

private:
    std::shared_ptr<AscendIndexCagraImpl> pIndexCagraImpl;
    std::mutex mtx;
};

} // namespace ascend
} // namespace faiss

#endif // ASCEND_INDEX_CAGRA_H
