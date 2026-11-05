# AscendCagraGraphBuilder

## 功能介绍

AscendCagraGraphBuilder提供单张Ascend 950上的CAGRA近邻图构建能力。输入Host侧连续FP32向量，使用平方L2距离构图，输出固定出度的邻接矩阵。

头文件为`${mxIndex_install_path}/mxIndex/include/faiss/ascend/AscendCagraGraphBuilder.h`，命名空间为`faiss::ascend`。
构图配置参见[AscendCagraGraphBuildConfig](./20_AscendCagraGraphBuildConfig.md)。

本类与[AscendIndexCagra](./18_AscendIndexCagra.md)分别负责构图和检索。`AscendIndexCagra::Add`接收已有图，不会自动调用Builder；需要NPU构图时，先创建Builder并调用`Build`，再将图传入`Add`。已有图的调用方可以直接使用检索接口。

## AscendCagraGraphBuilder接口

<table><tbody>
<tr><td width="140" align="center">API定义</td><td><strong><code>AscendCagraGraphBuilder();</code></strong></td></tr>
<tr><td align="center">功能描述</td><td>创建构图实例。构造函数不执行构图。</td></tr>
<tr><td align="center">输入/输出/返回值</td><td>无。</td></tr>
<tr><td align="center">约束说明</td><td>禁用拷贝构造和拷贝赋值；使用前须成功调用Init。对象析构时释放其持有的构图资源。</td></tr>
</tbody></table>

## Init接口

<table><tbody>
<tr><td width="140" align="center">API定义</td><td><strong><code>APP_ERROR Init(int dim, int graphDegree, int dataNum, const std::vector&lt;int&gt;&amp; deviceList);</code></strong></td></tr>
<tr><td align="center">功能描述</td><td>初始化向量维度、最终图度数、节点数及设备资源。</td></tr>
<tr><td align="center">输入</td><td><code>dim</code>：向量维度D。<br><code>graphDegree</code>：最终图每行邻居数G。<br><code>dataNum</code>：底库向量数N。<br><code>deviceList</code>：仅包含一个有效设备ID的列表。</td></tr>
<tr><td align="center">输出</td><td>无。</td></tr>
<tr><td align="center">返回值</td><td>APP_ERROR：0表示成功，非0表示失败。</td></tr>
<tr><td align="center">约束说明</td><td>● 仅支持单张Ascend 950。<br>● dim取值[1,3072]；常用验证档位为{64,128,256,384,512,1024,1536,2048,3072}。<br>● dataNum取值[2,INT32_MAX]，并受Host/Device内存限制。<br>● graphDegree满足0&lt;G&lt;N；执行Build时还须满足G≤M≤128、M&lt;N。<br>● 同一对象成功Init后不能再次Init；改变N/D/G时创建新对象。<br>● 构图OM的N/D/M/G须与运行参数一致。</td></tr>
</tbody></table>

## Build接口

<table><tbody>
<tr><td width="140" align="center">API定义</td><td><strong><code>APP_ERROR Build(const float* data, uint32_t* graph, const AscendCagraGraphBuildConfig&amp; config = {});</code></strong></td></tr>
<tr><td align="center">功能描述</td><td>执行NN-Descent迭代、剪枝及合并，将最终图写入调用方提供的Host缓冲区。每次调用构建完整图，不是增量插入。</td></tr>
<tr><td align="center">输入</td><td><code>data</code>：行主序FP32底库，长度N×D。<br><code>config</code>：构图配置，默认值及范围见配置文档。</td></tr>
<tr><td align="center">输出</td><td><code>graph</code>：调用方预先分配的Host uint32_t缓冲区，长度至少N×G；第i行第j个邻居位于<code>graph[i*G+j]</code>。</td></tr>
<tr><td align="center">返回值</td><td>APP_ERROR：0表示成功，非0表示失败。未初始化时返回APP_ERR_INDEX_NOT_INIT；非法指针或配置返回非0错误。</td></tr>
<tr><td align="center">约束说明</td><td>● 先成功调用Init，data和graph均不能为空。调用方保证缓冲区长度；裸指针接口不接收数组长度。<br>● 输入全部为有限FP32值；NaN及正负无穷不支持，需预先清洗。<br>● 中间度数M满足G≤M≤128且M&lt;N；maxIterations&gt;0，terminationThreshold取值[0,1]。<br>● 每行输出G个合法邻居ID，无自环及行内重复。<br>● 开启连通性保障时包含Host回传和补边处理；连通性按弱连通口径检查，不承诺强连通。<br>● 构图结果用于检索时，D、G、N须与检索接口及OM一致；TopK在检索接口中配置，支持规格见<a href="./18_AscendIndexCagra.md">AscendIndexCagra接口文档</a>。</td></tr>
</tbody></table>

## BuildToFile接口

<table><tbody>
<tr><td width="140" align="center">API定义</td><td><strong><code>APP_ERROR BuildToFile(const float* data, const std::string&amp; graphFilePath, const AscendCagraGraphBuildConfig&amp; config = {});</code></strong></td></tr>
<tr><td align="center">功能描述</td><td>执行完整构图并将邻接矩阵写入二进制文件；不是保存上一次Build的结果。</td></tr>
<tr><td align="center">输入</td><td><code>data</code>和<code>config</code>与Build相同。<br><code>graphFilePath</code>：非空输出路径，父目录须已存在且可写。</td></tr>
<tr><td align="center">输出</td><td>无头部的行主序uint32_t邻接矩阵，文件大小N×G×4字节；不包含底库、维度信息或量化编码。</td></tr>
<tr><td align="center">返回值</td><td>APP_ERROR：0表示构图及文件写入成功，非0表示失败。</td></tr>
<tr><td align="center">约束说明</td><td>● 与Build使用相同的参数、模型及数据约束。<br>● 已有同名文件将被覆盖；调用方保存N/G及对应底库，并保证路径可信。<br>● 读取图文件后可传入AscendIndexCagra::Add。两次独立构图不要求邻接ID或顺序逐项相同。</td></tr>
</tbody></table>

## 调用示例

以下代码以SIFT1M的D=128、G=64、TopK=32展示调用顺序，示例值可按接口支持规格调整并生成匹配的OM。底库、查询和输出缓冲由调用方准备，每次调用均检查返回码。

```cpp
#include <faiss/ascend/AscendCagraGraphBuilder.h>
#include <faiss/ascend/AscendIndexCagra.h>
#include <stdexcept>

using namespace faiss::ascend;
auto check = [](APP_ERROR ret) {
    if (ret != 0) throw std::runtime_error("CAGRA API failed");
};
const int dim = 128, degree = 64, count = 1000000, topK = 32;
std::vector<int> devices{0};
std::vector<uint32_t> graph(static_cast<size_t>(count) * degree);
std::vector<uint32_t> hash(static_cast<size_t>(count) * 2, 0);
AscendCagraGraphBuildConfig config;
config.verbose = true;
AscendCagraGraphBuilder builder;
check(builder.Init(dim, degree, count, devices));
check(builder.Build(baseData, graph.data(), config));

AscendIndexCagra index;
check(index.Init(dim, degree, count, topK, devices));
check(index.Add(graph.data(), hash.data(), baseData));
check(index.QuantizeData(queryNum, queryData, count, baseData));
check(index.Search(queryNum, queryData, topK, distances, labels));
// 底库不变，下一批查询复用底库量化结果。
check(index.QuantizeQueries(nextQueryNum, nextQueryData));
check(index.Search(nextQueryNum, nextQueryData, topK, nextDistances, nextLabels));
```

完整SIFT1M测试命令及模型生成方法见[用户指南的Cagra章节](../../05_user_guide.md#cagra)。
