# AscendIndexFlat<a id="ZH-CN_TOPIC_0000001506334757"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506334829"></a>

AscendIndexFlat是最基础的特征检索，存储FP16浮点数类型特征向量并执行暴力搜索。

支持多线程并发调用，需要设置“MX\_INDEX\_MULTITHREAD”环境变量为1，即export MX\_INDEX\_MULTITHREAD=1，设置为其他值或者不设置，则表示不开启多线程功能。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

> [!NOTE]
>AscendIndexFlat算法L2和IP距离支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，使用在线算子需要用户在应用程序的最后显式调用 \(void\)aclFinalize\(\) （需要包含头文件：\#include "acl/acl.h"）。

## Flat Bitmap过滤检索<a name="flat-bitmap-filtering"></a>

### 使用前准备

安装并配置IndexSDK、CANN、驱动和目标维度的Flat/AICPU算子，无需新增Bitmap专用算子。以下示例使用64维；性能示例需准备4096维算子。若启用在线算子转换，仍须遵守关于`aclFinalize()`的要求，并先释放索引资源。

接口及参数详见[search_with_selector接口](#search-with-selector)和[AscendIndexFlatConfig](./09_AscendIndexFlatConfig.md#flat-shard-mode)。本期提供C++接口，未增加Python绑定。

### 单设备调用与删除后查询

下面使用非连续自定义ID，选中5个候选，整个batch共用一份Bitmap。删除向量会改变位置映射，但不需要根据底库位置重排Bitmap。

```cpp
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/impl/IDSelector.h>

int main()
{
    constexpr int d = 64;
    constexpr faiss::idx_t nb = 64, nq = 3, k = 3;
    std::vector<float> xb(nb * d);
    for (std::size_t i = 0; i < xb.size(); ++i) {
        xb[i] = static_cast<float>((i * 17 + 3) % 101) / 101.0f;
    }
    std::vector<float> xq(xb.begin(), xb.begin() + nq * d);
    std::vector<faiss::idx_t> ids(nb);
    for (faiss::idx_t i = 0; i < nb; ++i) ids[i] = 1000 + i * 7;

    faiss::ascend::AscendIndexFlatConfig config({0});
    faiss::ascend::AscendIndexFlat index(d, faiss::METRIC_INNER_PRODUCT, config);
    index.add_with_ids(nb, xb.data(), ids.data());

    const faiss::idx_t allowed[] = {ids[3], ids[17], ids[29], ids[42], ids[57]};
    std::vector<uint8_t> bitmap(static_cast<std::size_t>(ids.back() / 8 + 1), 0);
    for (auto id : allowed) bitmap[id / 8] |= uint8_t(1u << (id % 8));
    faiss::IDSelectorBitmap selector(bitmap.size(), bitmap.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search_with_selector(nq, xq.data(), k, distances.data(), labels.data(), &selector);
    for (auto id : labels) assert(id == -1 || selector.is_member(id));

    faiss::IDSelectorRange removed(ids[17], ids[17] + 1);
    assert(index.remove_ids(removed) == 1);
    index.search_with_selector(nq, xq.data(), k, distances.data(), labels.data(), &selector);
    for (auto id : labels) {
        assert(id == -1 || (id != ids[17] && selector.is_member(id)));
    }
}
```

Bitmap构造长度是字节数。连续ID `[0, nb)` 可分配`(nb + 7) / 8`字节；自定义ID应按需要选中的最大非负ID确定长度，不能直接用底库条数计算。稀疏且很大的ID可能需要较大的位图。短位图未覆盖的ID不参与检索，空位图表示无候选，与空Selector指针不同。

使用`index.add(nb, xb.data())`生成与位置一致的连续ID时，内部自动启用Bitmap直接复制路径。删除或导入导致映射变化后，内部自动选择正确路径。FP16查询使用`const uint16_t*`重载，输入必须是真实的FP16位模式；不能强制转换FP32指针。L2可使用`METRIC_L2`或继承该接口的`AscendIndexFlatL2`。

### 多设备完整副本

进程启动前设置已有的多线程安全开关：

```bash
export MX_INDEX_MULTITHREAD=1
```

将单设备示例的配置替换为：

```cpp
faiss::ascend::AscendIndexFlatConfig config(
    {0, 1}, faiss::ascend::FLAT_DEFAULT_MEM, false);
```

- 默认`shard=true`表示分片；多设备Selector查询必须设为`false`，否则报错。单设备支持任一模式。
- 每个设备保存完整底库，每次调用轮询选择一个设备执行整个batch，不合并跨设备结果。每个设备需有足够内存。
- `MX_INDEX_MULTITHREAD`不是Bitmap功能新增的开关，它启用已有索引锁及线程安全资源，首次读取后缓存。多设备副本构造即检查该开关，初始化后再设置无效。
- 环境变量不自动创建查询线程。调用方使用固定工作线程并发查询，各线程使用独立输出缓冲；Selector及Bitmap只读共享，调用期间不得修改或释放。
- 添加、删除和重置同步作用于副本，`ntotal`和删除返回值计逻辑数量；`getBaseSize(device)`返回单个设备的向量数，`copyTo`导出一份底库。

## AscendIndexFlat接口<a name="ZH-CN_TOPIC_0000001456375308"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexFlat *index</code></strong>：CPU侧index资源。<br><strong><code>AscendIndexFlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

<a name="table1735274911381"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat的构造函数，基于一个已有的“index”创建Ascend上的检索Index。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIDMap *index</code></strong>：CPU侧Index资源。<br><strong><code>AscendIndexFlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexIDMap指针，该Index的成员索引维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

<a name="table142416323911"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat的构造函数，生成维度为dims的AscendIndexFlat（单个Index管理的一组向量的维度是唯一的），此时根据“config”中配置的值设置Device侧资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int dims</code></strong>：AscendIndex管理的一组特征向量的维度。<br><strong><code>faiss::MetricType metric</code></strong>：AscendIndexFlat在执行特征向量相似度检索的时候使用的距离度量类型。<br><strong><code>AscendIndexFlatConfig config</code></strong>：Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dims ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}。<br>● metric ∈ {faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

<a name="table5169814143913"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlat(const AscendIndexFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此index拷贝构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexFlat&amp;</code></strong>：常量AscendIndexFlat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

<a name="table04891725153918"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>virtual ~AscendIndexFlat();</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat的析构函数，销毁AscendIndexFlat对象，释放资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## copyFrom接口<a name="ZH-CN_TOPIC_0000001456535180"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexFlat *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat基于一个已有的Index拷贝到Ascend，清空当前的AscendIndexFlat底库，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexFlat *index</code></strong>：CPU侧index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，该Index的维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

<a name="table525914213409"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyFrom(const faiss::IndexIDMap *index);</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat基于一个已有的“index”拷贝到Ascend，清空当前的AscendIndexFlat底库，并保持原有的AscendIndex的Device侧资源配置。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const faiss::IndexIDMap *index</code></strong>：CPU侧index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">index需要为合法有效的IndexIDMap指针，否则可能造成程序崩溃或功能不可用，该Index的成员索引维度d参数取值范围为{32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}，底库向量总数的取值范围：0 ≤ n &lt; 1e9，metric_type参数取值为{faiss::MetricType::METRIC_L2, faiss::MetricType::METRIC_INNER_PRODUCT}。</td></tr>
</tbody></table>

## copyTo接口<a name="ZH-CN_TOPIC_0000001456535148"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexFlat *index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexFlat的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexFlat *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的CPU Index指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

<a name="table154531752144016"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void copyTo(faiss::IndexIDMap *index) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">将AscendIndexFlat的检索资源拷贝到CPU侧。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>faiss::IndexIDMap *index</code></strong>：CPU侧Index资源。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“index”需要为合法有效的IndexIDMap指针，Index占用的资源由用户释放内存。</td></tr>
</tbody></table>

## getBase接口<a name="ZH-CN_TOPIC_0000001456375236"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void getBase(int deviceId, char* xb) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该AscendIndexFlat在特定“deviceId”上管理的特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int deviceId</code></strong>：Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>char* xb</code></strong>：AscendIndexFlat在“deviceId”上存储的底库特征向量。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“deviceId”需要为合法值的设备ID。<br>“xb”需要为非空指针，且长度应该为dims * BaseSize * sizeof(float32)字节，否则可能出现越界读写错误并引起程序崩溃，其中BaseSize为getBaseSize函数的返回值。</td></tr>
</tbody></table>

## getBaseSize接口<a name="ZH-CN_TOPIC_0000001456854956"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>size_t getBaseSize(int deviceId) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该AscendIndexFlat在特定“deviceId”上管理的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int deviceId</code></strong>：Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">在特定“deviceId”上的特征向量数量。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“deviceId”需要为合法的设备ID。</td></tr>
</tbody></table>

## getIdxMap接口<a name="ZH-CN_TOPIC_0000001506334785"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void getIdxMap(int deviceId, std::vector&lt;idx_t&gt; &amp;idxMap) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">获取该AscendIndexFlat在特定“deviceId”上管理的特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>int deviceId</code></strong>：Device侧设备ID。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>std::vector&lt;idx_t&gt; &amp;idxMap</code></strong>：AscendIndexFlat在“deviceId”上存储的底库特征向量ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">“deviceId”需要为合法的设备ID。</td></tr>
</tbody></table>

## operator = 接口<a name="ZH-CN_TOPIC_0000001506495701"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>AscendIndexFlat&amp; operator=(const AscendIndexFlat&amp;) = delete;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">声明此Index赋值构造函数为空，即不可拷贝类型。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>const AscendIndexFlat&amp;</code></strong>：常量AscendIndexFlat。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">无</td></tr>
</tbody></table>

## search\_with\_masks接口<a name="ZH-CN_TOPIC_0000001810529650"></a>

<a name="table7235918388"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为<strong><code>0</code></strong>、<strong><code>1</code></strong>比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，<strong><code>1</code></strong>参与，<strong><code>0</code></strong>不参与。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const float *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const void *mask</code></strong>：特征底库掩码。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处“k”通常不允许超过4096。<br>● 此处指针“x”需要为非空指针，且长度应该为dim * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 此处指针“distances”/“labels”需要为非空指针，且长度应该为<strong><code>k</code></strong> * <strong><code>n</code></strong>，否则可能出现越界读写错误并引起程序崩溃。<br>● 此处指针“mask”需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。<br>● mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。<br>● 支持单device或多device完整副本（<code>shard=false</code>）；不支持将同一mask直接用于多device分片。副本模式需满足配置文档中的多线程约束。</td></tr>
</tbody></table>

<a name="table0628133121511"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search_with_masks(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const void *mask) const;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">功能描述</td><td valign="middle">AscendIndexFlat的特征向量查询接口，根据输入的特征向量返回最相似的k条特征的ID。mask为0、1比特串，每个比特代表底库中对应顺序的特征是否参与距离计算，1参与，0不参与。</td></tr>
<tr><td width="140" align="center" valign="middle">输入</td><td valign="middle"><strong><code>idx_t n</code></strong>：查询的特征向量的条数。<br><strong><code>const uint16_t *x</code></strong>：特征向量数据。<br><strong><code>idx_t k</code></strong>：需要返回的最相似的结果个数。<br><strong><code>const void *mask</code></strong>：特征底库掩码。</td></tr>
<tr><td width="140" align="center" valign="middle">输出</td><td valign="middle"><strong><code>float *distances</code></strong>：查询向量与距离最近的前“k”个向量间的距离值。<br><strong><code>idx_t *labels</code></strong>：查询的距离最近的前“k”个向量的ID。</td></tr>
<tr><td width="140" align="center" valign="middle">返回值</td><td valign="middle">无</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 此处“n”的取值范围：0 &lt; n &lt; 1e9。<br>● 此处“k”通常不允许超过4096。<br>● 此处指针“x”需要为非空指针，且长度应该为dim * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 此处指针“distances”/“labels”需要为非空指针，且长度应该为k * n，否则可能出现越界读写错误并引起程序崩溃。<br>● 此处指针“mask”需要为非空指针，且长度应该为n*ceil(ntotal/8)，否则可能出现越界读写错误并引起程序崩溃，其中ntotal为底库特征数量。<br>● mask是按照底库的顺序来设置的，如果调用此接口前有调用remove_ids删除特征向量，会导致底库特征顺序改变，请先通过调用getIdxMap接口获取底库特征的ID，进而设置mask。<br>● 支持单device或多device完整副本（<code>shard=false</code>）；不支持将同一mask直接用于多device分片。副本模式需满足配置文档中的多线程约束。</td></tr>
</tbody></table>

## search_with_selector接口<a name="search-with-selector"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">API定义</td><td valign="middle"><strong><code>void search_with_selector(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const faiss::IDSelector *sel) const;</code></strong><br><strong><code>void search_with_selector(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels, const faiss::IDSelector *sel) const;</code></strong></td></tr>
<tr><td align="center">功能描述</td><td>在Selector允许的候选集合内执行TopK检索，同一次调用的全部query共用一个Selector。支持IP和L2，返回外部ID。重点优化<code>faiss::IDSelectorBitmap</code>，其他Selector保留<code>is_member</code>通用路径。</td></tr>
<tr><td align="center">输入</td><td><code>n</code>：查询数量。<br><code>x</code>：主机侧连续查询数组，长度至少<code>n * d</code>。FP32重载内部转FP16；<code>uint16_t</code>重载接收FP16位模式，不能直接将FP32指针强制转换。<br><code>k</code>：每条query的返回数量。<br><code>sel</code>：非空Selector指针，需要包含<code>&lt;faiss/impl/IDSelector.h&gt;</code>。</td></tr>
<tr><td align="center">输出</td><td><code>distances</code>：主机侧FP32距离/相似度数组。<code>labels</code>：主机侧外部ID数组，均至少分配<code>n * k</code>个元素。候选不足K时沿用现有过滤检索填充规则，无效标签为<code>-1</code>，对应距离不应作为有效结果使用。</td></tr>
<tr><td align="center">返回值</td><td>无。参数校验、数据传输或设备检索失败时抛出异常。</td></tr>
<tr><td align="center">约束说明</td><td><code>0 &lt; n &lt; 1e9</code>，<code>1 &lt;= k &lt;= 4096</code>；<code>x</code>、<code>distances</code>、<code>labels</code>、<code>sel</code>均非空，缓冲容量由调用方保证。需事先向索引添加底库。仅支持单device或多device完整副本，多device分片会明确报错。Selector及底层Bitmap在整个调用期间有效且只读。</td></tr>
</tbody></table>

### Bitmap语义

- 构造方式：`faiss::IDSelectorBitmap(size_t n, const uint8_t *bitmap)`。此处构造参数`n`为**字节数**，与查询数量`n`不同。
- 外部ID `id` 对应 `bitmap[id / 8]` 的第 `id % 8` 位，低位优先。1允许参与检索，0排除；位图范围外的ID排除。
- Bitmap按**外部ID**排列，`search_with_masks` 按**底库位置**排列。自定义ID和删除后的位置变化由接口内部映射，无需调用方重新排列Bitmap。
- 若外部ID等于底库位置，直接复制有效字节；否则按ID重映射。短位图缺失部分补零，超出底库的位忽略。
- 不缓存Selector转换结果，两次调用之间可以修改Bitmap。并发调用时不得修改共享Bitmap，每个线程使用独立输出缓冲。
- 单device Bitmap接入本身不强制设置`MX_INDEX_MULTITHREAD=1`；并发调用及多device副本须遵守[配置约束](./09_AscendIndexFlatConfig.md#flat-shard-mode)。

Bitmap调用示例、设备准备及性能测试命令见[Flat Bitmap过滤检索](#flat-bitmap-filtering)。`AscendIndexFlatL2`继承此接口。
