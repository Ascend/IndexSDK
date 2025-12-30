# DiskIndex算法使用说明

## 接口总览

### OpenGauss适配接口

#### PQ量化相关结构体

```c++
typedef struct DiskPQParams {
    int pqChunks = 512;
    int funcType = 0;
    int dim = 4096;
    char *pqTable = nullptr;
    uint32_t *offsets = nullptr;
    char *tablesTransposed = nullptr;
    char *centroids = nullptr;
} DiskPQParams;

```

参数：

* `pqChunks`: int类型，表示将原始向量维度dim切分为pqChunks块。限制：`1 <= pqChunks <= min(dim, 512)`。使用较小pqChunks将使用更少内存，但会带来相应的精度损失。数据维度 >= 1024的场景下推荐使用最大chunk数 `512`。默认值为 `512`。

* `funcType`: int类型，表示进行PQ查表距离计算时使用的计算标准。限制: `funcType` $\in$ `[0, 1]`。`0` 表示使用L2距离， `1`表示使用IP距离。默认值为 `0`。**目前仅支持L2距离。**

* `dim`: int类型，表示原始数据维度。限制：`暂无`，*后需限制范围*。默认值为 `4096`。

* `pqTable`: char* 类型，存储码本数据的指针。默认值为nullptr。限制：

    * 码本构建阶段，pqTable必须为nullptr，在动态库内部将使用 `new []` 关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用 `delete []`）。内部申请的内存大小保证等于 `dim * 256 （256为每个chunk内的聚类数） * sizeof(使用的数据类型)` 字节。

    * 码字构建（底库压缩）阶段，pqTable必须指向内存大小为 `dim * 256 * sizeof(使用的数据类型)` 字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。

    * 检索阶段，仅需填充 `tablesTransposed` 指针内的内容 (描述见下)，对 `pqTable` 无要求。

    * **目前仅支持数据类型为 `float`。**

* `offsets`: uint32_t *类型，存储每个chunk在原始维度上起始和截至的维度。默认值为nullptr。限制：
    * 码本构建阶段，offsets必须为nullptr，在动态库内部将使用 `new []` 关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用 `delete []`）。内部申请的内存大小保证等于 `(pqChunks + 1) * sizeof(uint32_t)` 字节。

    * 码字构建（底库压缩）阶段和检索阶段，offsets必须指向内存大小为 `(pqChunks + 1) * sizeof(uint32_t)` 字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。

* `tablesTransposed`: char* 类型，存储码本数据的转置形态指针。默认值为nullptr。限制：

    * 码本构建阶段，tablesTransposed必须为nullptr，在动态库内部将使用 `new []` 关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用 `delete []`）。内部申请的内存大小保证等于 `dim * 256 * sizeof(使用的数据类型)` 字节。

    * 检索阶段，tablesTransposed必须指向内存大小为 `dim * 256 * sizeof(使用的数据类型)` 字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。

    * 码字构建（底库压缩） 阶段，仅需填充 `pqTable` 指针内的内容，对`tablesTransposed` 无要求。

    * 目前仅支持数据类型为 `float`。

* `centroids`: char *类型，存储每个维度的平均值，用于对数据进行中心化处理。默认值为nullptr。限制：
    * 码本构建阶段，centroids必须为nullptr，在动态库内部将使用 `new []` 关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用 `delete []`）。内部申请的内存大小保证等于 `dim * sizeof(使用的数据类型)` 字节。

    * 码字构建（底库压缩）阶段和检索阶段，centroids必须指向内存大小为 `dim * sizeof(使用的数据类型)` 字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。


#### 向量包装结构体

```c++
typedef struct VectorArrayData {
    int length;
    int maxlen;
    int dim;
    size_t itemsize;
    char *items;
} VectorArrayData;
```
参数：

* `length`: int类型，结构体中存储的向量条数。用户需保证`items`指向的数据字节大小等于 `dim * length * itemSize`。默认值为0，范围限制：`暂无，待补充`。

* `dim`: int类型，结构体中存储的向量维度。用户需保证`items`指向的数据字节大小等于 `dim * length * itemSize`。默认值为4096，范围限制：`暂无，待补充`。

* `itemSize`: size_t类型，`items`指向的实际数据的每个单位的字节大小。如float类型则为4，uint16_t/float16_t类型则为2。默认值为4。**目前由于内部仅支持float类型，该值目前内部暂未使用，仅作为保留字段供适配其他数据类型。**

* `items`: char* 类型，存储VectorArrayData内数据的指针。用户需保证`items`指向的数据字节大小等于 `dim * length * itemSize`。默认值为nullptr。

* `maxlen`: int类型。**TODO: OpenGauss对应的VectorArrayData结构体保留字段，之后需确定正确含义**

#### ComputePQTable

```c++
int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);
```
使用sample中存储的采样的底库数据计算PQ码本，并将码本相关的数据存储在参数`params`中的对应参数中。`params`中参数的具体内容见上。

参数：
* `sample`: 指向填充好采样底库数据的`VectorArrayData`实例的指针。不能为空指针。数据填充要求见上。

* `params`: 指向仅包含PQ参数，未填充训练好的PQ数据的`DiskPQParams`实例的指针。不能为空指针。数据填充要求见上。

返回值：

返回值为0时表示流程正常，返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。

#### ComputeVectorPQCode

```c++
int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);
```
使用填充好PQ数据的params，对baseData中的底库数据进行量化，并将量化数据写入pqCode指向的缓存区中。

参数：

* `baseData`: 指向填充好底库数据的`VectorArrayData`实例的指针。不能为空指针。数据填充要求见上。考虑到磁盘检索的内存限制，用户需在外层决定baseData中底库数据的大小。

* `params`: 指向填充好PQ参数和训练好的PQ数据的`DiskPQParams`实例的指针。不能为空指针。数据填充要求见上。

* `pqCode`: uint8_t *类型，接收返回的压缩好的底库向量的指针。不能为空指针。用户需保证指向的空间至少有 `length * pqChunks` 字节数大小 （`length`为`VectorArrayData`参数， `pqChunks`为`DiskPQParams`参数，具体含义见上）。

返回值：返回值为0时表示流程正常，返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。

#### GetPQDistanceTable
```c++
int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);
```

使用填充好PQ数据的params，对vec指向的query数据进行ADC PQ距离计算，并将pq距离表写入pqDistanceTable指向的缓存区中。

参数：

* `vec`: char *类型，指向待计算的query数据的指针。用户需保证vec指向的空间至少有 `dim * sizeof(使用的数据类型)` 字节大小的数据。目前仅支持float类型。

* `params`: 指向填充好PQ参数和训练好的PQ数据的`DiskPQParams`实例的指针。不能为空指针。数据填充要求见上。

* `pqDistanceTable`: float*类型，接收返回的query与每个chunk内每个centroid距离的指针。用户需保证 pqDistanceTable 指向的空间至少有 `pqChunks * 256 * sizeof(float)` 字节数大小。


返回值：返回值为0时表示流程正常，返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。

#### GetPQDistance

```c++
int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &pqDistance);
```
使用`baseCode`指向的底库向量对应的压缩码字数据和 `GetPQDistanceTable`接口中获取的pqDistanceTable，计算query与这个底库向量的PQ距离。（OpenGauss侧需确认是否需要提供一个计算query和一组底库向量距离的接口。）

参数：

* `basecode`: uint8_t *类型，指向一个底库向量对应的压缩码字数据的指针。用户需保证指针指向的数据至少有 `pqChunks` 个字节。

* `params`: 指向至少填充好`pqChunks`数值的`DiskPQParams`实例的指针。不能为空指针。

* `pqDistanceTable`: float*类型，指向query对应的ADC PQ距离表的指针。用户需保证 pqDistanceTable 指向的数据至少有 `pqChunks * 256 * sizeof(float)` 字节数大小。 

* `pqDistance`: float&类型，接收最终输出的pq距离的引用值。内部不会在使用前对 `pqDistance`置零，pqDistance最终结果为原 pqDistance 值 + 输出的query与baseCode的PQ距离，因此推荐输入该值为0。

返回值：返回值为0时表示流程正常，返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。


### Linux侧接口


#### 构造函数
```c++
explicit DiskIndex(diskann_pro::Metric distMetric, int searchListSize, int threadNum, bool verbose);
```
DiskIndex的构造函数。

*入参：*

* `distMetric`: 自定义的计算标准枚举类；类成员为：`L2` (0), `INNER_PRODUCT` (1)；默认值为 `L2`，范围限制：`[待填]`；
* `searchListSize`: int类型，索引构建/检索时使用的优先队列大小，直接影响检索精度和性能；默认值为 100，范围限制：`[待填]`；
* `threadNum`: int类型，索引构建/检索时使用的线程数，影响检索/构建时的并发能力；默认值为 1，范围限制：`[待填]`；
* `verbose`: bool类型，控制是否在执行中增加打印信息。默认值为false。

*返回值：*

无

***
#### Build

```c++
void Build(const DiskIndexBuildParams &buildParams);
```
根据参数构建DiskIndex索引文件并输出至对应前缀下。

*入参：*

* `buildParams`: 自定义`DiskIndexBuildParams`结构体，该结构体具体定义为：

```c++
struct DiskIndexBuildParams {
    std::string baseFilePath = "";
    int pqChunkNum = 0;
    int indexMemLimit = 0;
    int degree = 64;
    std::string outputPrefix = "";
};
```

该结构体各参数说明如下：

* `baseFilePath`: c++11字符串类型，索引构建阶段使用的原始数据在磁盘上的路径。该文件规格需要符合以下标准：
    * 文件首1-4字节需为uint32_t类型，描述文件中的总向量数npts (如原始底库为 1000条 * 256维底库向量，则该值需为 1000)；
    * 文件首5-8字节需为uint32_t类型，描述文件中的向量维度dim (如原始底库为 1000条 * 256维底库向量，则该值需为 256)；
    * 文件剩余字节需满足大小足够 dim * npts * sizeof(float) 字节，即真正待添加的原始底库向量数据；

* `pqChunkNum`: int类型，对底库向量进行PQ量化时对dim的分段数；默认值为：`代填`，范围限制：`0 < pqChunkNum <= dim`；

* `indexMemLimit`: int类型，底库构建时使用的最大内存大小，算法内部会通过该值对底库构建流程进行分片；默认值为：`代填`，范围限制：`待填`；

* `degree`: int类型，图索引构建时每个节点的最大邻居数，即图的度数；默认值为：`代填`，范围限制：`待填`；

* `outputPrefix`: c++11字符串类型，索引构建输出文件的前缀。索引最终输出构建文件为：
```
{prefix}_disk.index // 磁盘索引文件
{prefix}_disk.index_align // 图索引结构
{prefix}_pq_compressed.bin // PQ压缩后的底库向量
{prefix}_pq_pivots.bin // PQ码本
```
 如果`indexMemLimit`小于图索引构建需要的内存导致图索引构建分片的话，则额外输出2个文件：
```
{prefix}_disk.index.medoids.bin
{prefix}_disk.index.centroids.bin
```

*返回值：*

无

***
#### Load
```c++
void Load(const std::string &inputPrefix);
```
加载索引内容。

*入参：*

* `inputPrefix`: C++11字符串。加载索引的路径前缀；该前缀需要与`Build`接口内的`outputPrefix`一致，接口内会根据`inputPrefix`提供的前缀去寻找上述提到的索引文件。

*返回值：*

无

***

#### LoadCacheList

```c++
void LoadCacheList(int cacheNum);
```
检索阶段，检索前需缓存部分节点于内存内。

*入参：*

* `cacheNum`: int类型，

*返回值：*

无

***

（Work In Progress）