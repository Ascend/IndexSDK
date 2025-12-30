# IndexGreat算法使用说明

## 什么是IndexGreat算法？
IndexGreat预计将集成了三种我司自研的向量检索算法：`IVFHSP（AMode）`，`HNSW (KMode)`, 和 `KNN (GMode)`, 提供鲲鹏+昇腾高维超大底库近似检索能力。目前（6/17/2024），IndexGreat仅提供GMode的功能。

## 代码仓结构，编译和安装：
相关代码结构如下：
```
|-- mix-index
    |-- beta
        |-- CMakeLists.txt
    ...
    allRun.sh
```

* 运行 `allRun.sh` , 利用`beta/CMakeLists.txt`创建 `libmixsearch.so`二进制.so文件。

* 目前allRun.sh中启动测试的部分已被注释，之后，可以通过修改 `testNpuIndex.cpp` 中的代码和调整 `allRun.sh`中的数据传参来进行不同测试。


## 如何使用IndexGreat算法：

### 使用限制：
1. 目前仅支持精度为float32的向量；
2. 目前仅支持`GMode`接口，下文仅描述`GMode`接口的使用;
3. `索引创建`阶段，底库大小`nb`需要满足`1000 <= nb`，不建议`nb > 1e8`；
4. 距离计算目前仅支持`L2`距离。
****

### 使用前：

引入相关头文件:
```c++
#include "npu/IndexGreat.h"
```

IndexGreat主要功能大体分为两种：

1. `索引创建`：基于底库的特征，创建并保存向量检索索引。

2. `读取索引进行检索`：给定检索用例(query)，使用创建好的索引，在底库中检索topK个与给定query相似度最高的向量，并返回这些底库向量的id。

以上两种功能的使用都需要使用者创建 `IndexGreat` 索引并调用相关接口。    `IndexGreat` 接受两个参数：

1. Mode名称 (`"AMode", "KMode", "GMode"`); 

2. 初始化结构体 (`IndexGreatInitParams`); 。

用户将需要的索引构建参数传入初始化结构体 ，再将构建结构体传入 `IndexGreat` 的构建函数。根据不同的 `Mode`, 初始化结构体有3个构造函数，`GMode`的构造函数具体参数如下：
```c++
// GMode
IndexGreatInitParams(std::string mode, int dim, int R_KGN, int candListSize) {}
```
其中，

* `mode`: C++11字符串，必须为 `"GMode"`；
* `dim`: 底库向量的维度 ，例如底库向量为1024维，则`dim = 1024`；
* `R_KGN`: 仅用于`索引创建`阶段，控制图索引的复杂度，有效取值范围：`50-180`。一般情况下，`R`越大，图索引大小越大，检索精度越高。 
* `candListSize`: 仅用于`向量检索`阶段，规范检索时每个节点向周围检索的节点最大数量，有效取值范围：`candListSize >= 1`。一般情况下，`candListSize`越大，检索耗时越长，检索精度越高。


用例：
```c++
size_t dim = 1024; // 底库向量维度
size_t R = 50; // 图索引精确度
size_t candListSize = 20; // 节点向周围搜寻的最大范围；在检索阶段可调

// creating initilization parameters
auto initParams = ascendSearch::IndexGreatInitParams("GMode", dim, R, candListSize); 
auto index = new ascendSearch::IndexGreat("KGNMode", initParams);
```

***

### 索引创建：
**此部分我们假设用户使用milvus数据库，i.e. 用户需要将数据传入一个uint8指针指向的动态数组。**

创建前，用户需要

1. 提前将需要的底库数据载入容器内（`array`或者`vector`），并确保能有一个指向底库数据起始端的变量指针；
2. 获知底库向量的总数量 (`nb`);

获得以上信息后，用户可以通过`indexGreat.AddVectors_BaseRetain()`接口将底库数据加入创建好的索引内；

```c++
size_t nb = ...; // 底库向量总数量
float* base_ptr = ...; // 指向底库数据起始点的指针
index.AddVectors_BaseRetain(nb, base_ptr); // 将底库数据加入索引
```

**注意：** 此过程耗时较长，256核处理器创建1000万1024维度底库需要~6小时，创建1亿1024维底库需要~12小时。

索引创建完成后，需要通过`indexGreat.WriteIndex_viaPtr()`接口将图索引和底库数据传入uint8的指针内（这个uint8指针必须是空指针，因为该接口不负责管理这个uint8指针已经指向的内存），并传入一个`size_t`大小的参数`length`，记录图索引+地库数据的总字节数大小。

```c++
uint8_t* data = nullptr; // 一个uint8 空指针
size_t length = 0; // 一个size_t大小的参数length
index->WriteIndex_viaPtr(data, length);
```
`WriteIndex_viaPtr` 接口将会在内侧为`data`申请一个足够容纳图索引数据和底库数据的动态数组，并将这些数据存储进入这个动态数组。同时，我们会将这个动态数组的长度 （总字节长度）存储进入`length`变量内。

之后，我们可以选择将uint8指针内的内容和`length`内的数据一起落盘。

```c++
std::string indexPath = ...; //索引的路径
std::ofstream writer(indexPath.c_str(), std::ios::binary);
writer.write((char*)&length, sizeof(length));
writer.write((char*)data, length);
```

### 读取索引进行检索：
**此部分我们假设用户使用milvus数据库，i.e. 用户会尝试将存储的图索引和底库数据存储进入一个uint8指针指向的动态数组。**

首先将图索引数据和底库数据从磁盘内读取出来：

```c++
uint8_t* data = nullptr;
size_t length = 0;

std::string indexPath = ...; //索引的路径


std::ifstream reader(indexPath.c_str(), std::ios::binary);
reader.read((char *)&length, sizeof(length));
data = new uint8_t[length];
reader.read((char *)data, length);       
reader.close();
```
之后，可以调用`index->LoadIndex_viaPtr()`接口，将读出的索引数据存储进入索引，使索引进入可以开始检索的状态。

```c++
index->LoadIndex_viaPtr(data, length); // data和length变量必须按照上面的方式初始化，并存储需要的数据

```
在这之后，用户可以使用提供的`index->Search()`接口开始检索。用户可以选择不同的batch大小进行检索，例如：

```c++
size_t nq = ...;
size_t batchSize = ...; // 每个batch的大小
for (int i = 0; i < nq; i += batchSize) {
    size_t batch_processed = std::min(batchSize, nq - i);
        index->Search(batch_processed, query.data() + i*dim, topK, dists.data() + i*topK, labels.data() + i*topK);
}
```

## 接口总览

```c++
IndexGreat(std::string mode, const IndexGreatInitParams &params);
```
IndexGreat的构造函数。

*入参：*

* `mode`: C++11字符串。限制范围：必须为`"AMode"`, `"GMode`, `"KMode"`中的一个。目前版本，必须使用`"GMode"`。
* `params`: 自定义类索引构建结构体。`"GMode"`的结构体格式请见上。

*返回值：*

无 （构建函数）
***
```c++
void LoadIndex_viaPtr(uint8_t* data_ptr, const size_t& length);
```
支持milvus功能的索引读取接口。

*入参：*

* `data_ptr`: 已经指向图索引数据和底库数据的uint8指针。限制范围：必须指向上述数据。
* `length`: 已经指得图索引数据和底库数据总字节长度的`size_t`类别的变量。限制范围：无。

*返回值：*

无
***
```c++
void WriteIndex_viaPtr(uint8_t*& data_ptr, size_t& length);
```
支持milvus功能的索引存储接口。

*入参：*

* `data_ptr`: 尚未指向图索引数据和底库数据的uint8指针。限制范围：必须为空指针，否则该指针已经指向的内容将丢失。
* `length`: 尚未指得图索引数据和底库数据总字节长度的`size_t`类别的变量。限制范围：无。

*返回值：*

无

***

```c++
APP_ERROR AddVectors_BaseRetain(size_t n, const float* baseData);
```
支持milvus功能的添加底库数据进入索引接口。

*入参：*

* `n`: 底库大小。限制：>= 1000, 最好小于1e8。
* `baseData`: 指向底库数据的指针。限制：不能为空指针。

*返回值：*

APP_ERROR预定义宏，详见 ...
***
```c++
APP_ERROR AddVectorsWithIds(size_t n, const float* baseData, const uint64_t* ids);
```
在添加底库的基础上，允许用户输入一个自定义数组，将检索回的底库向量id映射到这个数组定义的id上。

*额外说明：*

在利用一份底库分散创建多个索引的场合，索引会将每段底库数据从0开始以此赋予id（底库向量在外侧的绝对id被转换成该向量在这个索引内的相对id），并在此基础上进行检索并返回`labels`，这个接口旨在使用户能将`labels`返回的相对id通过`ids`参数转换回绝对`id`，使得`labels`内的结果能参与`gt (ground truth)`的精度校验（因为gt内的底库id都为绝对id）。
例如，`ids = [32, 17, 64 ...]`, `labels`在内部的返回值为 `[0, 1, 2 ...]`, 那在用户获得最后的`labels`之前，`labels`内的值会被映射成`[32, 17, 64 ...]`。

*入参:*
* `n`: 底库大小。限制：>= 1000, 最好小于1e8。
* `baseData`: 指向底库数据的指针。限制：不能为空指针。
* `ids`: 指向映射id的数组。限制：不能为空指针，指向的数据大小必须大于等于`n`（每个传入的底库都必须得到映射）。

*返回值：*

APP_ERROR预定义宏，详见 ...

***
```c++
APP_ERROR Search(size_t n, const float* queryData, int topK, float* dists, int64_t* labels);
```
向量检索接口。

*入参：*

* `n`: 检索的batch大小，支持：1, 4, 8, 16, 32, 50  (不推荐使用超过50的batch)。
* `queryData`: 指向被检索的数据的指针。限制：不能为空指针。
* `topK`: 从底库中寻找 `topK` 个与 `query`相似的向量。限制：topK > 0;
* `dists`：对于每个`query`， 存储其每个返回的`topK`底库向量之间的L2/IP距离。 **注意：** `GMode`模式下，`dists`指向的内容不被被赋值, 该参数无效。
* `labels`: 对于每个`query`， 存储其每个返回的`topK`底库向量的id。

*返回值：*

APP_ERROR预定义宏，详见 ...

*注意事项：*
在`n`小于`nq` (待检索的向量总数量)时，需要对 `queryData`，`labels`, 以及非`GMode`下的`dists`进行指针偏移。例如，检索`dim = 1024`的向量时，如果`n = 4`, 那么代码可能如下：
```c++
size_t = 4;
size_t nq = ...;
size_t dim = 1024; 
for (int i = 0; i < nq; i += batchSize) {
    size_t batch_processed = std::min(batchSize, nq - i);
    index->Search(batch_processed, query.data() + i*dim, topK, dists.data() + i*topK, labels.data() + i*topK);
}
```
***
```c++
APP_ERROR SearchWithMask(size_t n, const float* queryData, int topK, const uint8_t* mask, float* dists, int64_t* labels);
```
在检索接口的基础上，添加了使用户可以输入一个uint8指针指向的数组来掩掉特定底库id的功能。

*入参：*

* `n`: 检索的batch大小，支持：1, 4, 8, 16, 32, 50  (不推荐使用超过50的batch)。
* `queryData`: 指向被检索的数据的指针。限制：不能为空指针。
* `topK`: 从底库中寻找 `topK` 个与 `query`相似的向量。限制：> 0;
* `mask`: 一个指向uin8数据的指针。对于每个uint8数据，将其转化为二进制表达形式，再转化该形式转化为小端序（代表1的bit在最左边）后, 对于为0的bit，在检索过程中无视拥有与该bit位位置对应的id的底库向量。

    限制：
    
    1. `mask`指向的uint数据总量必须大于等于 `nq * ceil(nb/8)`（i.e. 对于每个`query`，我们都提供覆盖足够覆盖整个底库的`mask`量）。
    
    2. 例如, 如果我们在底库id为8-15的底库向量中，想掩掉id == 8的底库向量，那 可得知`mask[1] = 1`，因为：
        ```
            1）8%8 = 0 => mask[1] 包含底库id == 8的mask的对应比特；
            2）若mask[1] == 1, 那 1 in uint8 format = 00000001
            3）转化为"代表1的bit在最左边"的形式 => 10000000，符合预期 （掩掉从左向右数的第八个bit）
        ```
* `dists`：对于每个`query`， 存储其每个返回的`topK`底库向量之间的L2/IP距离。 **注意：** `GMode`模式下，`dists`指向的内容不被被赋值, 该参数无效。
* `labels`: 对于每个`query`， 存储其每个返回的`topK`底库向量的id。

*返回值：*

APP_ERROR预定义宏，详见 ...

***
```c++
size_t GetDim();
```
获得初始化索引时的维度的接口。

*入参：*

无

*返回值：*
类型为`size_t`的初始化索引时的维度。
***

```c++
size_t GetNTotal();
```
获得添加底库数据时输入的底库大小的接口。

*入参：*

无

*返回值：*

类型为`size_t`的初始化索引时的维度。若尚未添加底库(`AddVectors_BaseRetain`)，则返回0。
***

```c++
void SetSearchParams(const IndexGreatSearchParams &params);
```
根据`IndexGreatSearchParams`结构体修改向量检索时的超参。

*入参：*

`IndexGreatSearchParams`自定义结构体。`GMode`下，该结构体的构造函数为：

```c++
IndexGreatSearchParams(std::string mode, int ef): mode(mode), ef(ef) {}
```
`mode`取值必须与索引创建时的`mode`参数保持一致；`ef`与`candListSize`指代同一个参数。

**注意**：此处设置的`ef`会在检索时与参数`topK`进行比较，两者之中的最大值会作为最终的`ef`参数应用于检索中。理论上，`ef`为检索时以`query`为中心向周围搜索的向量最大个数，因此如果最终`ef < topK`, 等同于我们指示算法“仅检索周围`ef`个底库向量，却需要返回`topK`个最接近的底库向量”，这显然是不合理的。因此最终`ef`必须大于等于`topK`。


*返回值：*

类型为`size_t`的初始化索引时的维度。若尚未添加底库(`AddVectors_BaseRetain`)，则返回0。

***

```c++
void Reset();
```
重置索引接口，`GMode`模式下，将索引内装载的底库和图索引清空。

*入参：*

无

*返回值：*

无









  


































