# IndexVstar算法使用说明

## 什么是IndexVstar算法？
IndexVstar为我司自研的晟腾侧向量检索算法,为用户提供昇腾侧高维大底库近似检索能力。

## 代码仓结构：
* 仅展示交付的tar包结构（交付的动态库同时包含`Vstar`, `Great`算法和`IVFSP`算法）。
相关代码结构如下：
```
| -- IVFSP (在此处不展开讲解结构)
| -- lib
    | -- libascendsearch.so（包含Vstar, Great, 和IVFSP算法）
|-- mix-index
    | -- include
    | -- ops
        | -- ascend_c
            | -- AscendcOps
            | -- opsBuild.sh (执行该脚本，在当前目录下生成/configs和/op_models目录，其中op_models/目录下包含生成的离线算子文件，在使用Vstar/Great算法时需要export MX_INDEX_MODELPATH=${pwd}/mix-index/ops/ascend_c/op_models)
    |-- READMEs (该文件所在目录)
    | -- tools
        | -- vstar_generate_models.py （离线算子文件生成脚本）
    | -- train
        | -- vstar_train_codebook.py （码本训练脚本）
        | -- vstar_trainer.py（码本训练逻辑代码）
```

## 如何使用IndexVstar算法：

### 物理硬件限制：
1. 当前机器必须配备`CANN 7.3.0.1.231:8.0.RC2`版本或以上的`Ascend ToolKit`包。 
2. 所处于的python环境需为`Python 3.9.11`或以上，并安装:
    * `torch`: 2.0.1；
    * `torch_npu`: 2.0.1.post4;
        * 若import时遭遇`.../libgomp.so: cannot allocate memory in static TLS block`报错，请执行：`LD_PRELOAD=.../libgomp.so`(报错中出现的`libgomp.so`路径)；
    * `numpy`: 1.25.2或以上;
        * 安装时，若`pip`提示无法安装如下依赖：
            ```python
            ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
            auto-tune 0.1.0 requires decorator, which is not installed.
            dataflow 0.0.1 requires jinja2, which is not installed.
            opc-tool 0.1.0 requires attrs, which is not installed.
            opc-tool 0.1.0 requires decorator, which is not installed.
            opc-tool 0.1.0 requires psutil, which is not installed.
            schedule-search 0.0.1 requires absl-py, which is not installed.
            schedule-search 0.0.1 requires decorator, which is not installed.
            te 0.4.0 requires attrs, which is not installed.
            te 0.4.0 requires cloudpickle, which is not installed.
            te 0.4.0 requires decorator, which is not installed.
            te 0.4.0 requires ml-dtypes, which is not installed.
            te 0.4.0 requires psutil, which is not installed.
            te 0.4.0 requires scipy, which is not installed.
            te 0.4.0 requires tornado, which is not installed.
            ```
            请执行：
            ```python
            pip install attrs cloudpickle decorator jinja2 ml-dtypes psutil scipy tornado absl-py
            ```
            若提示其他依赖安装问题，可暂时无视。例如如下依赖安装报错可暂时无视：
            ```python
            op-compile-tool 0.1.0 requires getopt, which is not installed.
            op-compile-tool 0.1.0 requires inspect, which is not installed.
            op-compile-tool 0.1.0 requires multiprocessing, which is not installed.
            ```
    * `sklearn`: 1.4.1.post1或以上;
    * `tqdm`: 4.66.1或以上；
### 使用限制：
使用流程为：
1. 获得待训练的底库数据；
2. 使用提供的python脚本和底库数据`训练码本`；
3. 创建`Vstar`实例，并利用码本，向未填充索引的实例内添加向量进行`索引创建`;
4. 使用填充好的索引（`索引创建`或`索引加载`后）进行检索；

针对上述的使用流程提出的限制：
1. `索引创建`阶段，底库大小`nb`建议满足`1e4 <= nb`，不建议`nb > 1e8`；
2. 距离计算目前仅支持`L2`距离（返回的`dists`数组内的距离计算的类型）。
3. 使用的数据(底库数据和待检索的数据)建议进行归一化处理（对于每个向量，它的每个元素需要除以这个向量的`L2`范数, i.e. 对于每个向量`v`, 归一化后的该向量每个元素$v_i = \frac{v_i}{||v||}$）。
****

### 使用前：

引入相关头文件:
```c++
#include "npu/NpuIndexVstar.h"
```

## 接口总览

### 构造函数1
```c++
explicit NpuIndexVStar(const IndexVstarInitParams &params);
```
IndexGreat的构造函数。

*入参：*

* `params`: 自定义类索引构建结构体 `IndexVstarInitParams`实例。定义如下：
```c++
struct IndexVstarInitParams {
    IndexVstarInitParams()
    {
    }
    IndexVstarInitParams(int dim, int subSpaceDimL1, int subSpaceDimL2, int nlistL1, int nlistL2,
                         std::vector<int> deviceList)
        : dim(dim),
          subSpaceDimL1(subSpaceDimL1),
          subSpaceDimL2(subSpaceDimL2),
          nlistL1(nlistL1),
          nlistL2(nlistL2),
          deviceList(deviceList)
    {
    }

    int dim = 1024;
    int subSpaceDimL1 = 128;
    int subSpaceDimL2 = 64;
    int nlistL1 = 1024;
    int nlistL2 = 16;
    std::vector<int> deviceList = {0};
    bool verbose = false;
};

```
* `dim`: 输入数据维度。限制范围：必须在`[128, 256, 512, 1024]`之内。
* `subSpaceDimL1`: 第一次降维后的维度。限制范围：1 `subSpaceDimL1 <= 128`; 2）`0 < subSpaceDimL1 < dim`；2）`subspaceDimL1 % 16 == 0`。
* `subSpaceDimL2`: 第二次降维后的维度。限制范围：1）`0 < subSpaceDimL2 < subSpaceDimL1`；2）`subspaceDimL2 % 16 == 0`。
* `nlistL1`: 一级聚类的数量。限制范围：1）`16 <= nlistL1 <= 16384`；2）`nlistL1必须是2的幂(2^x)；`。
* `nlistL2`: 二级聚类的数量。限制范围：1）`16 <= nlistL2 <= 1024`；2）`nlistL2必须是2的幂(2^x)`。
* `deviceList`: 包裹着NPU卡physical ID的vector容器。默认值：{0}。限制范围：内部数字 $\geq$ 0，具体范围请使用`npu-smi`命令查询对应的NPU卡physical ID。限制范围：`deviceList.size() == 1`。
* `verbose`: 布尔值，指定是否开启`verbose`选项，开启后部分操作提供额外的打印提示。默认值：`false`。

*返回值：*

无 （构建函数）

***
### 构造函数2

```c++
explicit NpuIndexVStar(const std::vector<int> &deviceList, const bool verbose);
```
IndexVstar的构造函数，仅用于未知输入数据维度和各种构建超参的情况。使用此构造函数创建的`IndexVstar`实例不能进行`LoadIndex`以外的其他操作。

*入参：*

* `deviceList`: 包裹着NPU卡physical ID的vector容器。默认值：{0}。限制范围：内部数字 $\geq$ 0，具体范围请使用`npu-smi`命令查询对应的NPU卡physical ID。

* `verbose`: 布尔值，指定是否开启`verbose`选项，开启后部分操作提供额外的打印提示。默认值：`false`。

*返回值：*

无 （构建函数）


***
### LoadIndex

```c++
void LoadIndex(std::string indexPath, const NpuIndexVStar *loadedIndex = nullptr);
```
索引导入接口。

*入参：*

* `indexPath`: C++11字符串。加载索引的路径。限制：该路径（绝对或相对）必须指向磁盘内一个有效的二进制文件，且该文件的内容必须为有效的索引（内部会进行检测）。
* `loadedIndex`: 仅在`MultiIndex`场景下使用。第2-n个索引通过这个参数,传入第一个已经填充好的索引。限制：在`MultiIndex`场景下不能为空指针，在SingleIndex场景下必须为空指针。若在SingleIndex场景下使用这个指针加载索引，加载的码本将被这个指针指向的`NpuIndexVstar`实例的码本替代。

*返回值：*

无

***

### WriteIndex

```c++
void WriteIndex(const std::string& indexPath);
```
索引写入磁盘接口。

*入参：*

* `indexPath`: C++11字符串。写入`KMode`索引的路径。限制：该路径（绝对或相对）必须是一个可写的路径（内部会进行检测）。

*返回值：*

无

***

### AddCodeBooks
```c++
APP_ERROR AddCodeBooks(const std::string& codeBooksPath);
```
将码本加入索引的接口。

*额外说明：*
1. 训练码本的脚本可见`train/train.py`。

*入参：*

* `codeBooksPath`: C++11字符串。加载码本的路径。限制：该路径（绝对或相对）必须指向磁盘内一个有效的二进制文件，且该文件的内容必须为有效的码本（L1码本+L2码本）（内部会进行检测）。


*返回值：*

APP_ERROR预定义宏，详见 ...

***

### AddCodeBooks (MultiIndex)
```c++
APP_ERROR AddCodeBooks(const NpuIndexVStar *loadedIndex = nullptr);
```
`MultiIndex`场景下，第2-n个索引通过这个参数, 传入第一个已经填充好的`NpuIndexVstar`实例，从而将该指针指向的`NpuIndexVstar`实例的码本载入当前索引。

*额外说明：*
1. 训练码本的脚本可见`train/train.py`。

*入参：*

* `loadedIndex`: 仅在`MultiIndex`场景下使用。第2-n个索引通过这个参数,传入第一个已经填充好的索引。限制：在MultiIndex场景下不能为空指针，在`SingleIndex`场景下必须为空指针。若在`SingleIndex`场景下使用这个指针加载索引，加载的码本将被这个指针指向的`NpuIndexVstar`实例的码本替代。

*返回值：*

APP_ERROR预定义宏，详见 ...

***

### AddVectors

```c++
APP_ERROR AddVectors(const std::vector<float> &baseRawData);
```
将底库数据添加进入索引。

*入参:*
* `baseData`: 包含底库数据的`vector<float>`容器。限制：不能为空，长度必须为`nb * dim`，`nb`为准备添加进入底库内部的向量数量, `dim`为每个向量的维度。

*返回值：*

APP_ERROR预定义宏，详见 ...

***

### AddVectorsWithIds

```c++
APP_ERROR AddVectorsWithIds(const std::vector<float> &baseRawData, const std::vector<uint64_t>& ids);
```
在添加底库的基础上，允许用户输入一个自定义数组，将检索回的底库向量id映射到这个数组定义的id上。

*额外说明：*

* 在将一份底库切割并创建多个索引 (`MultiIndex`) 的场合，索引会将每段底库数据从0开始以此赋予id（底库向量在外侧的绝对id被转换成该向量在这个索引内的相对id），并在此基础上进行检索并返回`labels`，这个接口旨在使用户能将`labels`返回的相对id通过`ids`参数转换回绝对`id`，使得`labels`内的结果能参与`gt (ground truth)`的精度校验（因为gt内的底库id一般都为绝对id）。
例如，`ids = [32, 17, 64 ...]`, `labels`在内部的返回值为 `[0, 1, 2 ...]`, 那在用户获得最后的`labels`之前，`labels`内的值会被映射成`[32, 17, 64 ...]`。

* 注意：`AddVectorsWithIds`与`MultiSearch`场景不兼容。意思是，`MultiSearch`场景下每个单独index如果使用了`AddVectorsWithIds`接口去覆盖所添加的底库向量的id，所使用的`ids`向量的内容不会覆盖最终`MultiSearch`出的`labels`的结果。



*入参:*
* `baseData`: 包含底库数据的`vector<float>`容器。限制：不能为空，长度必须为`nb * dim`，`nb`为准备添加进入底库内部的向量数量, `dim`为每个向量的维度。
* `ids`: 包含准备映射到`baseData`的id的`vector<uint64_t>`容器。限制：内部数据量必须大于等于`baseData.size()/dim`（每个传入的底库向量都必须得到映射）。


### DeleteVectors 1
```c++
APP_ERROR DeleteVectors(const std::vector<int64_t> &ids);
```
删除底库中的向量。

*入参:*
* `ids`: 包含待删除的底库数据的向量id的vector。限制：不能为空。

*返回值：*

APP_ERROR预定义宏，详见 ...

### DeleteVectors 2
```c++
APP_ERROR DeleteVectors(const int64_t &id);
```
删除底库中的向量。

*入参:*
* `id`: 一个待删除的底库向量的id。限制：无。

*返回值：*

APP_ERROR预定义宏，详见 ...

***
### DeleteVectors 3
```c++
APP_ERROR DeleteVectors(const int64_t &startId, const int64_t &endId);
```
删除底库中的向量，删除`[startId, endId]`内所有的底库id。

*入参:*
* `startId`: 起始底库id。限制：无。
* `endId`: 截至底库id。限制：必须大于startId。

*返回值：*

APP_ERROR预定义宏，详见 ...
### Search
```c++
APP_ERROR Search(const SearchImplParams &params);
```
向量检索接口。

*入参：*

* `params`：一个`SearchImplParams`实例。定义如下：

```c++
struct SearchImplParams {
    SearchImplParams(size_t n, std::vector<float> &queryData, int topK, std::vector<float> &dists, std::vector<int64_t> &labels)
        :n(n), queryData(queryData), topK(topK), dists(dists), labels(labels) {}

    size_t n = 0;
    const std::vector<float> &queryData;
    int topK = 100;
    std::vector<float> &dists;
    std::vector<int64_t> &labels;
};
```
* `n`: 检索的batch大小。限制：`0 < n <= nq`；
* `queryData`: 填充着待检索数据的`vector`容器。限制：指向的数据长度必须大于`nq * dim`；
* `topK`: 从底库中寻找 `topK` 个与 `query`相似的向量。限制：`0 < topK < nb`;
* `dists`：空的`vector`容器，将装载检索返回的底库向量距离每个query的dists。对于每个`query`， 存储其每个返回的`topK`底库向量之间的L2距离。限制：容量必须大于`nq * topK`;
* `labels`: 空的`vector`容器，将装载检索返回的底库向量的labels(id)。对于每个`query`， 存储其每个返回的`topK`底库向量的id。限制：容量必须大于`nq * topK`。

*返回值：*

APP_ERROR预定义宏，详见 ...

***
### Search2 (SearchWithMask)
```c++
    APP_ERROR Search(const SearchImplParams &params, const std::vector<uint8_t> &mask);
```
在检索接口的基础上，添加了使用户可以输入一个uint8指针指向的数组来掩掉特定底库id的功能。

*入参：*
* `params`：一个`SearchImplParams`实例；
* `mask`: 一个填充uin8数据的`vector`容器。对于每个uint8数据，将其转化为二进制表达形式，再转化该形式转化为小端序（代表1的bit在最左边）后, 对于为0的bit，在检索过程中无视拥有与该bit位位置对应的id的底库向量。

    限制：
    
    1. `mask`包含的uint8数据长度必须大于等于 `ceil(nb/8)`（i.e. 我们提供覆盖足够覆盖整个底库的`mask`量，每个query复用同一个`mask`）。
    
    2. 例如, 如果我们在底库id为8-15的底库向量中，想仅保留id == 8的底库向量，那 可得知`mask[1] = 1`，因为：
        ```
            1）8%8 = 0 => mask[1] 包含底库id == 8的mask的对应bit；
            2）若mask[1] == 1, 那 1 in uint8 format = 00000001
            3）转化为"代表1的bit在最左边"的形式 => 10000000，符合预期（掩掉除id == 8的底库向量以外的所有向量）
        ```

*返回值：*

APP_ERROR预定义宏，详见 ...

***
### MultiSearch
```c++
APP_ERROR MultiSearch(std::vector<NpuIndexVStar *> &indexes, const SearchImplParams &params, bool merge);
```
`MultiIndex`场景下，创建一个包含多个`IndexVstar`实例的vector容器，并使用这个vector第一个索引通过调用此接口进行检索。具体使用请见测试用例。

*入参：*
* `indexes`: 包含多个`IndexVstar`实例的vector容器，长度为`multiIndexNum`。限制：`multiIndexNum > 1`。具体能创建的`multiIndexNum`最大数值取决于当前使用的产品芯片最大容量和切割的底库分片大小。

* `params`：一个`SearchImplParams`实例；

* `merge`: 布尔值，决定是否将检索回的结果合并。若不合并，检索回的结果总长度为`nq * multiIndexNum * dim`; 若合并，检索回的结果总长度为`nq * dim`。

*返回值：*

APP_ERROR预定义宏，详见 ...

***
### MultiSearch2 (MultiSearchWithMask)
```c++
APP_ERROR MultiSearch(std::vector<NpuIndexVStar *> &indexes,
    const SearchImplParams &params, const std::vector<uint8_t> &mask, bool merge);
```
`MultiIndex`场景下，创建一个包含多个`IndexVstar`实例的vector容器，并使用这个vector第一个索引通过调用此接口进行检索, 并添加了使用户可以输入一个uint8指针指向的数组来掩掉特定底库id的功能。

*入参：*
* `indexes`: 包含多个`IndexVstar`实例的vector容器，长度为`multiIndexNum`。限制：`multiIndexNum > 1`。具体能创建的`multiIndexNum`最大数值取决于当前使用的产品芯片最大容量和切割的底库分片大小。

* `params`：一个`SearchImplParams`实例；

* `mask`: 一个填充uin8数据的`vector`容器。对于每个uint8数据，将其转化为二进制表达形式，再转化该形式转化为小端序（代表1的bit在最左边）后, 对于为0的bit，在检索过程中无视拥有与该bit位位置对应的id的底库向量。具体请见上述对mask的描述。`mask`的长度必须大于`(nb_max + 7)/8`，其中，`nb_max`为`indexes`内的所有`index`中`nTotal`（所含底库数量）最大的，i.e.我们需要确保传入的`mask`足够覆盖所有`index`的所有底库向量。

* `merge`: 布尔值，决定是否将检索回的结果合并。若不合并，检索回的结果总长度为`nq * multiIndexNum * dim`; 若合并，检索回的结果总长度为`nq * dim`。

*返回值：*

APP_ERROR预定义宏，详见 ...

***

### GetDim
```c++
int GetDim() const;
```
获得初始化索引时的维度的接口。

*入参：*

无

*返回值：*
类型为`int`的初始化索引时的维度。
***
### GetNTotal
```c++
uint64_t GetNTotal() const;
```
获得添加底库数据时输入的底库大小的接口。

*入参：*

无

*返回值：*

类型为`uint64_t`的初始化索引时的维度。若尚未添加底库，则返回0。
***
### SetSearchParams
```c++
void SetSearchParams(const IndexVstarSearchParams &params);
```
根据`IndexVstarSearchParams`结构体修改向量检索时的超参。

*入参：*

`IndexVstarSearchParams`自定义结构体。该结构体的构造函数为：

```c++
struct IndexVstarSearchParams {
    IndexVstarSearchParams(int nProbeL1, int nProbeL2, int l3SegmentNum)
    {
        params.nProbeL1 = nProbeL1;
        params.nProbeL2 = nProbeL2;
        params.l3SegmentNum = l3SegmentNum;
    }
    SearchParams params;
};

```
创建`SearchParams`结构体的参数限制如下：
* `nProbeL1`: 一阶段检索搜索的聚类数。限制：`16 < nProbeL1 <= nlistL1; nProbeL1 % 8 == 0`；
* `nProbeL2`: 二阶段检索搜索的聚类数。限制：`16 < nProbeL2 <= nProbeL1 * nlistL2; nProbeL2 % 8 == 0`；
* `l3Segment`: 三阶段检索的`Segment`数量。限制：`100 < l3Segment <= 5000;l3Segment % 8 == 0`；

***
### GetSearchParams
```c++
IndexVstarSearchParams GetSearchParams() const;
```
返回向量检索时的超参。

*入参：*

无。

*返回值：*
1个`IndexVstarSearchParams`类别的实例。

***
### Reset
```c++
void Reset();
```
重置索引接口。将保存的图索引数据清除，但保留用户初始化索引时输入的参数， 其中包括
* `dim`
* `nlistL1`
* `nlistL2`
* `subspaceDimL1`
*  `subspaceDimL2`
* `deviceList`
* `verbose`

如果用户使用`构造函数2`(使用无参数构造函数)，则这些值变为初始化值 (`0`)。

*入参：*

无

*返回值：*

无