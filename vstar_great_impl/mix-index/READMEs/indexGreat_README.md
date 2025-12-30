# IndexGreat算法使用说明

## 什么是IndexGreat算法？
IndexGreat集成了我司自研的晟腾侧和鲲鹏侧向量检索算法,为用户提供鲲鹏+昇腾高维大底库近似检索能力。

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

## 如何使用IndexGreat算法：

### 物理硬件限制：

* ARM/x86 64bit操作系统，CPU数量20+以上, 编译器版本GCC7以上；

* #### 如果使用`AKMode`:
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
3. `索引创建`阶段，底库大小`nb`需要满足`10000 <= nb`，不建议`nb > 1e8`；
****

### 使用前：

引入相关头文件:
```c++
#include "npu/IndexGreat.h"
```

## 接口总览

### 构造函数1
```c++
explicit IndexGreat(std::string mode, const IndexGreatInitParams &params);
```
IndexGreat的构造函数。

*入参：*

* `mode`: C++11字符串。限制范围：必须为`"AKMode"` 或 `"KMode"`中的一个。
* `params`: 自定义索引构建结构体`IndexGreatInitParams`实例。格式如下：
```c++
    struct IndexGreatInitParams {
        IndexGreatInitParams()
        {
        }
        // AKMode
        IndexGreatInitParams(const std::string mode, const IndexVstarInitParams &AInitParams,
                            const KModeInitParams &KInitParams, bool verbose = false)
        {
            ASCENDSEARCH_THROW_IF_NOT_MSG(AInitParams.dim == KInitParams.dim,
                                        "IndexGreatInitParams: dim mismatched between AMode and KMode.\n");
            this->mode = mode;
            this->AInitParams = AInitParams;
            this->KInitParams = KInitParams;
            this->verbose = verbose;
        }
        // KMode
        IndexGreatInitParams(const std::string mode, const KModeInitParams &KInitParams, bool verbose = false)
        {
            this->mode = mode;
            this->KInitParams = KInitParams;
            this->verbose = verbose;
        }
        std::string mode = "AKMode";
        IndexVstarInitParams AInitParams;
        KModeInitParams KInitParams;
        bool verbose = false;
    };
```
该结构体的传参如下：
* `mode`: C++11字符串。限制范围：必须为`"AKMode"` 或 `"KMode"`中的一个, 且必须与`IndexGreat`实例初始化的`mode`参数一致。
* `verbose`: 布尔值，指定是否开启`verbose`选项，开启后部分操作提供额外的打印提示。默认值：`false`。
* `AInitParams`: `IndexVstarInitParams`结构体实例，具体参数可见该目录下的`./indexVstar_README.md`文件。
    #### 注意：仅针对`AKMode`下的创建的`Vstar`索引，在`AInitParams`内传入的`verbose`的值会覆盖掉`IndexGreatInitParams`内的`verbose`值。（`AKMode`下的`KMode`仍遵循`IndexGreatInitParams`设置的`verbose`值）。
* `KInitParams`: `KModeInitParams`结构体实例，其定义如下：
```c++
    struct KModeInitParams {
        KModeInitParams()
        {
        }
        KModeInitParams(int dim, int R, int convPQM, int evaluationType, int efConstruction)
            : dim(dim), R(R), convPQM(convPQM), evaluationType(evaluationType), efConstruction(efConstruction)
        {
        }
        int dim = 256;
        int R = 50;
        int convPQM = 128;
        int evaluationType = 0;
        int efConstruction = 300;
    };
```
* `dim`: 输入数据维度。默认值：256。 限制范围：必须在`[128, 256, 512, 1024]`之内。
* `R`: DegRee, 构件图的精细度。默认值：50。限制范围：`40 < R < 60`。
* `convPQM`: PQ量化向量分段数。默认值：128。限制范围：1）`16 <= convPQM <= dim`； 2）`convPQM % 8 == 0`；3） `convPQM % dim == 0`。
* `evaluationType`: 距离计算标准。默认值：0 (IP距离)。限制范围：`0`（IP距离）或`1`（L2距离）。

*返回值：*

无 （构建函数）

***
### 构造函数2

```c++
explicit IndexGreat(const std::string mode, const std::vector<int>& deviceList, const bool verbose);
```
IndexGreat的构造函数，仅用于未知输入数据维度和各种构建超参的情况。使用此构造函数创建的`IndexGreat`实例不能进行`LoadIndex`以外的其他操作。

*入参：*

* `mode`: C++11字符串。限制范围：必须为`"AKMode"` 或 `"KMode"`中的一个。
* `deviceList`: ``AKMode``下``AMode``使用的包裹着NPU卡physical ID的vector容器。默认值：{0}。限制范围：内部数字 $\geq$ 0，具体范围请使用`npu-smi`命令查询对应的NPU卡physical ID。
* `verbose`: 布尔值，指定是否开启`verbose`选项，开启后部分操作提供额外的打印提示。默认值：`false`。

*返回值：*

无 （构建函数）


***
### LoadIndex (KMode)

```c++
void LoadIndex(const std::string& indexPath);
```
`KMode`模式下的索引导入接口。

*入参：*

* `indexPath`: C++11字符串。加载`KMode`索引的路径。限制：该路径（绝对或相对）必须指向磁盘内一个有效的二进制文件，且该文件的内容必须为有效的`KMode`索引（内部会进行检测）。

*返回值：*

无

***
### LoadIndex (AKMode)
```c++
void LoadIndex(const std::string& AModeindexPath, const std::string& KModeindexPath);
```
`AKMode`模式下的索引导入接口。

*入参：*

* `AModeindexPath`: C++11字符串。加载`AMode`索引的路径。限制：该路径（绝对或相对）必须指向磁盘内一个有效的二进制文件，且该文件的内容必须为有效的`AMode`索引（内部会进行检测）。
* `KModeindexPath`: C++11字符串。加载`KMode`索引的路径。限制：该路径（绝对或相对）必须指向磁盘内一个有效的二进制文件，且该文件的内容必须为有效的`KMode`索引（内部会进行检测）。

*返回值：*

无

***

### WriteIndex(KMode)

```c++
void WriteIndex(const std::string& indexPath);
```
`KMode`模式下的索引写入磁盘接口。

*入参：*

* `indexPath`: C++11字符串。写入`KMode`索引的路径。限制：该路径（绝对或相对）必须是一个可写的路径（内部会进行检测）。

*返回值：*

无

***

### WriteIndex (AKMode)
```c++
void WriteIndex(const std::string& AModeindexPath, const std::string& KModeindexPath);
```
`AKMode`模式下的索引导入接口。

*入参：*

* `AModeindexPath`: C++11字符串。写入`AMode`索引的路径。限制：该路径（绝对或相对）必须是一个可写的路径（内部会进行检测）。
* `KModeindexPath`: C++11字符串。写入`KMode`索引的路径。限制：该路径（绝对或相对）必须是一个可写的路径（内部会进行检测）。

*返回值：*

无

***

### AddCodeBooks
```c++
APP_ERROR AddCodeBooks(const std::string& codeBooksPath);
```
`AKMode`下，`AMode`将码本加入索引的接口。

*额外说明：*
1. 训练码本的脚本可见`HSP_CBTraning_final.py`。
2. 该接口仅能在索引初始化为`AKMode`时使用。

*入参：*

* `codeBooksPath`: C++11字符串。加载`AMode`码本的路径。限制：该路径（绝对或相对）必须指向磁盘内一个有效的二进制文件，且该文件的内容必须为有效的`AMode`码本（L1码本+L2码本）（内部会进行检测）。


*返回值：*

APP_ERROR预定义宏，详见 ...

***
### AddVectors

```c++
APP_ERROR AddVectors(const std::vector<float> &baseRawData);
```
将底库数据添加进入索引。

*额外说明：*

1. 若使用`AKMode`，则创建对应的`AMode`索引和`KMode`索引。
2. 若使用`KMode`，则仅创建对应的`KMode`索引。

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

在将一份底库切割并创建多个索引 (`MultiIndex`) 的场合，索引会将每段底库数据从0开始以此赋予id（底库向量在外侧的绝对id被转换成该向量在这个索引内的相对id），并在此基础上进行检索并返回`labels`，这个接口旨在使用户能将`labels`返回的相对id通过`ids`参数转换回绝对`id`，使得`labels`内的结果能参与`gt (ground truth)`的精度校验（因为gt内的底库id一般都为绝对id）。
例如，`ids = [32, 17, 64 ...]`, `labels`在内部的返回值为 `[0, 1, 2 ...]`, 那在用户获得最后的`labels`之前，`labels`内的值会被映射成`[32, 17, 64 ...]`。

* 注意：`AddVectorsWithIds`与`MultiSearch`场景不兼容。意思是，`MultiSearch`场景下每个单独index如果使用了`AddVectorsWithIds`接口去覆盖所添加的底库向量的id，所使用的`ids`向量的内容不会覆盖最终`MultiSearch`出的`labels`的结果。

*入参:*
* `baseData`: 包含底库数据的`vector<float>`容器。限制：不能为空，长度必须为`nb * dim`，`nb`为准备添加进入底库内部的向量数量, `dim`为每个向量的维度。
* `ids`: 包含准备映射到`baseData`的id的`vector<uint64_t>`容器。限制：内部数据量必须大于等于`baseData.size()/dim`（每个传入的底库向量都必须得到映射）。

*返回值：*

APP_ERROR预定义宏，详见 ...

***

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
    APP_ERROR SearchWithMask(const SearchImplParams &params, const std::vector<uint8_t> &mask);
```
在检索接口的基础上，添加了使用户可以输入一个uint8指针指向的数组来掩掉特定底库id的功能。

*入参：*
* `params`：一个`SearchImplParams`实例；
* `mask`: 一个填充uin8数据的`vector`容器。对于每个uint8数据，将其转化为二进制表达形式，再转化该形式转化为小端序（代表1的bit在最左边）后, 对于为0的bit，在检索过程中无视拥有与该bit位位置对应的id的底库向量。

    限制：
    
    1. `mask`包含的uint8数据长度必须大于等于 `ceil(nb/8)`（i.e. 我们提供覆盖足够覆盖整个底库的`mask`量，每个query复用同一个`mask`）。
    
    2. 例如, 如果我们在底库id为8-15的底库向量中，想掩掉id == 8的底库向量，那 可得知`mask[1] = 1`，因为：
        ```
            1）8%8 = 0 => mask[1] 包含底库id == 8的mask的对应比特；
            2）若mask[1] == 1, 那 1 in uint8 format = 00000001
            3）转化为"代表1的bit在最左边"的形式 => 10000000，符合预期（掩掉从左向右数的第八个bit）
        ```

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
void SetSearchParams(const IndexGreatSearchParams &params);
```
根据`IndexGreatSearchParams`结构体修改向量检索时的超参。

*入参：*

`IndexGreatSearchParams`自定义结构体。

```c++
struct IndexGreatSearchParams {
    // AKMode
    IndexGreatSearchParams()
    {
    }
    IndexGreatSearchParams(std::string mode, int nProbeL1, int nProbeL2, int l3SegmentNum, int ef)
        : mode(mode), nProbeL1(nProbeL1), nProbeL2(nProbeL2), l3SegmentNum(l3SegmentNum), ef(ef)
    {
    }

    // KMode
    IndexGreatSearchParams(std::string mode, int ef) : mode(mode), ef(ef)
    {
    }

    std::string mode = "AKMode";
    // AMode search params
    int nProbeL1 = 72;
    int nProbeL2 = 300;
    int l3SegmentNum = 1000;

    // KMode search params
    int ef = 150;
};
```
* `mode`: 取值必须与索引创建时的`mode`参数保持一致。

### GetSearchParams
```c++
IndexGreatSearchParams GetSearchParams() const;
```
返回向量检索时的超参。

*入参：*

无。

*返回值：*
1个`IndexGreatSearchParams`类别的实例，返回当前`mode`对应的索引的所有检索超参。`IndexGreatSearchParams`的具体定义见上。

***

```c++
void Reset();
```
重置索引接口。将对应`mode`内保存的图索引数据清除，但保留用户输入的初始化数据(用户初始化索引时输入的参数).

*入参：*

无

*返回值：*

无
