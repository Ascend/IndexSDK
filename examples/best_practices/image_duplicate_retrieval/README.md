# 媒资平台亿级图片版权治理：向量检索算法选型最佳实践

Pinterest 在论文 [Evolution of a Web-Scale Near Duplicate Image Detection System](https://arxiv.org/abs/2209.08433) 中介绍了图片版权治理面临的实际问题：同一图片经过裁剪、翻转、缩放或旋转后，会形成像素不同的近重复副本。本文据此设定媒资平台版权治理场景，使用 IndexSDK 将每天新增图片与海量历史图片比较，召回 Top-100 疑似副本供后续模型或人工审核，并提供算法选型测试和 100M 业务 Demo。

## 1. 业务规模与验收目标

假设平台已运营 5 年，经治理和去重后平均每天有 5 万张图片进入历史底库，则底库规模为：

$$
N_{base}=50,000\times365\times5=91,250,000
$$

本实践使用 100M 条历史向量。图片被嵌入为 128 维向量；底库和查询均做 L2 归一化。

版权治理入口每天处理 1M 张新增或转载候选图片，24 小时内完成所需的最低平均吞吐为：

$$
QPS_{min}=\frac{1,000,000}{24\times60\times60}=11.57
$$

选型的指标是：

1. 在 100M 底库上，Recall@100 不低于 95%；
2. 系统 QPS 不低于 11.57；
3. 系统尽可能用更少的资源达到更高的吞吐；

Recall@100 使用 CPU 精确 Top-100 作为 ground truth：

$$
\mathrm{Recall@100}=\frac{1}{Q}\sum_{i=1}^{Q}\frac{|R_i\cap G_i|}{100}
$$

其中 `R_i` 是 IndexSDK 的 Top-100 结果，`G_i` 是对应的精确 Top-100。

## 2. 候选算法介绍

[IndexSDK 硬件支持矩阵](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/01_introduction.md#支持的硬件和操作系统)列出的 Atlas A2 推理系列产品 检索算法还包括带属性过滤能力的 TS 系列。本场景没有时间、空间或附加属性过滤条件，因此比较以下四种基础算法。

| 算法 | 检索方式 | 主要取舍 |
|---|---|---|
| Flat | 全量扫描，不训练；本 Demo 使用 FP32 接口输入，设备索引以 FP16 保存 | 精度高、使用简单，计算量随底库增长 |
| Int8Flat | 本 Demo 在 Host 侧对每个分量按固定比例量化为 Int8，再由 Int8Flat 全量扫描 | 理论向量载荷较小，但有量化误差 |
| IVFFlat | 先训练倒排桶，查询只扫描 `nprobe` 个桶，桶内保存 FP32 向量 | 可减少距离计算，但需要训练和调参 |
| IVFRaBitQ | IVF 粗筛后用 RaBitQ 编码检索；本测试配置为粗排 Top-800 后在 Host 侧用 FP32 原始向量精排 | 理论索引载荷最小，但召回依赖粗筛和量化参数 |

Flat 和 Int8Flat 都是全量扫描，其单条查询时间可写为：

$$
T_{\mathrm{flat}}(N)=A+B\times N\times d
$$

IVF 系列还包括聚类中心比较和桶扫描。对查询 `q`，令 `P(q)` 为选中的 `nprobe` 个桶，`L_l` 为第 `l` 个桶，则实际扫描量为：

$$
T_{\mathrm{IVF}}(N)\approx A_{\mathrm{IVF}}+B_1\times n_{\mathrm{list}}\times d+B_2\times d\sum_{l\in\mathcal{P}(q)}|L_l|
$$

桶规模近似均衡时：

$$
\sum_{l\in\mathcal{P}(q)}|L_l|\approx N\times\frac{n_{\mathrm{probe}}}{n_{\mathrm{list}}}
$$

固定 `d`、`nlist` 和 `nprobe` 时 IVF 类的算法也是线性的。

综上，两类模型都可整理为 `T(N)=A+B×N`。因此，在本实验参数保持不变时，四种算法的检索时间开销都随底库规模 `N` 近似线性增长。

## 3. 模型选型实验设计

### 3.1 参数设置

测试环境为每张卡 64 GiB HBM 的 Ascend 910B3。BigANN SIFT 原始数据是 UInt8，本 Demo 先转为 FP32 并做 L2 归一化，供 Flat、IVFFlat 和 IVFRaBitQ 使用；Int8Flat 再将同一份归一化 FP32 数据量化为有符号 Int8。测试规模固定为 1M、10M、30M、40M、70M 和 100M。

参数只在 1M 底库上手动调整，直到 Recall@100 不低于 95%；随后冻结参数，只改变底库规模 `N`。四种算法共用 `TopK=100`、`resourceSize=2 GiB`、10K 查询和 `batch=64`。

| 算法 | 固定参数 |
|---|---|
| Flat | IP |
| Int8Flat | Int8 cosine，Host `scale=256`，`blockSize=16384` |
| IVFFlat | IP，`nlist=1024`，`nprobe=48`，100K 训练向量 |
| IVFRaBitQ | L2，`nlist=1024`，`nprobe=48`，`refineAlpha=8`，开启 refine，100K 训练向量 |

Int8Flat 采用逐分量、对称、固定比例的均匀量化，在 Host 侧量化底库和查询，再交给 `AscendIndexInt8Flat` 检索：

$$
q_i=\operatorname{clip}\left(\operatorname{round}(256x_i),-127,127\right)
$$

IVFRaBitQ 配置 L2 距离；归一化向量的 L2 与内积排序一致：

$$
\lVert x-y\rVert_2^2=2-2x^{\mathsf T}y
$$

本测试开启 refine：设备侧粗排返回 `refineAlpha×TopK=800` 个候选 ID，Host 侧读取对应 FP32 原始向量做 L2 精排，最终返回 Top-100。

性能测试不计下载、数据归一化、底库量化、训练和建库时间。每个规模先搜索一次预热，再对第二次完整搜索计时；Int8Flat 的查询量化在计时区间内。100M 业务 Demo 只搜索一次，不做预热。

### 3.2 QPS 与 Recall

每个单元格依次为 `QPS / R@100 / 卡数`，卡数是该次运行实际使用的 NPU 数量。

| 底库规模 | Flat | Int8Flat | IVFFlat | IVFRaBitQ |
|---:|---:|---:|---:|---:|
| 1M | 25568.6 / 0.9927 / 1 | 17583.2 / 0.9760 / 1 | 207.2 / 0.9705 / 1 | 4827.9 / 0.9559 / 1 |
| 10M | 6781.8 / 0.9896 / 1 | 2770.8 / 0.9686 / 1 | 197.9 / 0.9870 / 1 | 2864.6 / 0.9289 / 1 |
| 30M | 2539.5 / 0.9881 / 1 | 995.7 / 0.9647 / 1 | 176.2 / 0.9909 / 1 | 1659.2 / 0.9011 / 1 |
| 40M | 1948.5 / 0.9874 / 1 | 756.2 / 0.9636 / 1 | 167.4 / 0.9917 / 1 | 1297.9 / 0.8924 / 1 |
| 70M | 1118.2 / 0.9866 / 1 | 438.4 / 0.9613 / 1 | 79.3 / 0.9932 / 1 | 916.2 / 0.8732 / 1 |
| 100M | 801.3 / 0.9860 / 1 | 304.5 / 0.9598 / 1 | 154.8 / 0.9939 / 2 | 685.7 / 0.8590 / 1 |

为直观展示吞吐，下表给出 `batch=64` 检索中每条查询的平均摊销耗时，即 `1000000/QPS`；它不等同于单条查询独立调用的端到端延迟。

| 底库规模 | Flat | Int8Flat | IVFFlat | IVFRaBitQ |
|---:|---:|---:|---:|---:|
| 1M | 39.1 us | 56.9 us | 4826.3 us | 207.1 us |
| 10M | 147.5 us | 360.9 us | 5053.1 us | 349.1 us |
| 30M | 393.8 us | 1004.3 us | 5675.4 us | 602.7 us |
| 40M | 513.2 us | 1322.4 us | 5973.7 us | 770.5 us |
| 70M | 894.3 us | 2281.0 us | 12610.3 us | 1091.5 us |
| 100M | 1248.0 us | 3284.1 us | 6459.9 us | 1458.4 us |

Flat 的平均摊销耗时从 1M 时的 39.1 us 增至 100M 时的 1248.0 us。Flat 会扫描全部底库，因此耗时随 `N` 增长；小规模时固定启动开销占比较高，所以底库扩大 100 倍时，耗时不会恰好扩大 100 倍。

Int8Flat 的平均摊销耗时从 56.9 us 增至 3284.1 us，变化趋势同样接近线性。这符合全量扫描的实现：量化缩小了向量元素，但当前路径还需要量化查询、处理向量范数，并以较小的数据块执行搜索，因此量化并未转化为更高的实测 QPS。

IVFFlat 在单卡 1M–40M 间的平均摊销耗时只从 4826.3 us 增至 5973.7 us，说明这一区间主要受固定开销影响。原因是 IVFFlat 选桶后，还要在 Host 与设备之间传递选桶结果，并逐条查询组织桶内搜索；这些调度和同步工作不会随小规模底库明显变化。70M 时桶内数据增多，耗时升至 12610.3 us。100M 改用两卡分片后降至 6459.9 us，但卡数已经不同，不能与单卡数据直接比较。

IVFRaBitQ 的平均摊销耗时从 207.1 us 增至 1458.4 us，但 Recall@100 从 0.9559 降至 0.8590，并从 10M 开始低于 95%。固定 `nprobe=48` 和 800 个粗排候选后，底库越大，每个桶中的向量通常越多；真实近邻可能没有进入探测桶，也可能没有进入粗排 Top-800，这两种情况都无法由 Host 精排找回。因此该配置虽然耗时增长较缓，但召回率不满足业务要求。

### 3.3 HBM 占用分布

本实验使用 `aclrtGetMemInfo` 采样测试程序完整执行过程中的整卡 HBM 占用，并减去索引构造前的 baseline。

Flat 和 Int8Flat 每 5 ms 采样，IVFFlat 每 100 ms 采样，IVFRaBitQ 每 20 ms 采样。P80 表示 80% 的采样值不超过该值，P100 为观测峰值；多卡结果按同一采样时刻求和。

每个单元格依次为 `P70 / P80 / P90 / P99 / P100`，单位均为 GiB；卡数沿用性能表。

| 底库规模 | Flat | Int8Flat | IVFFlat | IVFRaBitQ |
|---:|---:|---:|---:|---:|
| 1M | 2.28 / 2.28 / 2.28 / 2.29 / 2.29 | 2.30 / 2.30 / 2.30 / 2.30 / 2.30 | 7.55 / 7.55 / 7.55 / 7.55 / 7.55 | 2.32 / 2.74 / 2.74 / 2.74 / 2.74 |
| 10M | 4.50 / 4.50 / 4.50 / 4.50 / 4.54 | 4.46 / 4.46 / 4.46 / 4.46 / 4.46 | 13.15 / 13.15 / 13.15 / 13.15 / 13.15 | 2.74 / 2.75 / 3.00 / 3.00 / 3.42 |
| 30M | 9.43 / 9.43 / 9.43 / 9.43 / 9.43 | 9.27 / 9.27 / 9.27 / 9.27 / 9.27 | 25.84 / 25.84 / 25.84 / 25.84 / 25.84 | 3.00 / 3.50 / 3.90 / 3.91 / 4.50 |
| 40M | 11.91 / 11.91 / 11.91 / 11.91 / 11.92 | 11.67 / 11.67 / 11.67 / 11.67 / 11.67 | 35.28 / 35.28 / 35.28 / 35.28 / 35.28 | 4.16 / 4.36 / 4.36 / 4.36 / 5.15 |
| 70M | 19.29 / 19.29 / 19.29 / 19.29 / 19.33 | 18.89 / 18.89 / 18.89 / 18.89 / 18.89 | 57.61 / 57.62 / 57.62 / 57.62 / 57.62 | 4.61 / 5.36 / 5.74 / 5.76 / 6.93 |
| 100M | 26.67 / 26.67 / 26.67 / 26.67 / 26.67 | 26.10 / 26.10 / 26.10 / 26.11 / 26.11 | 4.08 / 59.51 / 59.51 / 64.52 / 64.52 | 6.26 / 6.76 / 7.15 / 7.17 / 8.75 |

IVFFlat 的 100M 数据使用两卡，表中的 64.52 GiB 是两卡 HBM 增量之和；单卡峰值为 32.26 GiB。

下面估算的是 IndexSDK 核心持久结构的主索引载荷，不包含内存对齐、资源池、搜索工作区和运行时分配，因此需要与实测峰值结合使用。业务图片 ID 通过 Host 侧映射，不计入 HBM；IVF 公式中的 8-byte 字段是设备倒排表里的 IndexSDK `idx_t`。

Flat 在设备侧以 FP16 保存向量，每个分量占 2 byte：

$$
M_{\mathrm{Flat}}=\frac{N\times d\times2}{2^{30}}=23.84\text{ GiB}
$$

Int8Flat 为每条向量保存 128-byte Int8 编码和 2-byte FP16 范数：

$$
M_{\mathrm{Int8Flat}}=\frac{N\times(d+2)}{2^{30}}=12.11\text{ GiB}
$$

IVFFlat 为每条向量保存 128 维 FP32 数据和 8-byte `idx_t`：

$$
M_{\mathrm{IVFFlat,main}}=\frac{N\times(4d+8)}{2^{30}}=48.43\text{ GiB}
$$

IVFRaBitQ 为每条向量保存 16-byte 二进制编码、两个 4-byte 辅助量和 8-byte `idx_t`；较小的 `nlist` 相关结构、正交矩阵和 LUT 不计入主项：

$$
M_{\mathrm{IVFRaBitQ,main}}=\frac{N\times(16+4+4+8)}{2^{30}}=2.98\text{ GiB}
$$

理论主索引载荷、实际卡数与单卡实测峰值对照如下。

| 算法 | 100M 主索引总载荷/GiB | 卡数 | 单卡 HBM 增量峰值/GiB |
|---|---:|---:|---:|
| Flat | 23.84 | 1 | 26.67 |
| Int8Flat | 12.11 | 1 | 26.11 |
| IVFFlat | 48.43 | 2 | 32.26 |
| IVFRaBitQ | 2.98 | 1 | 8.75 |

Int8Flat 理论载荷仅 12.11 GiB，实测却达到 26.11 GiB。本环境的内存分配追踪显示，当前实现把数据拆成 6104 个 Int8 和范数小块；受 32 B 内存偏移与设备页对齐影响，逻辑上约 2 MiB 的 Int8 块实际约占 4 MiB。所有分块合计约 24.05 GiB，再加 2 GiB 资源池后约为 26.05 GiB，与实测一致。差额来自当前环境的分块分配，并非另外保存了一份 FP16 底库。

IVFFlat 在 70M 时的单卡 HBM 增量峰值已经达到 57.62 GiB，加上约 3.4 GiB baseline 后只剩约 2.98 GiB；从 70M 增至 100M，仅新增主索引就需要：

$$
\Delta M_{70M\rightarrow100M}=\frac{30,000,000\times(4\times128+8)}{2^{30}}=14.53\text{ GiB}
$$

新增载荷明显超过单卡余量，因此本配置在当前环境下使用两卡分片完成 100M 测试。搜索时每张卡还会分配约 5.00 GiB 的 FP32 `disVec` 临时工作区；该值由 `batch=64`、每个计算核处理 2 组分段、40 个计算核和每段最多 262,144 条向量共同决定：

$$
M_{\mathrm{workspace/card}}=\frac{64\times2\times40\times262144\times4}{2^{30}}=5.00\text{ GiB}
$$

两卡 HBM 增量峰值合计为 64.52 GiB，即每卡 32.26 GiB，其中已经包含每卡约 5.00 GiB 的搜索工作区和运行时分配。IVFRaBitQ 的 100M HBM 峰值为 8.75 GiB，是四种算法中最低的。

## 4. 100M 图片版权治理业务的选型与部署结论

结论是：对于 100M 条 128 维归一化图片向量，本场景推荐单卡 Flat，不需要多卡；

Flat 在 100M 底库上的 Recall@100 为 0.9860、QPS 为 801.3，均超过业务门槛；HBM 增量峰值为 26.67 GiB，加上约 3.4 GiB baseline 后仍可由单张 64 GiB 卡容纳：

$$
M_{card}=3.4+26.67=30.07\text{ GiB}<64\text{ GiB}
$$

按该吞吐处理每天 1M 条查询只需约 20.8 分钟：

$$
t_{day}=\frac{1,000,000}{801.3}=1,248\text{ s}\approx20.8\text{ min}<24\text{ h}
$$

Int8Flat 虽然 Recall@100 达到 0.9598，但 QPS 仅为 304.5，实测 HBM 又与 Flat 接近，没有体现量化的容量优势。IVFFlat 的 Recall@100 为 0.9939，但需要两卡且 QPS 仅为 154.8。IVFRaBitQ 的 HBM 最低，但 Recall@100 只有 0.8590，若要 95% Recall@100，则 QPS 会更低，低于 Flat。因此三者均不如单卡 Flat 适合本场景。

实测业务 Demo 使用同样的 100M 底库规模，实际搜索完整 1M 条查询且不预热，并对全部查询计算 Recall@100。实测 Recall@100 为 1.0000，search-only QPS 为 824.0，搜索耗时约 20.2 分钟；完整 `run_business.sh` 耗时 21 分 23.81 秒，Max RSS（host 峰值内存占用）约 2.98 GiB。

## 5. Demo 与复现

运行前请按照 IndexSDK [安装部署](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/04_installation_guide.md)完成环境配置，并确保 Python 可以导入 NumPy 和 Faiss。本 Demo 包含 IVFRaBitQ 算法，需要 Faiss 1.14.1，运行前请先切换软链接：

```bash
ln -sf /usr/local/faiss1.14.1 /usr/local/faiss
```

单卡测试使用 0 号设备，IVFFlat 100M 双卡测试使用 0、1 号设备；运行前应确保对应设备空闲。以下命令均在本 Demo 根目录执行。

```bash
# 下载 SIFT100M 原始数据
./scripts/download_sift100m.sh

# 生成归一化后的 FP32、Int8 数据和各规模精确 ground truth
python3 ./python/prepare_bench_data.py
python3 ./python/generate_bench_ground_truth.py

# 生成图片治理业务 100M Demo 的输入和精确 ground truth
python3 ./python/generate_business_data.py

# 生成本 Demo 使用的算子模型
./scripts/generate_models.sh

# 编译 C++ 程序
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 运行四种算法的扩展性测试并调用 Python 验证（IVFFlat 含单、双卡变体）
./run_bench.sh

# 运行推荐的单卡 Flat 100M 业务检索并调用 Python 验证
./run_business.sh
```
