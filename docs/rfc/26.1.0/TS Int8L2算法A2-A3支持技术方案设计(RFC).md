# TS Int8L2算法A2/A3支持技术方案设计(RFC)

## 1. 背景与范围

Issue #48要求将AscendIndexTS常用算法从310P迁移到Atlas A2/A3。
本方案在社区已有FlatIP和Int8Cos适配的基础上，补齐
`AlgorithmType::FLAT_L2_INT8`。

本次包含A2/A3上的初始化、入库、时间/token过滤检索、底库读取、
UT/ST与性能验证。310P继续使用原有路径。本次不新增L2 ExtraVal能力，
也不修改FlatIP、Int8Cos和Hamming。

## 2. 设计

`TSInt8FlatL2`根据SoC选择执行路径：

- 310P使用原有TBE/Tik算子。
- A2/A3使用`AscendcDistInt8FlatL2`。
- 大batch按已有gear拆分，底库按block和page分段计算。
- 共享mask和逐query mask分别生成对应模型。
- 分段候选结果复用已有AICPU TopK流程。

AscendC算子输入int8 query、Cube布局底库、底库norm和过滤mask，
输出分块距离和块内候选结果。修改包含：

- A2/A3核数对应的flag和op-size布局；
- tail block的实际向量数和DMA尾部空间；
- 共享mask shape；
- dim 256场景的query tile上限64。

底库仍保存signed int8，不增加持久化镜像。支持维度为
`{64, 128, 256, 384, 512, 768, 1024}`。

## 3. 模型生成

`int8flat_generate_model.py`为A2/A3生成L2共享与非共享mask模型：

```bash
python3 int8flat_generate_model.py -d 256 -t 910_9382
```

运行时CANN、自定义OPP和OM必须由同一套环境生成。

## 4. 测试

Mock UT覆盖A2/A3 SoC识别、shared/non-shared过滤、入库和检索。
真实设备ST覆盖：

- CPU L2参考结果与NPU top-k标签、距离对比；
- 全部支持维度；
- batch gear及256、10240拆分；
- topk 1到100000；
- 时间和token过滤。

```bash
cd feature_retrieval/src/ascendfaiss/ut
bash ci_run.sh

MX_INDEX_MODELPATH=/{model_path} MX_INDEX_DEVICE=0 \
  /{build_path}/test/DTascendhost/TestAscendIndexTSA2A3
```

## 5. 性能结果

测试配置为Ascend910_9382单die、CANN 9.0.0、底库1000万、dim 256、
batch 256、top100，warmup 3、测量10次。

| 场景 | 平均延迟 | QPS | 结论 |
| --- | ---: | ---: | --- |
| 无有效时间过滤 | 63.767 ms | 4014.587 | 超过3200 QPS |
| 共享时间过滤 | 77.986 ms | 3282.629 | 超过3200 QPS |

同机FAISS `IndexScalarQuantizer`在相同规模与检索参数下，单线程为
1.63 QPS，已测多线程最优为192.63 QPS。CPU和NPU性能程序的数据生成器
独立，因此该结果仅用于吞吐对标；精度由ST在同一份数据上逐项比较CPU
L2结果，不能用单线程性能结果替代精度验证。

## 6. 验收映射

| 验收项 | 实现 |
| --- | --- |
| A2/A3适配 | SoC分流和AscendC L2距离算子 |
| 原有功能 | 保留310P，覆盖维度、batch、topk和过滤 |
| UT/ST | Mock UT与真机CPU参考ST |
| 接口资料 | 中英文支持矩阵、用户指南和本RFC |
| 性能 | 目标规模下无过滤和共享时间过滤均超过3200 QPS |
