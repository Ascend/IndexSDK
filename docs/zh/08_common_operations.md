# 常用操作<a name="ZH-CN_TOPIC_0000001698088057"></a>

## 日志说明<a name="ZH-CN_TOPIC_0000001506334653"></a>

检索日志组件基于《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html)》以及《[CANN 日志参考](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/logreference/logreference_0001.html)》设计和开发。

对于标准态部署，检索的日志属于应用类日志，可以参考《CANN 日志参考》中的“[查看日志（Ascend EP标准形态）](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/logreference/logreference_0002.html)”章节的“查看应用类日志”描述。默认路径为“$HOME/ascend/log”。也可以使用环境变量ASCEND\_PROCESS\_LOG\_PATH指定日志落盘路径。命令参考如下：

```bash
export ASCEND_PROCESS_LOG_PATH=$HOME/xxx
```

可指定日志落盘路径为任意有读写权限的目录。

日志级别由低到高依次为DEBUG < INFO < WARNING < ERROR，级别越低，输出日志越详细，可以通过ASCEND\_GLOBAL\_LOG\_LEVEL环境变量设置日志级别。命令参考如下：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

不传入此参数，默认为ERROR等级。ASCEND\_GLOBAL\_LOG\_LEVEL全部取值说明如下：

0：DEBUG

1：INFO

2：WARNING

3：ERROR

4：NULL，NULL级别。不输出日志。

> [!NOTE]
>
>- 对于容器化场景中使用检索功能，应用类日志位于容器中，需要将日志目录挂载到宿主机才能实现持久化，否则日志将在容器退出时被销毁。
>- 应用类日志未配置自动轮转，日志会不断增多，因此需要用户定期清理该目录（可以使用系统自带的**logrotate**实现日志切分），否则可能导致磁盘空间不足，影响业务正常运行。
>- 软件包的安装升级卸载等管理面的相关日志会保存至“$HOME/log/mxIndex/deployment.log”，文件中保存有登录用户的用户名、访问端地址以及hostname，用于支持后续的日志记录及审计的操作。

## IVFRaBitQ 运行时诊断<a name="ivfrabitq-runtime-debug"></a>

AscendIndexIVFRaBitQ 提供三组**可选**调试环境变量，用于排查 coarse centroid（聚类中心）NPU 上传异常与 L1 粗排 probe 选择偏差。**默认全部关闭**，对生产路径零开销；仅在开发/联调环境按需开启。

> [!NOTE]
>
>- 环境变量在**进程启动时**读取，需在运行应用程序或测试用例**之前** export。
>- 诊断日志输出至 stderr 及 APP 日志（`IVFRABITQ_VERIFY_COARSE_CENTER`）；建议重定向到 `.log` 文件便于 grep。
>- `IVFRABITQ_VERIFY_L1_DIST` 与 L1 golden 对比会触发全量 D2H，**不要在性能基准测试中常开**。
>- 修改 `RotateAndL2AtFP32` 算子后须重新编译部署 custom opp，否则诊断结果可能仍反映旧版算子行为。

### 环境变量说明

**表 1** IVFRaBitQ 调试环境变量

|环境变量名|取值|触发时机|用途|
|--|--|--|--|
|IVFRABITQ\_VERIFY\_COARSE\_CENTER|非空且非 `0`| `copyFrom` / 训练后更新 centroid（`updateCoarseCenterImpl`）|分阶段 D2H 校验：区分 H2D Memcpy 失败 vs rotate 算子输出不完整|
|IVFRABITQ\_DEBUG\_L1\_PROBE|`1` / `stats` / `full`|每次 `search` 的 L1 阶段|观察 probe 是否仅落在 `[0,8192)` 而忽略后半段 list|
|IVFRABITQ\_VERIFY\_L1\_DIST|非空且非 `0`|每次 `search` 的 L1 阶段|CPU golden vs NPU L1 距离及 probe 一致性对比|

关闭方式：

```bash
unset IVFRABITQ_VERIFY_COARSE_CENTER IVFRABITQ_DEBUG_L1_PROBE IVFRABITQ_VERIFY_L1_DIST
# 或设为 0
export IVFRABITQ_VERIFY_COARSE_CENTER=0
```

### 诊断决策流程

1. **recall 低且发生在 `copyFrom` 之后** → 开启 `IVFRABITQ_VERIFY_COARSE_CENTER=1`，重跑 copyFrom。
2. 查看 `centroidsOnDevice_rotated_full` 日志中的 `zeroRowsAfter2512`：
   - **H2D 后 originCentroidsOnDevice 与 host 一致**，但 `zeroRowsAfter2512 > 0` → 故障在 `RotateAndL2AtFP32` 算子。
   - H2D 即 mismatch → 排查 Memcpy 参数或 buffer 容量。
3. **centroid 已确认正常，recall 仍低** → 开启 `IVFRABITQ_DEBUG_L1_PROBE=stats`，检查 probe 在 tile1/tile2 的分布。
4. **需对比 8192 边界 L1 距离** → 开启 `IVFRABITQ_VERIFY_L1_DIST=1` 或 `IVFRABITQ_DEBUG_L1_PROBE=full`。

### Coarse Center 上传诊断

```bash
export IVFRABITQ_VERIFY_COARSE_CENTER=1

# 边界测试（copyFrom 触发 updateCoarseCenterImpl）
./TestAscendIndexIVFRaBitQBoundary --gtest_filter=*CoarseCenterCopy10048* 2>&1 | tee coarse_verify.log

grep -E 'CoarseCenterVerify|zeroRowsAfter2512' coarse_verify.log
```

**修复后期望日志：**

```text
[CoarseCenterVerify] originCentroidsOnDevice: row 2512 OK vs host (devNorm=...)
[CoarseCenterVerify] centroidsOnDevice_rotated_full full: mismatchRows=0 zeroRowsBefore2512=0 zeroRowsAfter2512=0 / 7536
```

**修复前典型异常（nlist=10048）：**

```text
[CoarseCenterVerify] centroidsOnDevice_rotated: row 2512 is all-zero on device (devNorm=0.000000)
[CoarseCenterVerify] centroidsOnDevice_rotated_full full: zeroRowsAfter2512=7536 / 7536
```

关键判定：`2512 = 10048 / 4`，为四核 AIC 均匀分核时单核 batch 大小；row 2512 是首个“非 core 0”边界行。

### L1 Probe 分布诊断

```bash
# 打印 q0 前 8 个 probe id
export IVFRABITQ_DEBUG_L1_PROBE=1

# 仅统计 probe 在 [0,8192) 与 [8192,nlist) 的分布（推荐，开销小）
export IVFRABITQ_DEBUG_L1_PROBE=stats

# probe 列表 + CPU golden 对比 + 分布统计（开销最大）
export IVFRABITQ_DEBUG_L1_PROBE=full
```

**stats 模式示例：**

```text
[IVFRaBitQ] L1 probe stats q0: nprobe=1024 in[0,8192)=1024 in[8192,10048)=0 min=3 max=8191
```

异常信号：`in[8192,nlist)=0` 且 nlist > 8192，说明 probe 未覆盖后半段 list，常与 centroid device 零行或 L1 距离算子异常相关。

### L1 距离 Golden 对比

```bash
export IVFRABITQ_VERIFY_L1_DIST=1
# 运行 search 场景
grep 'L1 dist golden\|jaccard' search.log
```

**边界 id 距离（抽样 8191/8192/8193 及尾部行）：**

```text
[IVFRaBitQ] L1 dist golden id=8192 cpu=15.678901 npu=15.678902 absErr=0.000001
```

**Probe overlap：**

```text
[IVFRaBitQ] L1 probe overlap q0: nprobe=1024 overlap=980 jaccard=0.957 cpu_tile2=128
```

|字段|含义|健康参考|
|--|--|--|
|overlap|CPU top-nprobe 与 NPU probe 交集|接近 nprobe|
|jaccard|overlap / nprobe|> 0.95|
|cpu_tile2|CPU golden probe 落在 `[8192,nlist)` 的数量|若 NPU tile2=0 而 cpu_tile2>0，说明 NPU 遗漏后半段|

### 推荐组合

|场景|export 组合|
|--|--|
|验证 copyFrom 上传| `IVFRABITQ_VERIFY_COARSE_CENTER=1`|
|centroid 正常，排查 L1| `IVFRABITQ_DEBUG_L1_PROBE=stats`|
|8192 边界 L1 距离| `IVFRABITQ_VERIFY_L1_DIST=1`|
|一次性完整 dump| `IVFRABITQ_DEBUG_L1_PROBE=full`|

### 延伸阅读

- [FAQ：IVFRaBitQ recall 偏低](./07_faq.md#ivfrabitq-recall-low-nlist-10048)
