# 常用操作<a name="ZH-CN_TOPIC_0000001698088057"></a>

## 日志说明<a name="ZH-CN_TOPIC_0000001506334653"></a>

检索日志组件基于《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html)》以及《[CANN 日志参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/logreference/logreference_0001.html)》设计和开发。

对于标准态部署，检索的日志属于应用类日志，可以参考《CANN 日志参考》中的“[查看日志（Ascend EP标准形态）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/logreference/logreference_0002.html)”章节的“查看应用类日志”描述。默认路径为“$HOME/ascend/log”。也可以使用环境变量`ASCEND_PROCESS_LOG_PATH`指定日志落盘路径。命令参考如下：

```bash
export ASCEND_PROCESS_LOG_PATH=$HOME/xxx
```

可指定日志落盘路径为任意有读写权限的目录。

日志级别由低到高依次为DEBUG < INFO < WARNING < ERROR，级别越低，输出日志越详细，可以通过`ASCEND_GLOBAL_LOG_LEVEL`环境变量设置日志级别。命令参考如下：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

不传入此参数，默认为ERROR等级。`ASCEND_GLOBAL_LOG_LEVEL` 全部取值说明如下：

0：DEBUG

1：INFO

2：WARNING

3：ERROR

4：NULL，NULL级别。不输出日志。

> [!NOTE]
>
>- 对于容器化场景中使用检索功能，应用类日志位于容器中，需要将日志目录挂载到宿主机才能实现持久化，否则日志将在容器退出时被销毁。
>- 应用类日志未配置自动轮转，日志会不断增多，因此需要用户定期清理该目录（可以使用系统自带的**logrotate**实现日志切分），否则可能导致磁盘空间不足，影响业务正常运行。
>- 软件包的安装升级卸载等管理面的相关日志会保存至“$HOME/log/mxIndex/deployment.log”，文件中保存有登录用户的用户名、访问端地址以及 hostname，用于支持后续的日志记录及审计的操作。

## IVFRaBitQ 运行时诊断<a name="ivfrabitq-runtime-debug"></a>

AscendIndexIVFRaBitQ 提供三组**可选**调试环境变量，用于排查coarse centroid（聚类中心）NPU上传异常与L1粗排probe选择偏差。**默认全部关闭**，对生产路径零开销；仅在开发/联调环境按需开启。

> [!NOTE]
>
>- 环境变量在**进程启动时**读取，需在运行应用程序或测试用例**之前**export。
>- 诊断日志输出至 stderr 及 APP 日志（`IVFRABITQ_VERIFY_COARSE_CENTER`）；建议重定向到 `.log` 文件便于 grep。
>- `IVFRABITQ_VERIFY_L1_DIST`与L1 golden对比会触发全量D2H，**不要在性能基准测试中常开**。
>- 修改`RotateAndL2AtFP32`算子后须重新编译部署custom opp，否则诊断结果可能仍反映旧版算子行为。

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

1. **recall 低且发生在`copyFrom`之后** → 开启`IVFRABITQ_VERIFY_COARSE_CENTER=1`，重跑copyFrom。
2. 查看 `centroidsOnDevice_rotated_full` 日志中的 `zeroRowsAfter2512`：
   - **H2D后originCentroidsOnDevice与host一致**，但 `zeroRowsAfter2512 > 0` → 故障在 `RotateAndL2AtFP32` 算子。
   - H2D 即 mismatch → 排查Memcpy参数或buffer容量。
3. **centroid 已确认正常，recall仍低** → 开启`IVFRABITQ_DEBUG_L1_PROBE=stats`，检查probe在tile1/tile2的分布。
4. **需对比8192边界L1距离** → 开启`IVFRABITQ_VERIFY_L1_DIST=1`或 `IVFRABITQ_DEBUG_L1_PROBE=full`。

### Coarse Center 上传诊断

```bash
export IVFRABITQ_VERIFY_COARSE_CENTER=1

# 边界测试（copyFrom 触发 updateCoarseCenterImpl）
./TestAscendIndexIVFRaBitQBoundary --gtest_filter="*CoarseCenterCopy10048*" 2>&1 | tee coarse_verify.log

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

## IVFRaBitQ 距离算子 block 编排<a name="ivfrabitq-topk-layout"></a>

L2 距离算子的 block 在核网格上的编排顺序可通过环境变量切换。与上文的诊断开关不同，**这组环境变量默认启用新编排（`group`）**；设为 `legacy` 回退历史的 query-major 连续编排，可作为 A/B 对比的对照组。环境变量一览见《[附录](./09_appendix.md#ivfrabitq-topk-layout-env)》。

历史编排按 query-major 连续填充：query 0 的所有 block、再 query 1……靠后的 query 开始得晚、结束得也晚，TopK 侧呈阶梯状等待。`group` 编排在一个窗口内对若干相邻 query 做 round-robin，窗口之间首尾相接，使同一窗口的 query 几乎同时出完，下一窗口接着用刚空出来的线程。

> [!NOTE]
>
>- 这两个变量在 **`AscendIndexIVFRaBitQ` 构造时**读取一次并缓存，**不是**每次 `search` 重读。必须在创建索引**之前** export，运行中修改不生效。
>- 两种编排只改变 block 在网格中的落位，不改变参与计算的数据，TopK 输出逐位相同。`group=1` 的填充顺序与 `legacy` 完全一致。
>- 本优化使算子输入个数由 7 变为 8，**必须重新生成 `TopkIvfRabitqFp32.om`**（`aicpu_generate_model.py` 已同步）。沿用旧 om 会在校验处直接报 `KERNEL_STATUS_PARAM_INVALID`，不会静默出错。

### 环境变量说明

**表 2** IVFRaBitQ block 编排环境变量

|环境变量名|取值|默认|用途|
|--|--|--|--|
|IVFRABITQ\_TOPK\_LAYOUT|`legacy` 或 `group`（`grouped` 等价）|`group`|选择 block 编排方式。`legacy` 为历史的 query-major 连续编排，仅作 A/B 基线保留|

该变量的**非法取值静默回退默认值，不打印 WARNING**：取 `legacy`/`group`/`grouped` 以外的字符串按默认 `group` 处理。设置后如需确认实际生效值，请以 A/B 实测结果为准。

窗口宽度（一个窗口容纳多少个相邻 query）**无需配置，由运行时从设备查询**：真正约束它的是 TopK 算子能并行消费一个窗口的线程数，窗口再宽也只是把每个 query 的完成时间整体推后。索引构造时通过 ACL 的 `aclrtGetDeviceCapability(ACL_DEV_ATTR_AICPU_CORE_NUM)` 取本设备的 AI CPU 个数作为窗口宽度。`legacy` 编排下窗口宽度强制为 `1`。

构造时若该线程尚未执行 `aclrtSetDevice`，查询会失败，此时先用常量 `6` 顶替，并在**首次检索**时自动重试一次；重试成功会打印一条 INFO 记录最终生效的宽度。因此「先建索引、后设设备」的用法不会把回退值永久锁死。若设备始终无法应答（CANN 版本不含该属性、仿真环境），则一直使用常量 `6`。

> [!NOTE]
>ACL 返回的是**设备**的 AI CPU 个数，而算子侧看到的是**本次 context** 分到的个数（`CpuKernelUtils::GetCPUNum`，算子内已与 `nq` 取 min）。同一张卡被多个索引实例共享时，前者会偏大。这只影响并行度、不影响正确性——`blockSlots` 是一张显式的双射表，算子按表读回，窗口宽度不改变任何 query 扫描哪些 block；窗口过宽至多退化为逐 block 轮转，即 `legacy` 已代表的行为。

回退方式：

```bash
# 回退历史编排（A/B 的对照组）；此时窗口宽度被强制为 1
export IVFRABITQ_TOPK_LAYOUT=legacy

# 恢复默认（group 编排，窗口宽度自动取设备 AI CPU 个数）
unset IVFRABITQ_TOPK_LAYOUT
## IVFRaBitQ TopK 路径调优<a name="ivfrabitq-topk-tuning"></a>

AscendIndexIVFRaBitQ 的 L2（精排）TopK 提供两条可切换的实现路径，以及一个容量参数。与上文的诊断开关不同，**这三个环境变量默认开启**，即不设置任何环境变量时走的就是新路径；设为 `0` 回退到历史实现，可作为 A/B 对比的对照组。环境变量一览见《[附录](./09_appendix.md#ivfrabitq-topk-tuning-env)》。

> [!NOTE]
>
>- 环境变量在 **Host 侧**读取，通过算子 attr 传给 AICPU kernel（AICPU 不继承 Host 进程环境变量，kernel 内 `getenv` 恒为空），因此需在运行应用程序**之前** export。
>- 两条路径的 TopK 结果集等价，但**并列距离的先后次序不保证一致**；存在距离相等的候选时，第 k 名由哪个候选占据本身是任意的。对比两条路径时应按距离多重集比较，标签仅对严格优于第 k 名的条目比较。
>- 二者相互独立，可单独开关，便于在**同一次部署内**完成 A/B：跨部署对比的噪声量级与待测收益同阶。

### 环境变量说明

**表 3** IVFRaBitQ TopK 路径调优环境变量

|环境变量名|取值|默认|实际生效条件|用途|
|--|--|--|--|--|
|IVFRABITQ\_TOPK\_RESERVOIR|`0` 关闭，其余开启|**开启**|`k >= 2048`，索引内已存最大 id 不超过 `UINT32_MAX`，且 burst 长度为 64；**bitmap ID 过滤模式下始终不生效**（见下）|候选先顺序写入扁平缓冲区，满了再用 `nth_element` 压回最优 k 个，替代每个候选都下沉一次 k-heap|
|IVFRABITQ\_TOPK\_PACKED\_SORT|`0` 关闭，其余开启|**开启**|`k >= 2048`|输出排序改用 packed introsort（距离与 id 打包进单个 `uint64`），替代堆原地逐个出栈|
|IVFRABITQ\_TOPK\_REFILL\_CHUNK|`[256, 32768]` 区间内的 **2 的幂**，填数值本身（如 `4096`）而非指数|`2048`|仅 reservoir 路径|reservoir 缓冲区分块大小 B：缓冲区容量为 `k + B`，B 越大则压缩次数越少、单次压缩越贵|

不满足“实际生效条件”时会自动回退历史路径，不报错。其中 id 上界这一条是必要的：reservoir 的候选以 `{float dist; uint32_t id}` 存储（8 字节无填充，相对 int64 id 的 16 字节减半），若索引内存在超过 `UINT32_MAX` 的 id，走该路径会把标签截断为低 32 位，返回一个看起来合法的错误标签。Host 侧按**实际已存入的最大 id**（而非向量总数 `ntotal`）判断，因为调用方传入的 label 与向量计数无关；稠密序号（如 `vid = indexed_count_ + i`）天然满足该上界。

**bitmap ID 过滤（IDSelectorBitmap）与 reservoir 是互斥的**：bitmap 路径在算子内自带围绕预取窗口的内联扫描，没有可供 reservoir 接管的 accept 钩子，因此该模式下无论环境变量如何设置都走历史 k-heap。这不是正确性缺口——堆路径的结果与 reservoir 路径一致，只是一个尚未优化的组合。**packed introsort 不受影响，在 bitmap 模式下照常生效**（排序出口与扫描方式无关）。

`IVFRABITQ_TOPK_REFILL_CHUNK` 的非法取值（非 2 的幂、越界、非数字）**不会就近取整**，而是打印 WARNING 后回退默认值 `2048`，避免实测的 B 与设置的 B 不一致。

前两个开关的关闭判定是**整串精确匹配** `0`：`0`（以及 `unset` / 空串回退默认）关闭，其余任何取值——包括 `00`、`0x1`、`0abc`——都视为开启。

关闭方式：

```bash
# 回退到历史 k-heap + 堆出栈路径（A/B 的对照组）
export IVFRABITQ_TOPK_RESERVOIR=0
export IVFRABITQ_TOPK_PACKED_SORT=0

# 或仅单独关闭其中一个，以区分两项优化各自的收益
export IVFRABITQ_TOPK_RESERVOIR=0

# 恢复默认（开启）
unset IVFRABITQ_TOPK_RESERVOIR IVFRABITQ_TOPK_PACKED_SORT
```

### 使用示例

```bash
# 对照组：历史 query-major 编排
export IVFRABITQ_TOPK_LAYOUT=legacy
./your_benchmark 2>&1 | tee layout_legacy.log

# 实验组：默认 group 编排，窗口宽度自动取设备 AI CPU 个数
unset IVFRABITQ_TOPK_LAYOUT
./your_benchmark 2>&1 | tee layout_group.log
# 对照组：历史路径
export IVFRABITQ_TOPK_RESERVOIR=0
export IVFRABITQ_TOPK_PACKED_SORT=0
./your_benchmark 2>&1 | tee baseline.log

# 实验组：默认（两项均开启）
unset IVFRABITQ_TOPK_RESERVOIR IVFRABITQ_TOPK_PACKED_SORT
./your_benchmark 2>&1 | tee reservoir.log

# 调整 reservoir 分块大小（必须是 2 的幂）
export IVFRABITQ_TOPK_REFILL_CHUNK=4096
./your_benchmark 2>&1 | tee reservoir_b4096.log
```

## Device 内存调试<a name="ascendfaiss-mem-debug"></a>

AscendFaiss 提供两组**可选**环境变量，用于排查 Device HBM 分配失败、索引上传/扩容过程中的显存占用变化。**默认全部关闭**，对生产路径零开销；仅在开发/联调环境按需开启。环境变量一览见《[附录](./09_appendix.md#ascendfaiss-mem-debug-env)》。

> [!NOTE]
>
>- 环境变量在**进程首次查询时**读取并缓存，需在运行应用程序或测试用例**之前** export。
>- 调试日志输出至 stderr，并同步写入 APP 日志（`[MemDebug]` 前缀）；建议重定向到 `.log` 文件便于 grep。
>- 开启后会查询 `aclrtGetMemInfo(ACL_HBM_MEM)` 并抽样打印，**不要在性能基准测试中常开**。

### 环境变量说明

**表 4** Device 内存调试环境变量

<table><tbody>
<tr><td align="center" valign="middle"><strong>环境变量名</strong></td><td align="center" valign="middle"><strong>取值</strong></td><td width="70" align="center" valign="middle"><strong>默认</strong></td><td align="center" valign="middle"><strong>用途</strong></td></tr>
<tr><td valign="middle">ASCENDFAISS\_MEM\_DEBUG</td><td valign="middle">非空且非 <code>0</code> / <code>false</code> / <code>off</code>（大小写不敏感）</td><td width="70" align="center" valign="middle">关闭</td><td valign="middle">总开关：开启分配抽样、HBM 余量打印；<code>aclrtMalloc</code> 失败时 dump 最近分配环形缓冲</td></tr>
<tr><td valign="middle">ASCENDFAISS\_MEM\_DEBUG\_EVERY</td><td valign="middle">正整数 N</td><td width="70" align="center" valign="middle"><code>64</code></td><td valign="middle">抽样周期：分配序号满足 <code>seq % N == 0</code>，或 <code>size ≤ 4096</code> 时打印明细；未设置/非法/0 时回退为 <code>64</code></td></tr>
</tbody></table>

关闭方式：

```bash
unset ASCENDFAISS_MEM_DEBUG ASCENDFAISS_MEM_DEBUG_EVERY
# 或
export ASCENDFAISS_MEM_DEBUG=0
```

### 使用示例

```bash
# 开启内存调试（默认每 64 次分配抽样一次）
export ASCENDFAISS_MEM_DEBUG=1

# 提高采样密度（每 8 次分配打印一次）
export ASCENDFAISS_MEM_DEBUG=1
export ASCENDFAISS_MEM_DEBUG_EVERY=8

# 运行业务或 UT，并保存日志
./your_app 2>&1 | tee mem_debug.log
grep '\[MemDebug\]' mem_debug.log
```

### 日志解读

|日志关键字|含义|
|--|--|
|`[MemDebug] alloc seq=...`|一次 Device 分配的抽样记录：序号、size、space、device、分配前 HBM free/total|
|`[MemDebug] ... HBM free=...`|关键路径上的 HBM 余量快照（如 `copyVectorToDevice_*`、`DeviceMemArena::Grow`、`IndexIVFRaBitQ_resize`）|
|`[MemDebug] aclrtMalloc FAILED ...`|分配失败：当前请求 size/space/device、错误码、当时 HBM free/total|
|`dumping last N alloc records`|失败前最近最多 64 条分配记录（oldest→newest），用于定位耗尽显存的连续分配|

**典型排查流程：**

1. 复现 OOM / `aclrtMalloc` 失败场景前设置 `ASCENDFAISS_MEM_DEBUG=1`。
2. 在失败日志中查看 `HBM_free` 是否已接近 0，以及 `size` / `space`（`DEVICE` 或 `DEVICE_HUGEPAGE`）。
3. 根据 dump 的最近分配记录，对照业务阶段（索引上传、list resize、arena grow）缩小泄漏或峰值分配点。
4. 需要更密采样时增大频率：`export ASCENDFAISS_MEM_DEBUG_EVERY=1`（每笔分配都打印，日志量很大）。
