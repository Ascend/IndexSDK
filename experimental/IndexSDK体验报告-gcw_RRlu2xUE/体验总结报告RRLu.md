# 【MindSDK】基于 IndexSDK 完成检索算法验证任务 总结报告

> **一句话结论**：达成目标，三条任务全部完成；关键阻塞点：源码独立构建出的安装包版本字段为空（不影响功能，已提 Issue #216）；关键指标：Flat 100 万×512 维 / 128 查询 / Top10 → 227.2731 QPS；Int8Flat 100 万×256 维六档最高 22794.0403 QPS，Recall@1/@10/@100 均 100%。

## 1 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档类型 | 验证报告 |
| 提交人 | GitCode ID：dsdjafk |
| 文档版本 / 日期 | v1.1 / 2026-11-20 |
| 验证源码 | Ascend/IndexSDK，commit `99c64b7`（master，浅克隆 depth=1） |
| 运行设备 | Ascend 910B3 单卡（HBM 65536 MB），镜像【应用使能】IndexSDK / A2 |

## 2 任务背景与目标

- **任务目标**：在昇腾 A2 单卡环境按官方文档完成①源码安装方式构建不含 ILFlat 的 run 包并重装；②Flat 算子生成 + 用例编译并跑通检索用例；③阅读 Int8Flat 文档后完成其算子生成（注意生成参数）并跑通 `examples/TestAscendIndexInt8Flat.cpp`。
- **任务结果**：达成目标；关键阻塞点：有，源码独立构建得到的安装包文件名中版本字段为空（功能不受影响，已提 Issue #216 并附最小复现）；关键指标：Flat 227.2731 QPS（100 万条 512 维、128 查询、Top10，耗时 563.199 ms）；Int8Flat 六档查询批量 QPS 5533.2658 / 9802.7074 / 14435.8782 / 22294.9481 / 21963.6626 / 22794.0403，准确性用例 Recall@1 = @10 = @100 = 100.00。

## 3 环境准备

### 3.1 环境创建步骤

**步骤 1：申请 / 创建算力环境**

- 操作：HiDevLab 在线开发平台（https://hidevlab.huawei.com/online-develop），按任务书选择容器资源，A2 / 1 NPU / 内存 32 GB / 硬盘 300 GB。
- 截图：平台任务分配页与创建页 → `截图/00_task_assignment.png`、`截图/01_env_create.png`；实例列表 → `截图/01_env_instance.png`

**步骤 2：选择镜像**

- 操作：镜像选择「【应用使能】IndexSDK」，规格「1 NPU · 算子调测小模型训推」。
- 截图：镜像选择页 → `截图/02_env_image.png`

**步骤 3：启动并登录开发环境**

- 操作：开机至进度 100% 后，通过终端进入实例（本报告日志与截图中的实例标识与工作路径已脱敏为 `REDACTED`）。
- 截图：登录成功后的终端提示符 → `截图/03_env_login.png`、`截图/03_terminal.png`

### 3.2 环境信息

`npu-smi info`

```
| NPU  Name  | Health | Power(W) Temp(C) | HBM-Usage(MB) |
| 5    910B3 | OK     | 100.5    58      | 0 / 0         |
| 0          |        |                 | 3414 / 65536  |
```

Python 与依赖：`export PATH=/usr/local/python3.12.15/bin:$PATH`，`python3 --version` = Python 3.12.15；Faiss 1.10、gtest、OpenBLAS 均由镜像预置。

**截图**：`截图/04_terminal.png`

## 4 实际执行步骤

- **任务入口文档**：https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/04_installation_guide.md （安装）、`/docs/zh/03_quick_start.md`（快速入门）、`/docs/zh/05_user_guide.md`（使用指南）、`/examples/README.md`（用例构建）
- **样例 / 代码路径**：https://gitcode.com/Ascend/IndexSDK/tree/master/examples 。本次**未修改样例代码**，仅用 `git diff --exit-code HEAD -- examples/TestAscendIndexInt8Flat.cpp` 确认工作区干净。
- **唯一改动**：`feature_retrieval/src/ascendfaiss/CMakeLists.txt` 两处（任务书要求"不需要 ILFlat"），可用 `git checkout` 回退。

### 步骤 1：确认设备与工具链

    export PATH=/usr/local/python3.12.15/bin:$PATH
    npu-smi info
    python3 --version

- **预期**：能看到 910B3 单卡，Python 3.12 可用。
- **实际**：成功，`910B3`、Health OK、HBM 65536 MB，Python 3.12.15。
- **截图**：![步骤1](截图/04_terminal.png)

### 步骤 2：拉取源码

    git -c http.lowSpeedLimit=100 -c http.lowSpeedTime=30 clone --depth 1 --branch master \
        https://gitcode.com/Ascend/IndexSDK.git IndexSDK
    cd IndexSDK && git rev-parse HEAD

- **预期**：得到 master 最新提交 `99c64b7`，工作区干净。
- **实际**：成功。
- **截图**：![步骤2](截图/05_terminal.png)

### 步骤 3：关闭 ILFlat 并源码构建 run 包

    cfg=feature_retrieval/src/ascendfaiss/CMakeLists.txt
    sed -i 's/^OPTION(BUILD_ASCENDDEVICE.*/OPTION(BUILD_ASCENDDEVICE "Build ascenddevice" OFF)/' "$cfg"
    sed -i 's/^\([[:space:]]*\)ASCEND_MINIOS_HOME[[:space:]]*$/\1# ASCEND_MINIOS_HOME/' "$cfg"
    git diff -- "$cfg"
    bash build/build.sh
    ls -l build/output/*.run

- **预期**：生成 `Ascend-mindxsdk-mxindex_{版本}_linux-aarch64.run`，版本号非空。
- **实际**：成功产出 run 包，但**文件名版本字段为空**：`Ascend-mindxsdk-mxindex__linux-aarch64.run`（见第 6 节 Issue #216）。
- **截图**：![步骤3](截图/06_terminal.png)
- **口径**：本节从 `[START]` 到 `[END]` 共 **6 分 14 秒**（含构建 + `--check` 自校验）。

### 步骤 4：升级安装到 910B 平台

    bash build/output/Ascend-mindxsdk-mxindex__linux-aarch64.run --check
    bash build/output/Ascend-mindxsdk-mxindex__linux-aarch64.run --upgrade --quiet \
        --install-path=/usr/local/Ascend --platform=910B

- **预期**：自校验通过；安装器卸旧装新，输出升级成功。
- **实际**：成功。`Check successfully, exit with 0`；`Install package successfully` → `upgrade package successfully`。
- **截图**：![步骤4](截图/07_terminal.png)

### 步骤 5：安装随包自定义算子

    cd /usr/local/Ascend/mxIndex/ops && ./custom_opp_aarch64.run

- **预期**：`Self-extractable archive "custom_opp_aarch64.run" successfully created.`
- **实际**：成功。
- **截图**：![步骤5](截图/08_terminal.png)

### 步骤 6：生成 AICPU 与 Flat 模型

    cd /usr/local/Ascend/mxIndex/tools
    python3 aicpu_generate_model.py -t 910B3
    python3 flat_generate_model.py -d 512 -t 910B3
    cp -a op_models/. "$MX_INDEX_MODELPATH/"

- **预期**：生成公共 AICPU 模型与 512 维 Flat 模型，模型目录文件数不为 0。
- **实际**：成功，`Model files in .../models: 91`，耗时 **7 分 59 秒**。
- **截图**：![步骤6](截图/09_terminal.png)

### 步骤 7：编译 Flat 快速入门 demo

    cd "$RUN/flat"
    bash -ex compile.sh      # 编译命令直接取自 docs/zh/03_quick_start.md

- **预期**：编译产出可执行文件 `demo`，无链接错误。
- **实际**：成功。
- **截图**：![步骤7](截图/10_terminal.png)、![步骤7b](截图/10b_terminal.png)

### 步骤 8：运行 Flat 检索

    cd "$RUN/flat" && ./demo

- **预期**：输出 `search end!` 与性能统计行。
- **实际**：成功。
  ```
  search end!
  flat, base:1000000, dim:512, searchnum:128, topk:10, duration:563.199, QPS:227.2731
  ```
- **截图**：![步骤8](截图/11_terminal.png)、![步骤8实机控制台](截图/11_flat_live_console.png)

### 步骤 9：按样例参数生成 Int8Flat 模型

    # 第一次直接按 12.sh 生成（工具默认参数，产物与样例分块不匹配，未用于本例）
    cd /usr/local/Ascend/mxIndex/tools
    python3 int8flat_generate_model.py -d 256 -t 910B3

- **预期**：产出 256 维 int8 模型。
- **实际**：未直接采用。按 12.sh 生成的分块与样例默认分块（16384 × 16 = 262144）不一致，全量生成耗时也明显更长，因此改为第 10 步在隔离目录按样例尺寸重生成（本日志无 `[END]` 行，该次执行被 12b 替代）。
- **截图**：![步骤9](截图/12_terminal.png)

### 步骤 10：隔离目录限定生成范围

    cd "$RUN/IndexSDK" && git diff --exit-code HEAD -- examples/TestAscendIndexInt8Flat.cpp
    mkdir -p "$RUN/int8-model-build" && cd "$RUN/int8-model-build"
    python3 /usr/local/Ascend/mxIndex/tools/int8flat_generate_model.py -d 256 -t 910B3 -code 262144
    cp -a op_models/. "$MX_INDEX_MODELPATH/"

- **预期**：只生成样例所需分块大小的模型，产物数少于全量。
- **实际**：成功，`Int8 model files from isolated, sample-sized generation: 63`，耗时 **12 分 40 秒**（全流程最耗时一步）。
- **截图**：![步骤10](截图/12b_terminal.png)

### 步骤 11：编译 Int8Flat 用例

    cd "$RUN/IndexSDK/examples"
    bash build.sh TestAscendIndexInt8Flat

- **预期**：`build/TestAscendIndexInt8Flat` 生成。
- **实际**：成功，耗时 4 秒。
- **截图**：![步骤11](截图/13_terminal.png)

### 步骤 12：运行 Int8Flat 用例

    cd "$RUN/IndexSDK/examples/build" && ./TestAscendIndexInt8Flat

- **预期**：六个批量均输出 QPS；Acc 用例打印 Recall 且两个 TEST 均 OK。
- **实际**：成功，11 秒。
  ```
  case[0]: base:1000000, dim:256, search num:8,   QPS:5533.2658
  case[1]: base:1000000, dim:256, search num:16,  QPS:9802.7074
  case[2]: base:1000000, dim:256, search num:32,  QPS:14435.8782
  case[3]: base:1000000, dim:256, search num:64,  QPS:22294.9481
  case[4]: base:1000000, dim:256, search num:128, QPS:21963.6626
  case[5]: base:1000000, dim:256, search num:256, QPS:22794.0403
  Recall 100: @1 = 100.00, @10 = 100.00, @100 = 100.00
  [       OK ] TestAscendIndexInt8Flat.Acc (3556 ms)
  [  PASSED  ] 2 tests.
  ```
- **截图**：![步骤12](截图/14_terminal.png)、![步骤12实机控制台](截图/14_int8flat_live_console.png)

## 5 执行结果

- **结论**：通过
- **判据**：三条任务全部跑通——run 包构建 + `--check` 自校验 + `--upgrade` 升级成功；Flat 输出 `search end!` 并给出完整性能行；Int8Flat 六档 QPS 齐全、Recall 三项均为 100.00、GoogleTest `[  PASSED  ] 2 tests.`，进程退出码 0，全程无异常栈。

关键输出（日志原文）：

```
[mxIndex] [20261120-06:27:08] [INFO] Check successfully, exit with 0
[mxIndex] [20261120-06:33:07] [INFO] Install package successfully
[mxIndex] [20261120-06:33:07] [INFO] upgrade package successfully

search end!
flat, base:1000000, dim:512, searchnum:128, topk:10, duration:563.199, QPS:227.2731

case[5]: base:1000000, dim:256, search num:256, QPS:22794.0403
Recall 100: @1 = 100.00, @10 = 100.00, @100 = 100.00
[  PASSED  ] 2 tests.
```

## 6 发现的问题

| 问题 | 提交 issue |
| --- | --- |
| 独立克隆源码构建时，`build/package.sh` 读取不到版本配置仍继续打包，产出的 run 包文件名版本字段为空（`Ascend-mindxsdk-mxindex__linux-aarch64.run`），重装后版本文件同样为空，不影响功能但影响版本追踪与回滚 | [#216](https://gitcode.com/Ascend/IndexSDK/issues/216) |
| `examples/TestAscendIndexInt8Flat.cpp` 第 20 行文件头注释给的生成命令是 `flat_generate_model.py`，与用户指南要求的 Int8Flat 专用脚本 `int8flat_generate_model.py` 不一致，按注释执行会生成错误的 Flat 模型 | [#217](https://gitcode.com/Ascend/IndexSDK/issues/217) |

覆盖范围说明：以上为本任务三条规定动作（源码构建重装、Flat、Int8Flat）实际触达的路径；IVF/SQ/Binary/Cagra 等其余算法、多卡与 A3/950PR 平台未纳入本次验证。

## 7 总结

1. 关闭 ILFlat 后源码构建可用：从 `bash build/build.sh` 到 `--check` 自校验结束 **6 分 14 秒**，`--upgrade --platform=910B` 升级安装成功，样例可正常加载新库。
2. Flat 侧 100 万条 512 维、128 查询、Top10 实测 **563.199 ms / 227.2731 QPS**，输出 `search end!`，与快速入门给出的判据完全一致。
3. Int8Flat 侧必须按样例（dim=256、分块 262144）限定生成参数：按此生成的模型跑出六档 **5533 ~ 22794 QPS**，Recall@1/@10/@100 **均 100.00**，两个用例 PASSED；未限定分块的全量生成会产出与样例不匹配的模型。
4. 两个问题均为文档/脚本层面可用性问题（版本字段、样例注释），不影响检索功能，已按【MeetUp】规范提交 Issue #216、#217。

## 8 体验总结

| 维度 | 评价 | 说明 |
| --- | --- | --- |
| 文档易懂度 | 一般 | 快速入门的 Flat 部分可只看文档独立完成；Int8Flat 的生成脚本与分块参数分散在样例注释、使用指南、配置 API 三处，且样例注释本身写错（#217），需要交叉核对才能确定正确命令。 |
| 流程顺畅度 | 一般 | 环境创建→构建→重装→算子→模型→用例的链路完整清晰；最费时的是模型生成（Flat 7 分 59 秒、Int8Flat 12 分 40 秒），其中 Int8Flat 需要尝试"全量生成→发现分块不匹配→隔离目录按 -code 262144 重生成"一次返工。 |
| 报错可读性 | 一般 | 打包阶段版本配置缺失时脚本未失败而是静默产出空版本号包（#216），直到检查包名/版本文件才发现，容易误判为"构建成功即无问题"。 |
| 整体评价 | 3 / 5 | 三条任务均可复现，文档骨架可用；建议补齐源码构建的版本来源，并在 Int8Flat 样例入口直接给出专用生成命令。 |

- **最耗时的一步**：步骤 10（Int8Flat 模型按样例分块生成），约 **12 分 40 秒**；其次为步骤 6（AICPU + Flat 模型）7 分 59 秒、步骤 3（源码构建）6 分 14 秒。
- **最值得改进的点**：① `build/package.sh` 读取不到版本配置时应报错退出，而不是产出空版本包；② `TestAscendIndexInt8Flat.cpp` 文件头注释改为 `int8flat_generate_model.py`，并注明 `-d 256 -code 262144` 与样例分块的对应关系。

---

**附件清单**：`截图/` 共 22 张（`00_task_assignment` ~ `14_int8flat_live_console`）；`命令/03.sh–14.sh` 逐步命令；`日志/03.log–14.log` 逐步原始日志（含 `[START]/[END]` 时间戳，本报告的耗时口径全部取自此处）。实例标识与工作路径已脱敏。
