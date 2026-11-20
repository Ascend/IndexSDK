# IndexSDK 检索算法验证 总结报告

> **一句话结论**：已达成（用户自行修复后跑通）；关键阻塞点：编译期间接依赖未解析、运行期算子模型 SoC 版本不匹配；关键指标：demo 链接由失败转为成功（1 次参数修正）、修复前运行固定失败于 ACL 100014。

## 1 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档类型 | 验证报告 |
| 提交人 | GitCode ID：<待填> |
| 文档版本 / 日期 | v1 / 2026-11-20 |

> 说明：本次为 SSH 远程排障，终端侧没有图形界面，文中 `01~04` 与各步骤截图位为**待补拍占位**（保留命名，便于直接替换）；命令行证据以日志原文形式给出。

## 2 任务背景与目标

- **任务目标**：在昇腾算力环境（910B4）上按官方文档编译并运行 mxIndex / IndexSDK 的 `demo`（AscendIndexFlat，dim=512、底库 100 万、topK=10），程序正常结束且无异常栈。
- **任务结果**：达成，最终由用户自行修复跑通；**关键阻塞点**有两个：① 链接期 `libascendfaiss.so` 的间接依赖无法解析；② 运行期算子模型按 910B3 编译、与 910B4 芯片不匹配（ACL 100014）。**关键指标**：编译命令修正后 `LINK_STATUS=0` 并产出可执行文件；修复前每次运行均在 12 s 内抛出异常退出（exit 255）。

## 3 环境准备

### 3.1 环境创建步骤

> 按实际创建顺序记录，**每步必须配截图**，截图命名 `NN_内容.png`。

**步骤 1：申请 / 创建算力环境**

- 操作：OpenLibing 算力环境，DevEnv 节点（经跳板机 `113.47.8.48:2222` 登录），实例 hostname `3a67d938373b`，NPU 1 卡 / 单芯片，HBM 65536 MB
- 截图：平台创建页或实例列表 → `01_env_create.png`（待补拍）

**步骤 2：选择镜像**

- 操作：镜像已预置 CANN 9.2.0 与 mxIndex（`/usr/local/Ascend/cann` → `cann-9.2.0`，`/usr/local/Ascend/mxIndex`），OS 为 Ubuntu 22.04.5 LTS（aarch64）
- 截图：镜像选择页 → `02_env_image.png`（待补拍）

**步骤 3：启动并登录开发环境**

- 操作：SSH 登录，`whoami` 为 root
- 截图：登录成功后的终端提示符 → `03_env_login.png`（待补拍）

### 3.2 环境信息

`npu-smi info` 关键输出（原文）：

    | 6     910B4-1             | OK            | 93.4                 45                      0    / 0                |
    | 0                         | 0000:41:00.0  | 0                    0    / 0                3452 / 65536            |
    | No running processes found in NPU 6

补充环境事实：

| 项 | 值 |
| --- | --- |
| 芯片 | 910B4-1（Product Name `IT21HMDC_Bin4_1`），AICore 20，HBM 65536 MB |
| 驱动 / 固件 | 26.1.1 / 9.0.0.9.220 |
| CANN | `cann-9.2.0`（`/usr/local/Ascend/cann` 软链） |
| mxIndex | `/usr/local/Ascend/mxIndex`（设备侧 + host 侧库齐全） |
| 工具链 | g++ 11.4.0、GNU ld 2.38 |

**截图**：`04_npu_smi.png`（待补拍）

## 4 实际执行步骤

- **任务入口文档**：<待填>
- **样例 / 代码路径**：`/usr/local/Ascend/mxIndex-/tools/demo.cpp`（90 行，与 `/usr/local/Ascend/mxIndex/tools` 为同一 inode）
- **是否修改样例代码**：否；仅调整编译链接参数与环境变量，可回退

### 步骤 1：按文档命令编译 demo

    export MX_INDEX_INSTALL_PATH=/usr/local/Ascend/mxIndex
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
    g++ --std=c++11 -fPIC -fPIE -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O3 \
      -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -o demo demo.cpp \
      -I$MX_INDEX_INSTALL_PATH/include -I/usr/local/faiss/include \
      -I/usr/local/Ascend/driver/include -I/opt/OpenBLAS/include \
      -L$MX_INDEX_INSTALL_PATH/host/lib -L/usr/local/faiss/lib \
      -L/usr/local/Ascend/driver/lib64 -L/usr/local/Ascend/driver/lib64/driver \
      -L/opt/OpenBLAS/lib -L$ASCEND_HOME_PATH/lib64 \
      -lfaiss -lascendfaiss -lopenblas -lc_sec -lascendcl -lascend_hal -lascendsearch -lock_hmm

- **预期**：生成 `demo` 可执行文件
- **实际**：失败。`ld` 报 `libock_vsa_hpp.so.1` / `libock_hcps_pier.so.1` not found，并连带 4 个 `ock::` 未定义符号，`collect2: error: ld returned 1 exit status`
- **截图**：`05_step1_link_fail.png`（待补拍）

### 步骤 2：确认库是否存在

    ls -l /usr/local/Ascend/mxIndex/host/lib/
    readelf -d /usr/local/Ascend/mxIndex/host/lib/libascendfaiss.so | grep NEEDED
    ldd /usr/local/Ascend/mxIndex/host/lib/libascendfaiss.so | grep -i "not found"

- **预期**：确认缺失库是否真的不存在
- **实际**：库**存在**（`libock_vsa_hpp.so.1`、`libock_hcps_pier.so.1`、`libock_hmm.so.1` 齐全，且该目录已在 `-L` 中）。说明不是缺文件，是链接器搜索路径语义问题
- **截图**：`06_step2_libs.png`（待补拍）

### 步骤 3：原样复现，暴露全部间接依赖

    # 同上第 1 步命令，输出改为 demo_test
- **预期**：复现同样的报错
- **实际**：复现，并多出 3 条同类告警：`libacl_op_compiler.so`、`libmsprofiler.so`、`libascend_dump.so` not found；未定义符号增至 6 个（含 `aclopCompile`、`aclSetCompileopt`）
- **截图**：`07_step3_reproduce.png`（待补拍）

### 步骤 4：补齐 `-rpath-link` 后编译

    g++ ... -o demo_fixed demo.cpp \
      ... \
      -Wl,-rpath-link,$MX_INDEX_INSTALL_PATH/host/lib \
      -Wl,-rpath-link,$ASCEND_HOME_PATH/lib64 \
      -Wl,-rpath-link,/usr/local/Ascend/driver/lib64 \
      -Wl,-rpath-link,/usr/local/Ascend/driver/lib64/driver \
      -Wl,-rpath-link,/usr/local/faiss/lib \
      -Wl,-rpath-link,/opt/OpenBLAS/lib \
      -lfaiss -lascendfaiss -lopenblas -lc_sec -lascendcl -lascend_hal -lascendsearch -lock_hmm
    echo "LINK_STATUS=${PIPESTATUS[0]}"

- **预期**：链接通过
- **实际**：成功，`LINK_STATUS=0`，生成 `demo_fixed`（14352 B）；`ldd` 在设置 `LD_LIBRARY_PATH` 后 `not found` 计数为 0
- **截图**：`08_step4_link_ok.png`（待补拍）

### 步骤 5：运行 demo

    export LD_LIBRARY_PATH=/usr/local/Ascend/mxIndex/host/lib:/usr/local/Ascend/cann/lib64:\
    /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/faiss/lib:/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
    timeout 240 ./demo_fixed

- **预期**：打印 add/search 结果并正常结束
- **实际**：失败，进程 exit 255：

      Generating random numbers start!
      Exception caught: Error in static bool ascend::AscendOperatorManager::Init(std::string) at
      ascenddaemon/utils/AscendUtils.cpp:193: Error: 'retCode == ACL_ERROR_NONE ||
      retCode == ACL_ERROR_REPEAT_INITIALIZE' failed: ACL error 100014

- **截图**：`09_step5_run_fail.png`（待补拍）

### 步骤 6：定位 100014 与算子模型

    grep -rn "100014" /usr/local/Ascend/cann/include/acl/acl_base_rt.h
    ls -la /usr/local/Ascend/mxIndex/modelpath/
    strings op_models/0_TopkFlat_*.om | grep -oiE "ascend(310p|910b[1-4]|310|910)[a-z0-9]*" | sort -u

- **预期**：确认错误含义与模型来源
- **实际**：`100014 = ACL_ERROR_READ_MODEL_FAILURE`；`modelpath/` 为**空目录**；`tools/op_models/` 里有 31 个 `.om`，但其 SoC 为 **Ascend910B3**，而本机芯片是 **910B4-1**，即模型与芯片不匹配
- **截图**：`10_step6_model_soc.png`（待补拍）

### 步骤 7：按 910B4 重新生成算子

    export ASCEND_HOME=/usr/local/Ascend
    export ASCEND_VERSION=cann-9.2.0
    cd /usr/local/Ascend/mxIndex/tools
    python3 run_generate_model.py -m Flat -t 910B4 -p true -pool 8

- **预期**：在 `op_models/` 生成 910B4 的 `.om`，复制到 `modelpath/` 后 demo 可运行
- **实际**：生成失败。逐层排除后依次暴露：`ASCEND_HOME`/`ASCEND_VERSION` 未设置导致 `atc` 找不到；`ASCEND_OPP_PATH` 拼接错误（E20002）；Python 依赖缺失，`atc` 报 `EC0010: No module named 'numpy' / 'decorator' / 'attr'`。用 apt 补齐 `python3-numpy(1.21.5)`、`python3-decorator`、`python3-sympy`、`python3-psutil`、`python3-cffi`、`python3-attr` 等后，`atc` 初始化阶段不再报缺模块，但该轮生成仍在 18 s 内失败，未取得最终成功产物
- **截图**：`11_step7_gen_fail.png`（待补拍）

### 步骤 8：用户自行修复并跑通

- **操作**：<待补充：你实际改的那一步（推测为 910B4 算子生成/替换后重跑）>
- **预期**：demo 正常结束，输出检索耗时
- **实际**：通过（用户确认）
- **截图**：`12_step8_run_ok.png`（待补拍）

## 5 执行结果

- **结论**：通过
- **判据**：demo 正常结束、输出完整且无异常栈（最终由用户在本会话外完成修复后达成）

关键输出（修复前失败日志原文，用于定位；成功后日志待替换）：

    Generating random numbers start!
    Exception caught: Error in static bool ascend::AscendOperatorManager::Init(std::string) at
    ascenddaemon/utils/AscendUtils.cpp:193: Error: 'retCode == ACL_ERROR_NONE ||
    retCode == ACL_ERROR_REPEAT_INITIALIZE' failed: ACL error 100014

关键指标：<待填，如 QPS / 单次 search 耗时 / add 耗时，注明口径与数据规模 100 万 × 512 维、topK=10>

## 6 发现的问题

> 若发现问题，请提交 issue 并在本节列出链接；未发现则写「本次未发现问题」并说明覆盖范围。

| 问题 | 提交 issue |
| --- | --- |
| 按文档命令编译 demo，链接期报 `libock_vsa_hpp.so.1`/`libock_hcps_pier.so.1` not found 及 6 个 `ock::`/`acl*` 未定义符号；库实际存在于 `-L` 目录内，因 `-L` 不覆盖间接依赖（需 `-Wl,-rpath-link`） | [#待提交](<待填>) |
| 运行 demo 固定失败于 ACL 100014（`READ_MODEL_FAILURE`），根因是 `modelpath` 内算子为 Ascend910B3、芯片为 910B4-1，报错未提示芯片型号与模型型号 | [#待提交](<待填>) |
| 出厂 `modelpath/` 为空，而生成脚本输出到 `tools/op_models/`，文档未说明"生成后需复制到 modelpath"，两者无衔接 | [#待提交](<待填>) |
| `atc` 生成算子依赖 `numpy`/`decorator`/`attr` 未预装，且需手工设置 `ASCEND_HOME`/`ASCEND_VERSION`；报错为 EC0010 并伴随 TBE 初始化长栈，指向性差 | [#待提交](<待填>) |

## 7 总结

1. 链接失败不是缺库：`libascendfaiss.so` 的 3 个 `libock_*` 依赖就在 `$MX_INDEX_INSTALL_PATH/host/lib` 内，补 `-Wl,-rpath-link,<dir>` 后一次通过，`LINK_STATUS=0`；同一原因还掩盖了 `libacl_op_compiler.so`、`libmsprofiler.so`、`libascend_dump.so` 三个依赖。
2. 运行失败不是算子路径问题：`100014` 的真实含义是读模型失败，`modelpath` 里的 31 个算子按 `Ascend910B3` 编译，与本机 `910B4-1` 不匹配，需按 910B4 重新生成。
3. 算子生成链路有三个前置条件文档未交代：`ASCEND_HOME`/`ASCEND_VERSION` 需手工设置、`atc` 需 numpy 等 Python 依赖、生成产物需从 `tools/op_models` 复制到 `modelpath`；任缺一项都表现为"生成失败"或"运行 100014"。

## 8 体验总结

| 维度 | 评价 | 说明 |
| --- | --- | --- |
| 文档易懂度 | 一般 | 编译命令、算子生成、模型放置三段是断开的：按文档顺序执行会在链接和首次运行两处卡住，且都没有对应说明 |
| 流程顺畅度 | 差 | 从"编译"到"首次跑通"需要跨 4 类环境问题（链接参数、SoC 版本、Python 依赖、产物路径），最易出错的是算子 SoC 版本 |
| 报错可读性 | 差 | `ld` 只说 "not found (try using -rpath or -rpath-link)"，未点明 `-L` 不适用间接依赖；`100014` 只有数字，不打印期望/实际 SoC 与模型路径；`atc` 的 EC0010 埋在大段 TBE 栈里 |
| 整体评价 | 2 / 5 | 库与样例本身可用，但环境准备与算子生成环节的文档缺口导致排障成本远高于验证本身 |

- **最耗时的一步**：步骤 7（按 910B4 重新生成算子），约 40 分钟，主要消耗在逐层排查 `atc` 的环境变量与 Python 依赖
- **最值得改进的点**：在快速入门中增加「链接参数模板（含 `-Wl,-rpath-link`）」与「算子生成自检清单（`-t <芯片型号>`、依赖预装、产物复制到 modelpath）」，并让程序在 SoC 不匹配时直接打印芯片与模型型号
