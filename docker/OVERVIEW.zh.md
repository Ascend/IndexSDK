# Index SDK

> [English](https://gitcode.com/Ascend/IndexSDK/blob/master/docker/OVERVIEW.md) | 中文

## 1.快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/IndexSDK/issues)
  - [IndexSDK 代码](https://gitcode.com/Ascend/IndexSDK)
  - [IndexSDK API 参考](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/api/README.md)
  - [IndexSDK 文档](https://www.hiascend.com/document/detail/zh/mindsdk/730/indexn/indexug/mxindexfrug_0002.html)
  - [镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/indexsdk/)
  - [社区](https://www.hiascend.com/)

## 2.支持的 Tags 及 Dockerfile 链接

### 2.1 Tag 规范

Tag 遵循以下格式：

```bash
<indexsdk版本>-<芯片系列>-<操作系统>-<python版本>
```

| 字段         | 示例值                          | 说明             |
| ------------ | ------------------------------- | ---------------- |
| `indexsdk版本`   | `26.0.0`              | Index SDK 版本号      |
| `芯片系列`   | `950`、`910`、`a3`、`atlas 800`            | 目标芯片系列 |
| `操作系统`   | `ubuntu22.04`、`openeuler24.03` | 基础操作系统     |
| `python版本` | `py3.11`    | Python 版本      |

### 2.2 CANN 9.0.0 + 26.0.0 Index SDK镜像

| Tag                                | Dockerfile                                                   | 镜像内容        |
| ---------------------------------- | ------------------------------------------------------------ | --------------- |
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-910b-ubuntu22.04-py3.11/Dockerfile.910b.ubuntu) | toolkit + Index SDK |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-310p-ubuntu22.04-py3.11/Dockerfile.310p.ubuntu)      | toolkit + Index SDK |
| `26.0.0-a3-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-a3-ubuntu22.04-py3.11/Dockerfile.a3.ubuntu)         | toolkit + Index SDK |
| `26.0.0-950-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-950-ubuntu22.04-py3.11/Dockerfile.950.ubuntu)         | toolkit + Index SDK |
| `26.0.0-910-openeuler24.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-910-openeuler24.04-py3.11/Dockerfile.910.openeuler) | toolkit + Index SDK |
| `26.0.0-310p-openeuler24.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-310p-openeuler24.04-py3.11/Dockerfile.310p.openeuler)      | toolkit + Index SDK |
| `26.0.0-a3-openeuler24.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-a3-openeuler24.04-py3.11/Dockerfile.a3.openeuler)         | toolkit + Index SDK |
| `26.0.0-950-openeuler24.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/26.0.0-950-openeuler24.04-py3.11/Dockerfile.950.openeuler)         | toolkit + Index SDK |

## 3.快速开始

### 3.1 前置要求（可选）

#### 3.1.1 安装驱动

主机上必须安装与容器内 CANN 版本兼容的NPU 驱动。请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。

---

### 3.2 运行 Index 容器

#### 3.2.1 手动挂载设备

- 设备挂载 ：通过 --device 参数将宿主机的设备文件映射到容器中，确保容器能够访问指定的硬件资源。/dev/davinci为NPU加速卡（按需挂载），/dev/davinci_manager, /dev/devmm_svm， /dev/hisi_hdc为NPU管理设备（全部挂载）。

- 驱动与工具链挂载 ：将宿主机上的驱动文件和工具链目录（如 /usr/local/Ascend/driver 和 /usr/local/bin/npu-smi）以只读方式挂载到容器中，保证容器内的运行环境与宿主机一致。 以下样例代码中，/dev/davinci1 表示挂载 1 号设备。

- -ti后接对应的镜像标签，例如：-it swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-910b-ubuntu22.04-py3.11

```bash
docker run \
    --name index_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it atlas/index:tag bash
```

#### 3.2.2 执行npu-smi info命令检查驱动是否挂载正常

```bash
npu-smi info #正常显示npu卡信息无报错
```

#### 3.2.3 生成算子

```bash
cd /usr/local/Ascend/mxIndex/ops
./custom_opp_*.run
cd ../tools
python3 aicpu_generate_model.py -t npu-type
python3 flat_generate_model.py -t npu-type -d 512
mv op_models/* $MX_INDEX_MODELPATH
```

算子生成可以参考：[算子生成](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/user_guide.md#%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90%E4%BB%8B%E7%BB%8D)

#### 3.2.4 编译入门用例

[demo用例](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/user_guide.md#%E4%BD%BF%E7%94%A8%E6%A0%B7%E4%BE%8B)

```bash
export MX_INDEX_INSTALL_PATH=/usr/local/Ascend/mxIndex

g++ --std=c++11 -fPIC -fPIE -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O3 -Wl,-z,relro,-z,now,-z,noexecstack -s -pie \
  -o demo demo.cpp \
  -I$MX_INDEX_INSTALL_PATH/include \
  -I/usr/local/faiss/include \
  -I/usr/local/Ascend/driver/include \
  -I/opt/OpenBLAS/include \
  -L$MX_INDEX_INSTALL_PATH/host/lib \
  -L/usr/local/faiss/lib \
  -L/usr/local/Ascend/driver/lib64 \
  -L/usr/local/Ascend/driver/lib64/driver \
  -L/opt/OpenBLAS/lib \
  -L$ASCEND_HOME_PATH/lib64 \
  -lfaiss -lascendfaiss -lopenblas -lc_sec -lascendcl -lascend_hal -lascendsearch -lock_hmm
```

- MX_INDEX_INSTALL_PATH：Index SDK 安装路径，默认值为 /usr/local/Ascend/mxIndex
- ASCEND_HOME_PATH：Toolkit 安装路径，默认值为 /usr/local/Ascend/cann

#### 3.2.5 运行入门用例

```bash
./demo
```

### 3.3 如何本地构建

```bash
docker build -t {your_repo}/index:latest -f Dockerfile .
```

---

## 4. 支持的硬件

| 芯片系列  | 产品示例                        | 架构           |
| --------- | ------------------------------- | -------------- |
| Atlas 950  | A5                              | ARM64 / x86_64 |
| Atlas 910  | Atlas 800I A2                   | ARM64 / x86_64 |
| Atlas A3   | Atlas 800I A3                   | ARM64 / x86_64 |
| Atlas 800 | Atlas 300I Pro、Atlas 300V Pro  | ARM64 / x86_64 |

---

## 5. 许可证

查看这些镜像中包含的 CANN 和 Mind 系列软件的[许可证信息](https://github.com/Ascend/cann-container-image/blob/main/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
