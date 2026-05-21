# Index SDK

> English | [中文](https://gitcode.com/Ascend/IndexSDK/blob/master/docker/OVERVIEW.zh.md)

## 1.Quick Reference

- Where to get help

  - [Issue Feedback](https://gitcode.com/Ascend/IndexSDK/issues)
  - [IndexSDK Code](https://gitcode.com/Ascend/IndexSDK)
  - [IndexSDK API Reference](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/api/README.md)
  - [IndexSDK Documentation](https://www.hiascend.com/document/detail/zh/mindsdk/730/indexn/indexug/mxindexfrug_0002.html)
  - [Image Repository](https://www.hiascend.com/developer/ascendhub/detail/indexsdk/)
  - [Community](https://www.hiascend.com/)

## 2.Supported Tags and Dockerfile Links

### 2.1 Tag Naming Convention

Tags follow this pattern:

```bash
<indexsdk_version>-<chip_series>-<os>-<python_version>
```

| Field            | Example Values                  | Description               |
| ---------------- | ------------------------------- | ------------------------- |
| `indexsdk_version` | `26.0.0`   | IndexSDK version              |
| `chip_series`    | `950`, `910`, `a3`, `atlas 800`            | Target chip family |
| `os`             | `ubuntu22.04`, `openeuler24.03` | Base operating system     |
| `python_version` | `py3.11`    | Python version            |

### 2.2 CANN 9.0.0 + 26.0.0 Index SDK Image

| Tag                                | Dockerfile                                                   | Image Content        |
| ---------------------------------- | ------------------------------------------------------------ | --------------- |
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.910b.ubuntu) | toolkit + Index SDK |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.310p.ubuntu)      | toolkit + Index SDK |
| `26.0.0-a3-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.a3.ubuntu)         | toolkit + Index SDK |
| `26.0.0-950-ubuntu22.04-py3.11` | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.950.ubuntu)         | toolkit + Index SDK |
| `26.0.0-910b-openeuler24.03-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.910b.openeuler) | toolkit + Index SDK |
| `26.0.0-310p-openeuler24.03-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.310p.openeuler)      | toolkit + Index SDK |
| `26.0.0-a3-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.a3.openeuler)         | toolkit + Index SDK |
| `26.0.0-950-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/IndexSDK/tree/master/docker/Dockerfile.950.openeuler)         | toolkit + Index SDK |

## 3.Quick Start

### 3.1 Prerequisites (optional)

#### 3.1.1 Install Driver

An Atlas NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.

---

### 3.2 Running a Index Container

#### 3.2.1 Mount Devices Manually

- Device Mounting: Map host device files to the container using the --device parameter to ensure the container can access specified hardware resources. /dev/davinci is the NPU accelerator card (mount as needed), while /dev/davinci_manager, /dev/devmm_svm, and /dev/hisi_hdc are NPU management devices (mount all).

- Driver and Toolchain Mounting: Mount driver files and toolchain directories (such as /usr/local/Ascend/driver and /usr/local/bin/npu-smi) from the host to the container in read-only mode to ensure consistent runtime environment. In the example code below, /dev/davinci1 represents mounting device 1.

- After -ti, specify the corresponding image tag, for example: -it swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-310p-ubuntu22.04-py3.11

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

#### 3.2.2 Check Driver Mount Status

```bash
npu-smi info # Should display NPU card information without errors
```

#### 3.2.3 Generate Operators

```bash
cd /usr/local/Ascend/mxIndex/ops
./custom_opp_*.run
cd ../tools
python3 aicpu_generate_model.py -t npu-type
python3 flat_generate_model.py -t npu-type -d 512
mv op_models/* $MX_INDEX_MODELPATH
```

For more details on operator generation, refer to: [Operator Generation](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/user_guide.md#%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90%E4%BB%8B%E7%BB%8D)

#### 3.2.4 Compile Demo

[Demo Example](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/user_guide.md#%E4%BD%BF%E7%94%A8%E6%A0%B7%E4%BE%8B)

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

- MX_INDEX_INSTALL_PATH: Index SDK installation path, default is /usr/local/Ascend/mxIndex
- ASCEND_HOME_PATH: Toolkit installation path, default is /usr/local/Ascend/cann

#### 3.2.5 Run Demo

```bash
./demo
```

### 3.3 How to Build Locally

```bash
docker build -t {your_repo}/index:latest -f Dockerfile .
```

---

## 4. Supported Hardware

| Chip Series | Product Examples                | Architecture   |
| ----------- | ------------------------------- | -------------- |
| Atlas 950  | A5                              | ARM64 / x86_64 |
| Atlas 910  | Atlas 800I A2                   | ARM64 / x86_64 |
| Atlas A3   | Atlas 800I A3                   | ARM64 / x86_64 |
| Atlas 300I Pro | Atlas 300I Pro, Atlas 300V Pro  | ARM64 / x86_64 |

---

## 5. License

View the [license information](https://github.com/Ascend/cann-container-image/blob/main/LICENSE) for CANN and MindSeries software included in these images.

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
