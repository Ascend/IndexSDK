# 昇腾 Index SDK

> [English](./OVERVIEW.md) | 中文

## 快速参考
- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/IndexSDK/issues)
  - [IndexSDK 代码](https://gitcode.com/Ascend/IndexSDK)
  - [IndexSDK API 参考](../docs/zh/api/README.md)
  - [IndexSDK 文档](https://www.hiascend.com/document/detail/zh/mindsdk/730/indexn/indexug/mxindexfrug_0002.html)
  - [AscendHub镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/indexsdk/)
  - [Ascend社区](https://www.hiascend.com/)

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范

Tag 遵循以下格式：

```bash
<indexsdk版本>-<芯片系列>-<操作系统>-<python版本>
```

| 字段         | 示例值                          | 说明             |
| ------------ | ------------------------------- | ---------------- |
| `indexsdk版本`   | `26.0.0`              | Index SDK 版本号      |
| `芯片系列`   | `910b`、`a3`、`310p`            | 目标昇腾芯片系列 |
| `操作系统`   | `ubuntu22.04`、`openeuler24.03` | 基础操作系统     |
| `python版本` | `py3.11`    | Python 版本      |



### CANN 9.0.0 + 26.0.0 Index SDK镜像

| Tag                                | Dockerfile                                                   | 镜像内容        |
| ---------------------------------- | ------------------------------------------------------------ | --------------- |
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](./9.0.0-beta-910b-ubuntu22.04-py3.11/Dockerfile.910b.ubuntu) | toolkit + Index SDK |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](./9.0.0-beta-310p-ubuntu22.04-py3.11/Dockerfile.310p.ubuntu)      | toolkit + Index SDK |
| `26.0.0-a3-ubuntu22.04-py3.11` | [Dockerfile](./9.0.0-beta-a3-ubuntu22.04-py3.11/Dockerfile.a3.ubuntu)         | toolkit + Index SDK |
| `26.0.0-910b-openeuler24.03-py3.11`    | [Dockerfile](./9.0.0-beta-910b-openeuler24.03-py3.11/Dockerfile.910b.openeuler) | toolkit + Index SDK |
| `26.0.0-310p-openeuler24.03-py3.11`    | [Dockerfile](./9.0.0-beta-310p-openeuler24.03-py3.11/Dockerfile.310p.openeuler)      | toolkit + Index SDK |
| `26.0.0-a3-openeuler24.03-py3.11` | [Dockerfile](./9.0.0-beta-a3-openeuler24.03-py3.11/Dockerfile.a3.openeuler)         | toolkit + Index SDK |
---

## 快速开始

### 前置要求（可选）

#### 安装驱动

主机上必须安装与容器内 CANN 版本兼容的昇腾 NPU 驱动。请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。

---

### 运行 Index 容器

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
    -it ascend/index:tag bash
```

### 如何本地构建

```bash
docker build -t {your_repo}/index:latest -f Dockerfile .
```

### 如何二次开发

```bash
# 以 CANN 镜像为基础镜像，叠加用户软件
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

RUN apt update -y && \
    apt install gcc ...

...
```

---

## 支持的硬件

| 芯片系列  | 产品示例                        | 架构           |
| --------- | ------------------------------- | -------------- |
| 昇腾 910B | Atlas 800I A2                   | ARM64 / x86_64 |
| 昇腾 A3   | Atlas 800I A3                   | ARM64 / x86_64 |
| 昇腾 310P | Atlas 300I Pro、Atlas 300V Pro  | ARM64 / x86_64 |

---

## 许可证

查看这些镜像中包含的 CANN 和 Mind 系列软件的[许可证信息](https://github.com/Ascend/cann-container-image/blob/main/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
