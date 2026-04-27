# Ascend Index SDK

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- Where to get help

  - [Issue Feedback](https://gitcode.com/Ascend/IndexSDK/issues)
  - [IndexSDK Code](https://gitcode.com/Ascend/IndexSDK)
  - [IndexSDK API Reference](../docs/zh/api/README.md)
  - [IndexSDK Documentation](https://www.hiascend.com/document/detail/zh/mindsdk/730/indexn/indexug/mxindexfrug_0002.html)
  - [AscendHub Image Repository](https://www.hiascend.com/developer/ascendhub/detail/indexsdk/)
  - [Ascend Community](https://www.hiascend.com/)

## Supported Tags and Dockerfile Links

### Tag Naming Convention

Tags follow this pattern:

```bash
<indexsdk_version>-<chip_series>-<os>-<python_version>
```

| Field            | Example Values                  | Description               |
| ---------------- | ------------------------------- | ------------------------- |
| `indexsdk_version` | `26.0.0`   | IndexSDK version              |
| `chip_series`    | `910b`, `a3`, `310p`            | Target Ascend chip family |
| `os`             | `ubuntu22.04`, `openeuler24.03` | Base operating system     |
| `python_version` | `py3.11`    | Python version            |

### CANN 9.0.0 + 26.0.0 Index SDK Image

| Tag                                | Dockerfile                                                   | Image Content        |
| ---------------------------------- | ------------------------------------------------------------ | --------------- |
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](./9.0.0-beta-910b-ubuntu22.04-py3.11/Dockerfile.910b.ubuntu) | toolkit + Index SDK |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](./9.0.0-beta-310p-ubuntu22.04-py3.11/Dockerfile.310p.ubuntu)      | toolkit + Index SDK |
| `26.0.0-a3-ubuntu22.04-py3.11` | [Dockerfile](./9.0.0-beta-a3-ubuntu22.04-py3.11/Dockerfile.a3.ubuntu)         | toolkit + Index SDK |
| `26.0.0-910b-openeuler24.03-py3.11`    | [Dockerfile](./9.0.0-beta-910b-openeuler24.03-py3.11/Dockerfile.910b.openeuler) | toolkit + Index SDK |
| `26.0.0-310p-openeuler24.03-py3.11`    | [Dockerfile](./9.0.0-beta-310p-openeuler24.03-py3.11/Dockerfile.310p.openeuler)      | toolkit + Index SDK |
| `26.0.0-a3-openeuler24.03-py3.11` | [Dockerfile](./9.0.0-beta-a3-openeuler24.03-py3.11/Dockerfile.a3.openeuler)         | toolkit + Index SDK |

---

## Quick Start

### Prerequisites (optional)

#### Install Driver

An Ascend NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.

---

### Running a Index Container

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

### How to build

```bash
docker buildx build -t {your_repo}/index:latest -f Dockerfile .
```

### Development

```bash
# Add required software by developer
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910b-ubuntu22.04-py3.11

RUN apt update -y && \
    apt install gcc ...

...
```

---

## Supported Hardware

| Chip Series | Product Examples                | Architecture   |
| ----------- | ------------------------------- | -------------- |
| Ascend 910B | Atlas 800I A2                   | ARM64 / x86_64 |
| Ascend A3   | Atlas 800I A3                   | ARM64 / x86_64 |
| Ascend 310P | Atlas 300I Pro, Atlas 300V Pro  | ARM64 / x86_64 |

---

## License

View the [license information](https://github.com/Ascend/cann-container-image/blob/main/LICENSE) for CANN and MindSeries software included in these images.

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
