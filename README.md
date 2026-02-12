# IndexSDK

- [最新消息](#最新消息)
- [简介](#简介)
- [目录结构](#目录结构)
- [版本说明](#版本说明)
- [环境部署](#环境部署)
- [编译流程](#编译流程)
- [快速入门](#快速入门)
- [功能介绍](#功能介绍)
- [安全声明](#安全声明)
- [分支维护策略](#分支维护策略)
- [版本维护策略](#版本维护策略)
- [License](#License)
- [贡献声明](#贡献声明)
- [建议与交流](#建议与交流)

# 最新消息

- [2025.12.30]: 🚀 INDEXSDK 开源发布

# 简介

Index SDK是基于Faiss开发的昇腾NPU异构检索加速框架，针对高维空间中的海量数据，提供高性能的检索，采用与Faiss风格一致的C++语言，结合TBE，Ascendc算子开发，支持ARM和x86_64平台。
用户可以在此框架上实现面向应用场景的检索系统。更多详情可查看[简介](./docs/zh/introduction.md)。
<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/IndexSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/IndexSDK)

</div>

# 目录结构

``` 
├── build
├── feature_retrieval
├── ivfsp_impl
├── ivfsp_utils
├── vsa_hpp
└── vstar_great_impl
```

# 版本说明

Index SDK的版本说明包括软件版本配套关系和每个版本的特性变更说明，详情可查看[版本说明](./docs/zh/release_notes.md)。

# 环境部署

介绍Index SDK的安装方式，详情可查看[安装指南](./docs/zh/installation_guide.md)。


1. 安装NPU驱动固件和CANN

	| 软件类型     | 软件包名称                                           | 获取方式     |
	| ------------ | ---------------------------------------------------- | ------------ |
	| NPU驱动      | Ascend-hdk-xxx-npu-driver_{version}_linux-{arch}.run | 昇腾社区下载 |
	| NPU固件      | Ascend-hdk-xxx-npu-firmware_{version}.run            | 昇腾社区下载 |
	| CANN软件包   | Ascend-cann-toolkit_{version}_linux-{arch}.run       | 昇腾社区下载 |
	| 开放态场景包 | Ascend-cann-device-sdk_{version}_linux-{arch}.run    | 昇腾社区商用版资源申请 |

2. 安装依赖

	若执行后续 `bash build/build.sh` 编译流程可跳过此步骤，脚本下载OpenBLAS或faiss时遇到网络问题，可手动下载安装包到项目根目录下。

	手动安装运行和构建依赖可执行 `bash build/install_deps.sh`，安装ut运行依赖可执行 `bash build/install_deps.sh ut`。

3. 安装Index SDK

# 编译流程

本节以CANN 8.3.RC2相关配套为例，介绍如何通过源码编译生成Index SDK，其中NPU驱动、固件和CANN软件包可以通过昇腾社区下载，开放态场景包可以通过登录 ```https://support.huawei.com``` 搜索CANN 8.3.RC2，在相关页面申请商业版下载。

1. 执行编译
	
	执行以下命令编译：
    ```bash
    bash build/build.sh
	```

2. 生成的 run 包在 ```build/output``` 下：```Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run```

3. 执行测试用例

	执行以下命令运行测试用例：
	```bash
	bash build/build.sh ut
	```

# 快速入门

Index SDK提供了一个简单的样例，帮助用户快速体验运用Index SDK进行检索的流程。详情可参考[使用样例](./docs/zh/user_guide.md#使用样例)。

# 功能介绍

- [全量检索](./docs/zh/api/full_retrieval.md)
- [近似检索](./docs/zh/api/approximate_retrieval.md)
- [属性过滤](./docs/zh/api/attribute_filtering-based_retrieva.md)
- [批量检索](./docs/zh/api/multi-index_batch_retrieval.md)


# 安全声明

安全要求：使用API读取文件时，用户需要保证该文件的owner必须为自己，且权限不大于640，避免发生提权等安全问题。外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。

更多详情请查看[安全加固](./docs/zh/security_hardening.md)与[附录](./docs/zh/appendix.md)。

# 分支维护策略
 
版本分支的维护阶段如下：
 
| 状态                | 时间     | 说明                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| 计划                | 1-3个月  | 计划特性                                                     |
| 开发                | 3个月    | 开发新特性并修复问题，定期发布新版本                         |
| 维护                | 3-12个月 | 常规分支维护3个月，长期支持分支维护12个月。对重大BUG进行修复，不合入新特性，并视BUG的影响发布补丁版本 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                         |
 
# 版本维护策略
 
| 版本     | 维护策略 | 当前状态 | 发布日期         | 后续状态                      | EOL日期    |
| -------- | -------- | -------- | ---------------- | ----------------------------- | ---------- |
| master   | 长期支持 | 开发     | 2025-12-30       |                               | -          |

# License
IndexSDK以Mulan PSL v2许可证许可，对应许可证文本可查阅[LICENSE](LICENSE.md)。

IndexSDK docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

# 贡献声明
1. 提交错误报告：如果您在Index SDK中发现了一个不存在安全问题的漏洞，请在Index SDK仓库中的Issues中搜索，以防该漏洞已被提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
	1. Fork本项目的仓库。
	2. Clone到本地。
	3. 创建开发分支。
	4. 本地自测，提交前请通过所有的已经单元测试，以及为您要解决的问题新增单元测试。
	5. 提交代码。
	6. 新建Pull Request。
	7. 代码检视，您需要根据评审意见修改代码，并再次推送更新。此过程可能会有多轮。
	8. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
	9. 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

# 建议与交流
欢迎大家为社区做贡献。贡献前，请先签署开放项目贡献者许可协议（CLA）。如果有任何疑问或建议，请提交[GitCode Issues](https://gitcode.com/Ascend/IndexSDK/issues)，我们会尽快回复。 感谢您的支持。