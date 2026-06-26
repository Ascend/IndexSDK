# Index SDK

- [Latest Updates](#latest-updates)
- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Version Description](#version-description)
- [Environment Deployment](#environment-deployment)
- [Build Process](#build-process)
- [Quick Start](#quick-start)
- [Features](#features)
- [Security Statement](#security-statement)
- [Branch Maintenance Strategy](#branch-maintenance-strategy)
- [Version Maintenance Strategy](#version-maintenance-strategy)
- [License](#license)
- [Contribution Statement](#contribution-statement)
- [Suggestions and Communication](#suggestions-and-communication)

# Latest Updates

- [Dec. 30, 2025]: 🚀 Index SDK is released as open source.

# Introduction

Index SDK is a heterogeneous retrieval acceleration framework based on Faiss for Ascend NPUs. It provides high-performance retrieval for massive datasets in high-dimensional spaces. It uses Faiss-style C++, combines TBE with Ascend C operator development, and supports the Arm and x86_64 platforms.
You can build retrieval systems for specific application scenarios on this framework. For details, see [Introduction](./docs/en/introduction.md).
<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/IndexSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/IndexSDK)

</div>

# Directory Structure

```text
├── build
├── feature_retrieval
├── ivfsp_impl
├── ivfsp_utils
├── vsa_hpp
└── vstar_great_impl
```

# Version Description

The release notes for Index SDK include software version compatibility and the feature changes in each version. For details, see [Release Notes](./docs/en/release_notes.md).

# Environment Deployment

This section describes how to install Index SDK. For details, see [Installation Guide](./docs/en/installation_guide.md).

1. Install the NPU driver, firmware, and CANN.

    | Software Type | Package Name | How to Obtain |
    | ------------ | ---------------------------------------------------- | ------------ |
    | NPU driver | Ascend-hdk-xxx-npu-driver_{version}_linux-{arch}.run | Download from the Ascend Community. |
    | NPU firmware | Ascend-hdk-xxx-npu-firmware_{version}.run | Download from the Ascend Community. |
    | CANN package | Ascend-cann-toolkit_{version}_linux-{arch}.run | Download from the Ascend Community. |
    | Open-mode scenario package | Ascend-cann-device-sdk_{version}_linux-{arch}.run | Apply for access to the commercial-version download in the Ascend Community. |

2. Install dependencies.

    If you run the following `bash build/build.sh` build process, you can skip this step. If the script encounters network issues when downloading OpenBLAS or Faiss, manually download the installation package to the project root directory.

    To install the runtime and build dependencies manually, run `bash build/install_deps.sh`. To install the UT runtime dependencies, run `bash build/install_deps.sh ut`.

3. Install Index SDK.

# Build Process

This section uses the matching packages for CANN 8.3.RC2 as an example to describe how to build Index SDK from source. The NPU driver, firmware, and CANN package can be downloaded from the Ascend Community. For the open-mode scenario package, log in to `https://support.huawei.com`, search for CANN 8.3.RC2, and request the commercial-version download on the relevant page.

1. Run the build process.

    Run the following command to build:

    ```bash
    bash build/build.sh
    ```

2. The generated `.run` package is in the `build/output` directory: `Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run`

3. Run the test cases.

    Run the following command to run the test cases:

    ```bash
    bash build/build.sh ut
    ```

# Quick Start

Index SDK provides a simple example that helps users quickly get started with the retrieval workflow in Index SDK. For details, see [Usage Example](./docs/en/user_guide.md#usage-example).

# Features

- [Full Retrieval](./docs/en/api/full_retrieval.md)
- [Approximate Retrieval](./docs/en/api/approximate_retrieval.md)
- [Attribute Filtering](./docs/en/api/attribute_filtering-based_retrieval.md)
- [Batch Retrieval](./docs/en/api/multi-index_batch_retrieval.md)

# Security Statement

Security requirements: When you use APIs to read a file, ensure that the file owner is you and that the permissions are no greater than 640 to avoid privilege escalation and other security issues. Software code or programs downloaded from external sources may be risky. You are responsible for ensuring their security.

For details, see [Security Hardening](./docs/en/security_hardening.md) and [Appendix](./docs/en/appendix.md).

# Branch Maintenance Strategy

The maintenance phases of version branches are as follows:

| Status | Time | Description |
| -------- | -------- | ------------------------------------------------------------ |
| Planning | 1 to 3 months | Planned features |
| Development | 3 months | Develop new features, fix issues, and release new versions regularly. |
| Maintenance | 3 to 12 months | Regular branches are maintained for 3 months, and long-term support branches are maintained for 12 months. Major bugs are fixed. No new features are merged. Patch versions are released based on the impact of the bug. |
| End of Life (EOL) | N/A | The branch no longer accepts any changes. |

# Version Maintenance Strategy

| Version | Maintenance Strategy | Current Status | Release Date | Next Status | EOL Date |
| -------- | -------- | -------- | ---------------- | ----------------------------- | ---------- |
| master | Long-term support | Development | 2025-12-30 |  | - |

# License

Index SDK is licensed under Mulan PSL v2. The corresponding license text is available in [LICENSE](LICENSE.md).

The documents in the `docs` directory of Index SDK are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

# Contribution Statement

1. Submit bug reports: If you find a vulnerability in Index SDK that is not a security issue, search the Issues list in the Index SDK repository first to check whether the vulnerability has already been reported. If you cannot find it, create a new Issue. If you find a security issue, do not make it public. See the security issue handling process. Bug reports should include complete information.
2. Security issue handling: For security issues in this project, notify the core project team by email for confirmation and handling.
3. Resolve existing issues: You can find issues that need attention by checking the repository Issues list, and you can try to resolve one of them.
4. How to propose a new feature: Use the Feature label in Issues. We regularly review and confirm development.
5. Start contributing:

    1. Fork the repository.
    2. Clone it locally.
    3. Create a development branch.
    4. Run local self-tests. Before submitting, pass all existing unit tests and add new unit tests for the issue you want to fix.
    5. Commit your code.
    6. Open a Pull Request.
    7. Review the code. Revise the code according to the review comments and push the updates again. This process may take multiple rounds.
    8. When your PR receives enough reviewer approvals, the committer performs the final review.
    9. After the review and tests pass, CI merges your PR into the main branch of the project.

# Suggestions and Communication

Everyone is welcome to contribute to the community. Before contributing, please sign the Contributor License Agreement (CLA). If you have any questions or suggestions, submit them through [GitCode Issues](https://gitcode.com/Ascend/IndexSDK/issues), and we will reply as soon as possible. Thank you for your support.
