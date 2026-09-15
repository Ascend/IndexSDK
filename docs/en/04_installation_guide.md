# Installation and Deployment

## Installation Instructions

Index SDK supports three installation methods: [offline installation](#offline-installation), [image installation](#image-installation), and [source installation](#source-installation).

If you use offline installation or source installation, first [install the required dependencies](#dependency-installation). If you use image installation, skip this step.

**Precautions**

- For third-party open-source software, if a vulnerability exists in a given version, promptly fix and update it according to the corresponding instructions for the open-source version.
- (Optional) For installation of Ascend Docker Runtime, see [Ascend Docker Runtime](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.1.0/docs/zh/scheduling/05_developer_guide/00_installation_deployment/00_manual_installation/02_ascend_docker_runtime.md) under **Developer Guide** > **Installation and Deployment** > **Manual Installation** in the *MindCluster Cluster Scheduling User Guide*.
- (Optional) Index SDK supports virtualized environments. You can deploy and run Index SDK services in a virtualized environment. For details about environment deployment, see section **[Virtual Instance Feature Guide](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.1.0/docs/zh/scheduling/04_usage/02_virtual_instance/menu_virtual_instance.md)** under **Usage** in the *MindCluster Cluster Scheduling User Guide*.

## Dependency Installation

### Installing the NPU Driver Firmware and CANN

Refer to [CANN Quick Installation](https://www.hiascend.com/en/cann/download), select CANN 9.1.0 and the corresponding driver version, install the Ascend NPU driver firmware and CANN software (including the Toolkit and ops packages), and configure the environment variables.

> [!NOTE]
> The user installing CANN and the user installing Index SDK must be the same. A regular user is recommended.

### Installing Other Dependencies

#### Other Dependencies

| Dependency Name | Recommended Version | Acquisition Recommendation |
| ----------------- | --------------------- | ---------------------------- |
| gcc | >=7.5.0 | You are advised to obtain the source package and compile and install it. |
| cmake | >=3.24.0 | You are advised to install it using a package manager. See the following example installation command.<br>`sudo apt-get install -y cmake`<br>If the version available through the package manager does not meet the minimum version requirement, you can install it from source. |
| Python | 3.9/3.10/3.11/3.12 | You are advised to obtain the source package and compile and install it. |

Run the following commands to check whether GCC, CMake, and other dependencies are installed.

```bash
gcc --version
cmake --version
python3 --version
```

If the following information is returned, the corresponding software is installed. The output is for reference only. Use the actual output as the basis.

```bash
gcc 7.5.0
cmake version 3.24.0
Python 3.9.11
```

#### Python Dependencies

After Python is installed, see the following table for the pip dependency names, recommended versions, and acquisition recommendations.

| Dependency Name | Recommended Version | Acquisition Recommendation |
| ----------------- | --------------------- | ---------------------------- |
| numpy | >=1.25.0 | See the following example installation command.<br>`pip3 install "numpy>=1.25.0"`<br> |
| decorator | >=5.2.1 | See the following example installation command.<br>`pip3 install "decorator>=5.2.1"`<br> |
| sympy | >=1.14 | See the following example installation command.<br>`pip3 install "sympy>=1.14"`<br> |
| cffi | >=1.15.1 | See the following example installation command.<br>`pip3 install "cffi>=1.15.1"`<br> |
| pyyaml | N/A | See the following example installation command.<br>`pip3 install pyyaml`<br> |
| pathlib2 | N/A | See the following example installation command.<br>`pip3 install pathlib2`<br> |
| protobuf | N/A | See the following example installation command.<br>`pip3 install protobuf`<br> |
| scipy | N/A | See the following example installation command.<br>`pip3 install scipy`<br> |
| requests | N/A | See the following example installation command.<br>`pip3 install requests`<br> |
| attrs | N/A | See the following example installation command.<br>`pip3 install attrs`<br> |
| psutil | N/A | See the following example installation command.<br>`pip3 install psutil`<br> |
| faiss-cpu | 1.13.2 | See the following example installation command.<br>`pip3 install faiss-cpu==1.13.2`<br> |

#### Installing OpenBLAS

You are advised to use the corresponding OpenBLAS version. This section only provides installation instructions for OpenBLAS v0.3.10. Follow the actual OpenBLAS version and environment you use.

**Procedure**

1. Download and decompress the OpenBLAS v0.3.10 source package.

    ```bash
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
    tar -xf OpenBLAS-0.3.10.tar.gz
    ```

2. Go to the OpenBLAS directory.

    ```bash
    cd OpenBLAS-0.3.10
    ```

3. Build and install OpenBLAS.

    ```bash
    make FC=gfortran USE_OPENMP=1 -j
    # OpenBLAS is installed to /opt/OpenBLAS by default.
    make install
    # Or run the following command to install it to a specified path.
    # make PREFIX=/your_install_path install
    ```

4. Configure the environment variable for the library search path.

    ```bash
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
    # Configure /etc/profile.
    vim /etc/profile
    # Add export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH to /etc/profile.
    source /etc/profile
    ```

5. Verify that the installation is successful.

    ```bash
    cat /opt/OpenBLAS/lib/cmake/openblas/OpenBLASConfigVersion.cmake | grep 'PACKAGE_VERSION "'
    ```

    If the software version is displayed correctly, the installation is successful.

#### Installing Faiss

**Installation Notes**

- Before installing Faiss, complete the OpenBLAS installation in the preceding section.
- The Index SDK build script builds a single-version business shared library based on Faiss 1.10.x by default. If you need features such as IVFRaBitQ/RaBitQ that depend on Faiss 1.14, you can specify a single-version business shared library based on Faiss 1.14.1. When using it, link to the business shared library, header files, and `libfaiss.so` corresponding to Faiss 1.14.1. If you need to support both Faiss 1.10.x and Faiss 1.14.1, you can enable coexistence of multiple versions of the business shared libraries during the build. If you do not use IVFRaBitQ/RaBitQ features and need to maintain compatibility with legacy environments, you can use the Faiss 1.10.x business shared library.
- You are advised to install different Faiss versions in separate directories, such as `/usr/local/faiss/faiss1.10.0` and `/usr/local/faiss/faiss1.14.1`. Do not switch versions by overwriting `/usr/local/lib/libfaiss.so`. When compiling and running programs, explicitly select the required Faiss version through `-I`, `-L`, and `LD_LIBRARY_PATH`.
- This section only provides installation instructions for Faiss v1.10.0. Follow the actual Faiss version and environment you use.

> [!NOTE]
>
> - On ARM platforms, adapt the Faiss source code according to the gcc version before compiling and installing Faiss.
> - On ARM platforms, some older versions of gcc, such as 4.8.5, do not support direct compilation of Faiss 1.10.0. Some older compiler versions do not support the implementation related to `simdlib_neon.h` either. In this case, use the default SIMD implementation on the CPU. The functions can run normally with this method, but some Index algorithms, such as IVF and SQ, may suffer significant performance degradation. You are advised to use gcc 7.5.0 for compilation and installation. Versions later than gcc 9.5.0 may have compatibility issues.

**Procedure**

1. Download the Faiss source package and decompress it.

    ```bash
    # Faiss 1.10.0
    wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz
    tar -xf v1.10.0.tar.gz && cd faiss-1.10.0/faiss
    ```

2. Create the `install_faiss_sh.sh` script.

    ```bash
    vi install_faiss_sh.sh
    ```

3. Write the following content into the `install_faiss_sh.sh` script.

    ```bash
    # modify source code
    # Step 1: modify the Faiss source code
    arch="$(uname -m)"
    if [ "${arch}" = "aarch64" ]; then
      gcc_version="$(gcc -dumpversion)"
      if [ "${gcc_version}" = "4.8.5" ];then
        sed -i '20i /*' utils/simdlib.h
        sed -i '24i */' utils/simdlib.h
      fi
    fi
    sed -i "149 i\\
        \\
        virtual void search_with_filter (idx_t n, const float *x, idx_t k,\\
                                         float *distances, idx_t *labels, const void *mask = nullptr) const {}\\
    " Index.h
    sed -i "49 i\\
        \\
    template <typename IndexT>\\
    IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids):\\
        index (index),\\
        own_fields (false)\\
    {\\
        this->is_trained = index->is_trained;\\
        this->metric_type = index->metric_type;\\
        this->verbose = index->verbose;\\
        this->d = index->d;\\
        id_map = ids;\\
    }\\
    " IndexIDMap.cpp
    sed -i "30 i\\
        \\
        explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids);\\
    " IndexIDMap.h
    sed -i "217 i\\
      utils/sorting.h
    " CMakeLists.txt
    # modify source code end
    cd ..
    ls
    # Step 2: Faiss build configuration
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    # Step 3: build and install
    cd build && make -j && make install
    ```

4. Press `Esc`, type `:wq!`, and press `Enter` to save and exit the editor.
5. Run the installation script.

    ```bash
    bash install_faiss_sh.sh
    ```

    > [!NOTE]
    > - Compiling Faiss 1.10.0 requires CMake 3.24.0 or later. If CMake reports that the version is too old when compiling Faiss, refer to [CMake error information when compiling Faiss 1.10.0](./07_faq.md#cmake-error-when-compiling-faiss-1100) for a solution.
    > - The default installation directory for Faiss is `/usr/local/lib`. If you need to specify an installation directory, for example, `install_path=/usr/local/faiss/faiss1.10.0`, add the `-DCMAKE_INSTALL_PREFIX=${install_path}` option to the CMake build configuration.
    >
    > ```bash
    > install_path=/usr/local/faiss/faiss1.10.0
    > cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${install_path}
    > ```
    >
    > - When using the IVFRaBitQ/RaBitQ features, you need to install Faiss 1.14.1 separately. You are advised to use an independent installation directory, such as `/usr/local/faiss/faiss1.14.1`, and set `-DCMAKE_INSTALL_PREFIX=/usr/local/faiss/faiss1.14.1` in the build configuration.

6. Configure the environment variable for the system library search path.

   Programs that dynamically link against Faiss need to know where the Faiss shared library is located at runtime. Therefore, add the Faiss library directory to the `LD_LIBRARY_PATH` environment variable.

    ```bash
    # Configure /etc/profile.
    vim /etc/profile
    # Add the following to /etc/profile: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    # /usr/local/lib is the Faiss installation directory. If Faiss is installed in another directory, replace /usr/local/lib with the actual Faiss installation path.
    # In some operating systems and environments, Faiss may be installed in another directory. For example, on CentOS, the path is /usr/local/lib64.
    source /etc/profile
    ```

7. Verify that the installation is successful.

    ```bash
    cat /usr/local/share/faiss/faiss-config-version.cmake |grep 'PACKAGE_VERSION "'
    ```

    If the software version is displayed correctly, the installation is successful.

> [!NOTE]
> If an error occurs after compiling Faiss on openEuler, refer to [undefined reference error returned when linking libfaiss.so](./07_faq.md#undefined-reference-error-when-linking-libfaissso) for a solution.

#### Installing AscendMiniOs (Optional)

In addition to the preceding dependencies, you also need to determine whether to install the **open-state scenario package** based on whether you need to use the ILFlat algorithm.

- If you do not need it, skip this step.
- If you need it, first download the [Ascend-cann-device-sdk installation package](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0).

```bash
unzip Ascend-cann-device-sdk_{version}_linux-{arch}.zip
# The decompression produces CANN-runtime-*-minios.{arch}.run.
./CANN-runtime-*-minios.{arch}.run --devel --install-path=/usr/local/AscendMiniOs
./CANN-runtime-*-minios.{arch}.run --run --install-path=/usr/local/AscendMiniOSRun
```

## Installation Methods

### Offline Installation

Download the Index SDK feature retrieval package (`Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run`) from the [download link](https://www.hiascend.com/en/developer/software/mindsdk/download).

**Installation Notes**

- Users who install and run Index SDK must meet the following requirements:
    - The same user must be used to install and run Index SDK, and this user must also be the user who installed CANN. Otherwise, permission issues may occur when accessing CANN while running generated operators.
    - The user who installs and runs Index SDK is advised to be a regular user. Index SDK depends on shared libraries for low-privilege users in the CANN package. If the program is run as root, there is a security risk that the linked shared libraries could be tampered with by a low-privilege user.
    - The owner of the package directory and the installation target directory must be the installation user.
    - When installing Index SDK, ensure that the `~` directory exists and that the installation user has read and write permissions for the directory.

- Feature retrieval is provided as a binary shared library, and the package is installed to a user-defined local path using the run package.

**Installation Steps**

1. Log in to the installation environment as the package installation user.
2. Upload the package to any path in the installation environment (for example, `/home/work/FeatureRetrieval`) and go to the directory where the package is located.
3. Grant execute permission to the package.

    ```bash
    chmod u+x Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run
    ```

4. Run the following command to verify the package's consistency and integrity.

    ```bash
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --check
    ```

    If the following information is displayed, the package has passed verification.

    ```bash
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
    ```

5. Create the installation path for the package.
    - **If the user does not specify an installation path**, the software is installed to the following default path:
        - If installed as root, the default installation path is `/usr/local/Ascend`.
        - If installed as a non-root user, the default installation path is `${HOME}/Ascend`, where `${HOME}` is the user directory.
    - **If the user wants to specify an installation path**, create the installation path first. For example, to use `/home/work/FeatureRetrieval`:

        ```bash
        mkdir -p /home/work/FeatureRetrieval
        ```

6. Run the installation command.

    ```bash
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=<path> --platform=<npu_type>
    ```

The following table describes the command-line options.

| Option | Description |
| ------ | ----------- |
| `--help \| -h` | Displays help information. |
| `--info` | Displays package build information. |
| `--list` | Displays the file list. |
| `--check` | Checks package integrity. |
| `--quiet \| -q` | Optional. Enables silent installation and reduces interactive output. |
| `--nox11` | Deprecated. Has no actual effect. |
| `--noexec` | Extracts the package to the current directory without executing the installation script. Use it together with `--extract=<path>`, in the form `--noexec --extract=<path>`. |
| `--extract=<path>` | Extracts files in the package to the specified directory. Can be used together with `--noexec`, `--install`, or `--upgrade`. |
| `--tar arg1 [arg2 ...]` | Runs the `tar` command on the package, using parameters following `tar` as command arguments. For example, `--tar xvf` extracts the contents of the run installation package to the current directory. |
| `--version` | Displays the Index SDK version of the installation package. |
| `--install` | Installs the feature retrieval package. |
| `--install-path=<path>` | Optional. Customizes the root installation directory for the feature retrieval package. If not set, the current directory where the command is executed is used by default. The path must start with `/` or `~` and can contain only letters, digits, `-`, `_`, `.`, and `/`.<br>If not specified, the package is installed to the default path:<ul><li>If installed as root, the default installation path is `/usr/local/Ascend`.</li><li>If installed as a non-root user, the default installation path is `${HOME}/Ascend`, where `${HOME}` is the user directory.</li></ul>If this option is used to specify the installation directory, other users must not have write permission to the directory. If a regular user is specified for installation, the owner of the installation directory must be the current installation user. |
| `--upgrade` | Upgrades the feature retrieval package to the Index SDK version included in the package. |
| `--platform=<npu_type>` | Ascend AI Processor type.<ul><li>For <term>Atlas Inference Series products</term>, enter `310P`.</li><li>For Atlas 800I A3 Supernode Servers, enter `A3`.</li><li>For <term>Atlas A2 Inference Series products</term>, run the `npu-smi info` command on a server where the Ascend AI Processor is installed to query it, and remove the last digit from the queried `Name` value. The result is the value of `--platform`.</li></ul> |
| `--faiss-version=<version>` | Optional. For a multi-version run package, selects the Faiss ABI version to be activated after installation. For a single-version run package, verifies whether the version selected by the user is consistent with the version included in the package. Supported values include `1.10`, `1.10.0`, `faiss1.10`, `1.14`, `1.14.1`, and `faiss1.14`. The default value is the default version included in the package. When `1.10` is selected, the business shared libraries and header files built against Faiss 1.10.x are activated. When `1.14` is selected, the business shared libraries and header files built against Faiss 1.14.1 are activated. This option does not need to be set for a single-version run package. If it is set to a version not included in the package, the installation exits with an error. |

> [!NOTE]
> The following options are not shown in the `--help` output. Do not use them directly.
>
> - `--xwin`: Runs in `xwin` mode.
> - `--phase2`: Requires the second-step action to be executed.

**Multi-Version Run Package**

The Index SDK multi-version run package supports selecting different Faiss ABI versions of the business shared libraries to activate during installation:

```bash
# Multi-version run package: activate the Faiss 1.10.x business shared libraries for scenarios that need compatibility with legacy business environments.
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=<path> --platform=<npu_type> --faiss-version=1.10

# Multi-version run package: activate the Faiss 1.14.1 business shared libraries for scenarios that use the IVFRaBitQ/RaBitQ features.
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=<path> --platform=<npu_type> --faiss-version=1.14
```

After the multi-version run package is installed, the package configures the following symbolic links based on `--faiss-version`:

```bash
mxIndex/host/lib/libascendfaiss.so -> faiss1.1x/libascendfaiss.so
mxIndex/host/lib/libascendsearch.so -> faiss1.1x/libascendsearch.so
mxIndex/include/faiss -> faiss1.1x/faiss
mxIndex/include/ascend -> faiss/ascend
```

> [!NOTE]
> The run package only provides or switches the Index SDK business shared libraries and header files. It does not install or replace `libfaiss.so` in the user's environment. When compiling and running applications, add the include and lib directories for the corresponding Faiss version to the build parameters and `LD_LIBRARY_PATH`. Use Faiss 1.14.1 when using the IVFRaBitQ/RaBitQ features. If you do not use the IVFRaBitQ/RaBitQ features and need to maintain compatibility with legacy environments, you can use Faiss 1.10.x.
> A single-version run package contains only one Faiss ABI version. The installation script verifies whether `--faiss-version` is consistent with the version included in the package. If `--faiss-version=1.14` is specified for a Faiss 1.10.x single-version package, or `--faiss-version=1.10` is specified for a Faiss 1.14.1 single-version package, the installation exits with an error.

If the application directly includes Faiss header files or calls Faiss APIs, such as `faiss::read_index`, `faiss::write_index`, or `faiss::IndexIVFRaBitQ`, you must also explicitly select the Faiss version consistent with `--faiss-version` during compilation and runtime. Using `/home/work/FeatureRetrieval` as the installation path as an example:

```bash
# Non-IVFRaBitQ/RaBitQ business scenario, using Faiss 1.10.x
g++ test.cpp -I/home/work/FeatureRetrieval/mxIndex/include -I/usr/local/faiss/faiss1.10.0/include \
    -L/home/work/FeatureRetrieval/mxIndex/host/lib -L/usr/local/faiss/faiss1.10.0/lib \
    -lascendfaiss -lascendsearch -lfaiss
export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib:/usr/local/faiss/faiss1.10.0/lib:$LD_LIBRARY_PATH

# IVFRaBitQ/RaBitQ business scenario, using Faiss 1.14.1
g++ test.cpp -I/home/work/FeatureRetrieval/mxIndex/include -I/usr/local/faiss/faiss1.14.1/include \
    -L/home/work/FeatureRetrieval/mxIndex/host/lib -L/usr/local/faiss/faiss1.14.1/lib \
    -lascendfaiss -lascendsearch -lfaiss
export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib:/usr/local/faiss/faiss1.14.1/lib:$LD_LIBRARY_PATH
```

If the following information is returned, the feature retrieval package has been installed successfully.

```bash
Install package successfully.
```

### Image Installation

Refer to the [Index SDK image repository](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2) to complete the containerized deployment of feature retrieval.

### Source Installation

When compiling and installing from source, in addition to the preceding dependencies, you also need to determine whether to install the **open-state scenario package** based on whether you need to use the ILFlat algorithm.

- If you do not need it, set the `BUILD_ASCENDDEVICE` option in `feature_retrieval/src/ascendfaiss/CMakeLists.txt` to `OFF` before compilation, and comment out line 38, `# ASCEND_MINIOS_HOME`.
- If you need it, first download the [Ascend-cann-device-sdk installation package](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0).

```bash
unzip Ascend-cann-device-sdk_{version}_linux-{arch}.zip
# The decompression produces CANN-runtime-*-minios.{arch}.run.
./CANN-runtime-*-minios.{arch}.run --devel --install-path=/usr/local/AscendMiniOs
./CANN-runtime-*-minios.{arch}.run --run --install-path=/usr/local/AscendMiniOSRun
```

Go to the `build` directory and run the following command to compile:

```bash
bash build.sh
```

- By default, a single-version run package based on Faiss 1.10.x is built using `MULTI_FAISS_PACKAGE=OFF DEFAULT_FAISS_ABI=faiss1.10 bash build/build.sh`. This package contains only the business shared libraries and header files built against Faiss 1.10.x. You do not need to specify `--faiss-version` during installation.
- To build a single-version run package based on Faiss 1.14.1, run `MULTI_FAISS_PACKAGE=OFF DEFAULT_FAISS_ABI=faiss1.14 bash build/build.sh`. This package contains only the business shared libraries and header files built against Faiss 1.14.1 and is intended for scenarios that use the IVFRaBitQ/RaBitQ features.
- To provide both the Faiss 1.10.x and Faiss 1.14.1 business shared libraries in the same run package, run `MULTI_FAISS_PACKAGE=ON DEFAULT_FAISS_ABI=faiss1.10 bash build/build.sh` to build a multi-version run package. During installation, use `--faiss-version` to select the Faiss ABI version to activate. If this parameter is not specified, the version specified by `DEFAULT_FAISS_ABI` during the build is activated by default.

The generated run package is located in the `build/output` directory: `Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run`. Run the corresponding installation command to complete the installation.

## Upgrade

> [!NOTE]
>
> - The upgrade operation involves uninstalling and then reinstalling the installation directory. If other files exist in the directory, they will also be deleted. Before performing the upgrade, ensure that all data has been properly handled.
> - When changing the deployment from the open-state deployment of Index SDK 5.0.RC2 to the standard-state deployment of a version later than 5.0.RC2, [uninstall](#uninstallation) the open-state deployment and then redeploy standard-state feature retrieval.
> - During deployment, link to the shared libraries in the `mxIndex-{version}/host` directory and regenerate the environment variables for the operator and operator model file directories.

Use the following command to upgrade the feature retrieval package. For command-line option descriptions, see the following table.

```bash
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=<npu_type> --install-path=<path>
```

| Option | Description |
| ---------------- | ------------- |
| `--upgrade` | Installation command for upgrading the feature retrieval package, which upgrades feature retrieval to the Index SDK version included in the package. |
| `--platform=<npu_type>` | Corresponding Ascend AI Processor type.<ul><li>For Atlas Inference Series products, enter `310P`.</li><li>For Atlas 800I A3 Supernode Servers, enter `A3`.</li><li>For Atlas A2 Inference Series products, run the `npu-smi info` command on a server where the Ascend AI Processor is installed to query it, and remove the last digit from the queried `Name` value. The result is the value of `--platform`.</li></ul> |
| `--install-path=<path>` | (Optional) Customizes the root installation directory for the feature retrieval package. If not set, the default is `/usr/local/Ascend`. If a custom installation directory is used, you are advised to specify this option during the upgrade. |
| `--faiss-version=<version>` | (Optional) For a multi-version run package, selects the Faiss ABI version to be activated after the upgrade. For a single-version run package, verifies whether the version selected by the user is consistent with the version included in the package. Supported values include `1.10`, `1.10.0`, `faiss1.10`, `1.14`, `1.14.1`, and `faiss1.14`. The default value is the default version included in the package. If you need to continue using the IVFRaBitQ/RaBitQ features after the upgrade, use the Faiss 1.14.1 version run package, or specify `--faiss-version=1.14` in a multi-version run package. This option does not need to be set for a single-version run package. If it is set to a version not included in the package, the upgrade exits with an error. |

If the following information is returned, the feature retrieval package has been upgraded successfully.

```bash
Upgrade package successfully.
```

## Uninstallation

> [!NOTE]
> The uninstallation operation involves deleting the installation directory. If other files exist in the directory, they will also be deleted. Before performing the uninstallation, ensure that all data has been properly handled.
> Operator files must be deleted manually. When uninstalling, also delete the retrieval-related operator files, where `{ASCEND_OPP_PATH}` is the directory specified by the `ASCEND_OPP_PATH` environment variable.
>
> - For Index SDK versions earlier than 5.0.0, the operator file installation directories are `${ASCEND_OPP_PATH}/op_impl` and `${ASCEND_OPP_PATH}/op_proto`.
> - For Index SDK versions 5.0.0 and later, the operator file installation directory is `${ASCEND_OPP_PATH}/vendors/mxIndex`.
> - You can view the specific operator files by running `./custom_opp_*.run --list`.

**Procedure**

1. Go to the installation directory `mxIndex-{version}`.

    ```bash
    cd mxIndex-{version}
    ```

2. Go to the `script` directory.

    ```bash
    cd script
    ```

3. Grant execute permission to `uninstall.sh` and run it to complete the uninstallation.

    ```bash
    chmod u+x uninstall.sh
    ./uninstall.sh
    ```
