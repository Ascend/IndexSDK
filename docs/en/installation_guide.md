# Installation and Deployment<a name="ZH-CN_TOPIC_0000001506414717"></a>

## Installation Instructions<a name="ZH-CN_TOPIC_0000002022551553"></a>

Index SDK supports deployment on bare metal servers and in containers. This document describes deployment on bare metal. If you need to deploy feature retrieval in a Docker environment, follow the instructions in the [Index image repository](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2) to complete the containerized deployment.

For instructions on installing Ascend Docker runtime, refer to the "Install > Installation and Deployment > Manual Installation > [Ascend Docker Runtime](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/installation_guide/03_installation/manual_installation/02_ascend_docker_runtime.md)" section in the *MindCluster Cluster Scheduling User Guide*.

Index SDK now supports virtualized environments. You can deploy and run Index SDK services in such environments. For details about environment deployment, refer to the "Usage > [Virtual Instance Feature Guide](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/usage/virtual_instance/menu_virtual_instance.md)" section in the *MindCluster Cluster Scheduling User Guide*.

Using Index SDK for feature retrieval depends on the NPU firmware driver package, CANN software package, [OpenBLAS](https://github.com/xianyi/OpenBLAS/tree/v0.3.9), and [Faiss](https://github.com/facebookresearch/faiss/tree/v1.6.1). Install the dependency packages first, and then install Index SDK. The installation flow is as follows:

1. Install the NPU driver firmware and CANN. See [Installing the NPU Driver Firmware and CANN](#installing-the-npu-driver-firmware-and-cann).
2. Install OpenBLAS. See [Installing OpenBLAS](#installing-openblas).
3. Install Faiss. See [Installing Faiss](#installing-faiss).
4. Install Index SDK. See [Obtain the Index SDK Package](#index-sdk-package-download) and [Installing Index SDK](#index-sdk-installation).

**Notes<a name="section947444412510"></a>**

- When deploying retrieval in a Docker container environment, ensure that the retrieval-related directories are mounted correctly in the Docker container (for example, `/usr/local/Ascend/driver` and `/usr/local/Ascend/develop/`). Otherwise, compilation of retrieval code inside the container may fail.
- For third-party open source software, if a vulnerability exists in a given version, fix and update it promptly according to the instructions provided by the open source project.

## Dependency Installation<a name="ZH-CN_TOPIC_0000001506414849"></a>

### Ubuntu<a name="ZH-CN_TOPIC_0000001631987505"></a>

For the required dependencies, recommended versions, and acquisition suggestions in Ubuntu environments, see [Table 1](#table20540329125613).

**Table 1**  Dependencies and recommended versions for Ubuntu
<a name="table20540329125613"></a>

|Dependency|Recommended Version|Acquisition Suggestion|
|--|--|--|
|gcc|7.5.0|Compile and install from the source package.|
|cmake|3.24.0 or later|Install through the package manager. An example installation command is as follows.<br>```sudo apt-get install -y cmake```<br>If the version in the package manager does not meet the minimum requirement, install it from source.|
|Python|3.9/3.10/3.11/3.12|Compile and install from the source package.|

Use the following commands to check whether GCC, CMake, and other dependencies are installed.

```bash
gcc --version
cmake --version
python3 --version
```

If the following information is returned, the corresponding software is installed. The following output is only an example. See the actual output.

```bash
gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
cmake version 3.24.0
Python 3.9.11
```

### CentOS<a name="ZH-CN_TOPIC_0000001632546921"></a>

For the required dependencies, recommended versions, and acquisition suggestions in CentOS environments, see [Table 2](#table20540329125611).

**Table 2** Dependencies and recommended versions for CentOS
<a id="table20540329125611"></a>

|Dependency|Recommended Version|Acquisition Suggestion|
|--|--|--|
|gcc|7.5.0|Compile and install from the source package.|
|cmake|3.24.0 or later|Install through the package manager. A sample installation command is as follows.<br>```sudo yum install -y cmake```<br>If the version in the package manager does not meet the minimum requirement, install it from source.|
|Python|3.9|Compile and install from the source package.|

Use the following commands to check whether GCC, CMake, and other dependencies are installed.

```bash
gcc --version
cmake --version
python3 --version
```

If the following information is returned, the corresponding software is installed. The following output is only an example. See the actual output.

```bash
gcc 7.5.0
cmake version 3.24.0
Python 3.9.11
```

### Python Dependencies<a name="ZH-CN_TOPIC_0000001632546921"></a>

After Python is installed, the required pip dependency names, recommended versions, and acquisition suggestions are shown in [Table 3](#table20540329125612).

**Table 3**  Dependencies and recommended versions for pip
<a id="table20540329125612"></a>

|Dependency|Recommended Version|Acquisition Suggestion|
|--|--|--|
|numpy|1.25.0|Example installation command:<br>```pip3 install numpy==1.25.0```<br>|
|decorator|5.2.1|Example installation command:<br>```pip3 install decorator==5.2.1```<br>|
|sympy|1.14|Example installation command:<br>```pip3 install sympy==1.14```<br>|
|cffi|1.15.1|Example installation command:<br>```pip3 install cffi==1.15.1```<br>|
|pyyaml|N/A|Example installation command:<br>```pip3 install pyyaml```<br>|
|pathlib2|N/A|Example installation command:<br>```pip3 install pathlib2```<br>|
|protobuf|N/A|Example installation command:<br>```pip3 install protobuf```<br>|
|scipy|N/A|Example installation command:<br>```pip3 install scipy```<br>|
|requests|N/A|Example installation command:<br>```pip3 install requests```<br>|
|attrs|N/A|Example installation command:<br>```pip3 install attrs```<br>|
|psutil|N/A|Example installation command:<br>```pip3 install psutil```<br>|
|faiss-cpu|1.13.2|Example installation command:<br>```pip3 install faiss-cpu==1.13.2```<br>|

### Installing the NPU Driver Firmware and CANN<a name="ZH-CN_TOPIC_0000001456854880"></a>

**Download Dependency Packages<a name="section119752030133014"></a>**

**Table 1**  Package List

<table>
<tr>
<th>Software Type</th>
<th>Package Name</th>
<th>Acquisition Method</th>
</tr>
<tr>
<td>Ascend NPU driver</td>
<td>Ascend-hdk-xxx-npu-driver_{version}_linux-{arch}.run</td>
<td rowspan="4">Click the [Get Link](https://www.hiascend.com/developer/download/commercial/result?module=cann) link. In the left-side accessory resources, configure settings in "Edit Resource Selection", filter the matching software packages, confirm the version information, and then obtain the required packages.</td>
</tr>
<tr>
<td>Ascend NPU firmware</td>
<td>Ascend-hdk-xxx-npu-firmware_{version}.run</td>
</tr>
<tr>
<td>CANN package</td>
<td>Ascend-cann-toolkit_{version}_linux-{arch}.run<br>CANN 8.5.0 and later also require installation of the matching CANN operator package.</td>
</tr>
<tr>
<td>CANN operator package</td>
<td>Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run<br>For CANN versions earlier than 8.5.0, this package is named Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run.</td>
</tr>
</table>

> [!NOTE]
>
>- `{version}` indicates the software version.
>- `{arch}` indicates the CPU architecture.
>- `{chip_type}` indicates the chip type.
>- Atlas A2 inference products require CANN 8.0.RC1 or later, and Ascend HDK 24.1.RC1 or later.

**Installing the NPU driver firmware, CANN, and OPS<a name="section451714713564"></a>**

1. Refer to the "[Installing the NPU Driver and Firmware](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Debian)" section in the *CANN Software Installation Guide* (commercial edition) or the "[Installing the NPU Driver and Firmware](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler)" section (community edition) to install the NPU driver firmware.
2. Refer to the "[Installing CANN](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netconda)" section in the *CANN Software Installation Guide* (commercial edition) or the "[Installing CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netconda)" section in the *CANN Software Installation Guide* (community edition) to install CANN.

    > [!NOTE]
    >- To ensure that Index SDK works properly, install the CANN dependencies together with CANN.
    >- The same user must be used to install both CANN and Index SDK. A regular user is recommended.

### Installing OpenBLAS<a name="ZH-CN_TOPIC_0000001506414813"></a>

You are advised to use the corresponding OpenBLAS version. This section only provides installation instructions for OpenBLAS v0.3.10. Follow the actual OpenBLAS version and environment you use.

**Procedure<a name="section97897492564"></a>**

1. Download the OpenBLAS v0.3.10 source archive and decompress it.

    ```bash
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
    tar -xf OpenBLAS-0.3.10.tar.gz
    ```

2. Go to the OpenBLAS directory.

    ```bash
    cd OpenBLAS-0.3.10
    ```

3. Build and install.

    ```bash
    make FC=gfortran USE_OPENMP=1 -j
    # By default, OpenBLAS is installed to /opt/OpenBLAS.
    make install
    # Or run the following command to install it to a specified path.
    #make PREFIX=/your_install_path install
    ```

4. Configure the environment variable for the library path.

    ```bash
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
    # Configure /etc/profile
    vim /etc/profile
    # Add export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH to /etc/profile
    source /etc/profile
    ```

5. Verify that the installation is successful.

    ```bash
    cat /opt/OpenBLAS/lib/cmake/openblas/OpenBLASConfigVersion.cmake | grep 'PACKAGE_VERSION "'
    ```

    If the software version is displayed correctly, the installation is successful.

### Installing Faiss

**Installation Notes<a name="section1541154012292"></a>**

- Before installing Faiss, complete the OpenBLAS installation. See [Installing OpenBLAS](#installing-openblas) for details.
- The Index SDK package supports coexistence of multiple versions of business shared libraries built against different Faiss versions. It is compatible by default with Faiss 1.10.x business chains. If you need features that depend on Faiss 1.14, such as IVFRaBitQ/RaBitQ, use the business shared library, header files, and `libfaiss.so` corresponding to Faiss 1.14.1.
- Install different Faiss versions into separate directories, such as `/usr/local/faiss/faiss1.10.0` and `/usr/local/faiss/faiss1.14.1`. Do not switch versions by overwriting `/usr/local/lib/libfaiss.so`. When compiling and running programs, explicitly select the required Faiss version through `-I`, `-L`, and `LD_LIBRARY_PATH`.
- This section only provides installation instructions for Faiss v1.10.0. Follow the actual Faiss version and environment you use.

> [!NOTE]
>
>- On Arm platforms, adapt the Faiss source code according to the gcc version before compiling and installing Faiss.
>- On Arm platforms, some older versions of gcc, such as 4.8.5, do not support direct compilation of Faiss 1.10.0. Some older compilers also do not support the implementation related to `simdlib_neon.h`, so you need to switch to the default SIMD implementation on CPU. With this method, the functions work normally, but some Index algorithms, such as IVF and SQ, may suffer significant performance degradation. Using gcc 7.5.0 for compilation and installation is recommended. Versions later than gcc 9.5.0 may have compatibility issues.

**Procedure<a name="section94317151588"></a>**

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

3. Write the following content into `install_faiss_sh.sh`.

    ```bash
    # Modify source code
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
    # End of source code modification
    cd ..
    ls
    # Step 2: configure the Faiss build
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    # Step 3: build and install
    cd build && make -j && make install
    ```

4. Press `Esc`, type <b>:wq!</b>, and press `Enter` to save and exit the editor.
5. Download the Faiss source package and decompress it for installation.

    ```bash
    bash install_faiss_sh.sh
    ```

    > [!NOTE]
    >- Compiling Faiss 1.10.0 requires CMake 3.24.0 or later. If CMake reports that the version is too old when compiling Faiss, refer to [CMake error information when compiling Faiss 1.10.0](./faq.md#cmake-error-during-faiss-1100-compilation) for a solution.
    >- The default installation directory for Faiss is `/usr/local/lib`. If you need to specify an installation directory, for example `install_path=/usr/local/faiss/faiss1.10.0`, add the `-DCMAKE_INSTALL_PREFIX=${install_path}` option to the CMake build configuration.
    >
    >```bash
    >install_path=/usr/local/faiss/faiss1.10.0
    >cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${install_path}
    >```
    >
    > When using the IVFRaBitQ/RaBitQ feature, you need to install Faiss 1.14.1 separately. You are advised to use an independent installation directory such as `/usr/local/faiss/faiss1.14.1`, and set `-DCMAKE_INSTALL_PREFIX=/usr/local/faiss/faiss1.14.1` in the build configuration.

6. Configure the environment variable for the system library search path.

    Programs that dynamically link against Faiss need to know where the Faiss shared library is located at runtime. Therefore, add the Faiss library directory to the `LD_LIBRARY_PATH` environment variable.

    ```bash
    # Configure /etc/profile
    vim /etc/profile
    # Add the following to /etc/profile: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    # /usr/local/lib is the Faiss installation directory. If Faiss is installed elsewhere, replace /usr/local/lib with the actual Faiss installation path. In some operating systems and environments, Faiss may be installed in a different directory. For example, on CentOS, the path is /usr/local/lib64.
    source /etc/profile
    cd ..
    ```

7. Verify that the installation is successful.

    ```bash
    cat /usr/local/share/faiss/faiss-config-version.cmake |grep 'PACKAGE_VERSION "'
    ```

    If the software version is displayed correctly, the installation is successful.

> [!NOTE]
> If an error occurs after compiling Faiss on openEuler, refer to [undefined reference returned when linking libfaiss.so](./faq.md#an-undefined-reference-error-is-returned-when-linking-libfaissso) for a solution.

## Index SDK Package Download<a name="ZH-CN_TOPIC_0000001456695124"></a>

Obtain the required packages and the corresponding digital signature files in this section.

**Table 1**  Packages

|Component Name|Package Name|Acquisition Method|
|--|--|--|
|Index SDK|Feature retrieval package|[Get Link](https://www.hiascend.com/zh/developer/download/community/result?module=sdk%2Bcann)|

**Software Digital Signature Verification<a name="section10830205518487"></a>**

To prevent malicious tampering during transmission or storage, download the corresponding digital signature file together with the package for integrity verification.

After downloading the package, refer to the *OpenPGP Signature Verification Guide* to verify the package's PGP digital signature. If the verification fails, do not use the package and contact Huawei technical support for assistance.

Before installing or upgrading with the package, verify the package's digital signature by following the same procedure to ensure that the package has not been tampered with.

Carrier customers: [https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

Enterprise customers: [https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)

## Index SDK Installation<a name="ZH-CN_TOPIC_0000001456375296"></a>

**Installation Notes<a name="section3134195618512"></a>**

- Users who install and run Index SDK must meet the following requirements:
    - The same user must be used to install and run Index SDK, and the user must be the same as the one used to install CANN. Otherwise, permission issues may occur when running generated operators.
    - The user who installs and runs Index SDK is advised to be a regular user. Index SDK depends on shared libraries from low-privilege users in the CANN package. If the program is run as root, there is a security risk that the linked shared libraries could be tampered with by a low-privilege user.
    - The owner of the package directory and the installation target directory must be the installation user.
    - When installing Index SDK, ensure that the `~` directory exists and that the installation user has read and write permissions for that directory.

- Feature retrieval is released as a binary shared library, and the run package installs it locally to a user-defined path.

**Installation Steps<a name="section109931516193714"></a>**

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
    - **If the user does not specify an installation path**, the software is installed to the following default paths:
        - If installed as root, the default installation path is `/usr/local/Ascend`.
        - If installed as a non-root user, the default installation path is `${HOME}/Ascend`, where `${HOME}` is the user directory.
    - **If the user specifies an installation path**, create it first. For example, to use `/home/work/FeatureRetrieval`:

        ```bash
        mkdir -p /home/work/FeatureRetrieval
        ```

6. Obtain the installation command. Select the corresponding option below according to the actual situation to obtain the Index SDK installation command.

    **Table 1**  Quick Installation Details
    <table>
    <tr>
    <th>Product Series</th>
    <th>Product Model</th>
    <th>CPU Architecture</th>
    <th>Whether an Installation Path Is Specified</th>
    <th>Installation Command</th>
    </tr>
    <tr>
    <td rowspan="8">Atlas 200/300/500 inference products</td>
    <td rowspan="4">Atlas 300I inference card (model 3000)</td>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310 <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310 <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="20">Atlas inference products</td>
    <td rowspan="4">Atlas 300I Pro inference card</td>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300V video analytics card</td>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300V Pro video analytics card</td>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300I Duo inference card</td>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 200I SoC A1 core board</td>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas A2 inference products</td>
    <td rowspan="4">Atlas 800I A2 inference server</td>
    <td rowspan="2">x86_64</td>
    <td>Yes</td>
    <td>./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=npu_type<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed. <i>npu_type</i> indicates the chip name. On a server where the Ascend AI Processor is installed, run `npu-smi info` to query it. Delete the last digit of the queried "Name" value to obtain the value of `npu_type`.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=npu_type<br># </td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=npu_type <br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed. <i>npu_type</i> indicates the chip name. On a server where the Ascend AI Processor is installed, run `npu-smi info` to query it. Delete the last digit of the queried "Name" value to obtain the value of `npu_type`.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=npu_type<br># The installation path defaults to the directory where the command is run. <i>npu_type</i> indicates the chip name. On a server where the Ascend AI Processor is installed, run `npu-smi info` to query it. Delete the last digit of the queried "Name" value to obtain the value of `npu_type`.</td>
    </tr>
    <tr>
    <td rowspan="2">Atlas 800I A3 SuperNode server</td>
    <td rowspan="2">Atlas 800I A3 SuperNode server</td>
    <td rowspan="2">aarch64</td>
    <td>Yes</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=A3<br># This command uses `/home/work/FeatureRetrieval` as an example. Replace it as needed.</td>
    </tr>
    <tr>
    <td>No</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=A3<br># The installation path defaults to the directory where the command is run.</td>
    </tr>
    </table>

    The Index SDK run package includes business shared libraries built separately against Faiss 1.10.x and Faiss 1.14.1. During installation, you can use `--faiss-version` to select the default Faiss ABI version to be activated after installation. If this option is not specified, Faiss 1.10.x is activated by default, which is compatible with older business chains.

    ```bash
    # Activate the business shared libraries for Faiss 1.10.x, suitable for scenarios that do not use IVFRaBitQ/RaBitQ
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=/home/work/FeatureRetrieval --platform=910B --faiss-version=1.10

    # Activate the business shared libraries for Faiss 1.14.1, suitable for scenarios that use IVFRaBitQ/RaBitQ
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=/home/work/FeatureRetrieval --platform=910B --faiss-version=1.14
    ```

    After installation, the package configures the following symbolic links according to `--faiss-version`:

    ```text
    mxIndex/host/lib/libascendfaiss.so -> faiss1.1x/libascendfaiss.so
    mxIndex/host/lib/libascendsearch.so -> faiss1.1x/libascendsearch.so
    mxIndex/include/faiss -> faiss1.1x/faiss
    mxIndex/include/ascend -> faiss/ascend
    ```

    > [!NOTE]
    >
    > The run package only switches the symbolic links for the business shared libraries and header files provided by Index SDK. It does not install or replace `libfaiss.so` in the user's environment. When compiling and running applications, add the include and lib directories for the corresponding Faiss version to the build arguments and `LD_LIBRARY_PATH`. Use Faiss 1.14.1 for IVFRaBitQ/RaBitQ scenarios. Use Faiss 1.10.x for non-IVFRaBitQ/RaBitQ scenarios.

    1. Run the installation command to install the package. The user must ensure that the entire installation process is performed by the same user, and that only this user can access the installation path and extraction path.

        > [!NOTE]
        > The `--install` command also supports optional parameters, as shown in [Table 2](#table7138521890). Parameters not listed may still work or may cause an error.

    2. Add the environment variable for the Index package path. Using `/home/work/FeatureRetrieval` as the installation path for Index SDK as an example:

        ```bash
        export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib/:$LD_LIBRARY_PATH
        ```

        If the application directly includes Faiss header files or calls Faiss APIs, such as `faiss::read_index`, `faiss::write_index`, or `faiss::IndexIVFRaBitQ`, you must also explicitly select the Faiss version that matches `--faiss-version` during compilation and runtime. Using `/home/work/FeatureRetrieval` as an example:

        ```bash
        # Non-IVFRaBitQ/RaBitQ scenarios, use Faiss 1.10.x
        g++ test.cpp -I/home/work/FeatureRetrieval/mxIndex/include -I/usr/local/faiss/faiss1.10.0/include \
            -L/home/work/FeatureRetrieval/mxIndex/host/lib -L/usr/local/faiss/faiss1.10.0/lib \
            -lascendfaiss -lascendsearch -lfaiss
        export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib:/usr/local/faiss/faiss1.10.0/lib:$LD_LIBRARY_PATH

        # IVFRaBitQ/RaBitQ scenarios, use Faiss 1.14.1
        g++ test.cpp -I/home/work/FeatureRetrieval/mxIndex/include -I/usr/local/faiss/faiss1.14.1/include \
            -L/home/work/FeatureRetrieval/mxIndex/host/lib -L/usr/local/faiss/faiss1.14.1/lib \
            -lascendfaiss -lascendsearch -lfaiss
        export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib:/usr/local/faiss/faiss1.14.1/lib:$LD_LIBRARY_PATH
        ```

**Reference Information<a name="section111812571483"></a>**

**Table 2**  Optional Parameters for the `--install` Command<a id="table7138521890"></a>

|Parameter Name|Description|
|--|--|
|--help \| -h|Display help information.|
|--info|Display package build information.|
|--list|Display the file list.|
|--check|Check package integrity.|
|--quiet\|-q|Optional parameter that enables silent installation. Using this parameter reduces interactive output.|
|--nox11|Deprecated interface. Has no actual effect.|
|--noexec|Decompress the package to the current directory without executing the installation script. Use it together with `--extract=<path>`, in the form `--noexec --extract=<path>`.|
|--extract=\<path>|Extract files in the package to the specified directory. Can be used together with `--noexec`, `--install`, or `--upgrade`.|
|--tar arg1 [arg2 ...]|Run tar on the package and use the arguments after `tar` as command arguments. For example, `--tar xvf` extracts the contents of the run installation package to the current directory.|
|--version|Display the Index SDK version of the installation package.|
|--install|Installation command for the feature retrieval package.|
|--install-path=*\<path>*|Optional. Customize the root installation directory for the feature retrieval package. If not set, the current directory where you run the command is used by default. The path must start with `/` or `~`, and can contain only letters, digits, `-`, `_`, `.`, and `/`.<br>If not specified, the package is installed to the default path:<li>If installed as root, the default installation path is `/usr/local/Ascend`.</li><li>If installed as a non-root user, the default installation path is `${HOME}/Ascend`, where `${HOME}` is the user directory.</li><br>If this parameter is used to specify the installation directory, other users must not have write permission to that directory. If a regular user is specified for installation, the owner of the installation directory must be the current installation user.|
|--upgrade|Installation command for upgrading the feature retrieval package, which upgrades feature retrieval to the Index SDK version included in the package.|
|--platform=*\<npu_type>*|Corresponding Ascend AI Processor type.<li>For Atlas 200/300/500 inference products, enter 310.</li><li>For Atlas inference products, enter 310P.</li><li>For Atlas 800I A3 SuperNode server, enter `A3`.</li><li>For Atlas A2 inference products, run `npu-smi info` on a server where the Ascend AI Processor is installed to query it, then remove the last digit of the queried "Name" value. The result is the value of `--platform`.</li>|
|--faiss-version=*\<version>*|Optional. Select the default activated Faiss ABI version after installation. Supported values include `1.10`, `1.10.0`, `faiss1.10`, `1.14`, `1.14.1`, and `faiss1.14`. The default value is `1.10`. When `1.10` is selected, the business shared libraries and header files built against Faiss 1.10.x are activated. When `1.14` is selected, the business shared libraries and header files built against Faiss 1.14.1 are activated.|

> [!NOTE]
> The following parameters are not shown in `--help`. Do not use them directly.
>
>- `--xwin`: Run in xwin mode.
>- `--phase2`: Require the second-step action.

## Upgrade<a name="ZH-CN_TOPIC_0000001675534950"></a>

### Before You Upgrade<a name="ZH-CN_TOPIC_0000001649833012"></a>

**Notes<a name="section44827188568"></a>**

- Upgrade involves uninstalling and then reinstalling the installation directory. If other files exist in the directory, they will also be deleted. Before performing an upgrade, ensure that all data has been properly handled.
- When changing deployment from the open-state deployment of Index SDK 5.0.RC2 to the standard-state deployment of a version later than 5.0.RC2, refer to [Uninstallation](#uninstallation) and redeploy the standard-state feature retrieval after uninstalling the open-state deployment.
- During deployment, link the shared libraries in the `mxIndex-{version}/host` directory and regenerate the environment variables for the operator and operator model file directories.

### Upgrade Procedure<a name="ZH-CN_TOPIC_0000001675532836"></a>

Use the following command to upgrade the feature retrieval package. For parameter descriptions, see [Table 1](#table121021026102016).

```bash
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=platform --install-path={mxIndex_install_path}
```

**Table 1**  Parameter Names and Descriptions<a id="table121021026102016"></a>

|Parameter Name|Description|
|--|--|
|--upgrade|Installation command for upgrading the feature retrieval package, which upgrades feature retrieval to the Index SDK version included in the package.|
|--platform=*\<npu_type>*|Corresponding Ascend AI Processor type.<li>For Atlas 200/300/500 inference products, enter 310.</li><li>For Atlas inference products, enter 310P.</li><li>For Atlas 800I A3 SuperNode server, enter `A3`.</li><li>For Atlas A2 inference products, run `npu-smi info` on a server where the Ascend AI Processor is installed to query it, then remove the last digit of the queried "Name" value. The result is the value of `--platform`.</li>|
|--install-path=*\<path>*|Optional. Customize the root installation directory for the feature retrieval package. If not set, the current directory where you run the command is used by default.<br>If a custom directory is used for installation, specify this parameter during upgrade.|
|--faiss-version=*\<version>*|Optional. Select the default activated Faiss ABI version after upgrade. Supported values include `1.10`, `1.10.0`, `faiss1.10`, `1.14`, `1.14.1`, and `faiss1.14`. The default value is `1.10`. If you need to continue using IVFRaBitQ/RaBitQ after the upgrade, specify `--faiss-version=1.14`.|

**Procedure<a name="section1479912418555"></a>**

1. Example commands for upgrading the feature retrieval package are shown below, using `/home/work/FeatureRetrieval` as the installation path.

    - For Atlas 200/300/500 inference products:

        ```bash
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=310 --install-path=/home/work/FeatureRetrieval
        ```

    - For Atlas inference products:

        ```bash
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=310P --install-path=/home/work/FeatureRetrieval
        ```

    - For Atlas A2 inference products:

        ```bash
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=platform --install-path=/home/work/FeatureRetrieval
        ```

        If the value of `platform` is unclear, run `npu-smi info` on a server where the Ascend AI Processor is installed to query it. Then delete the last digit of the queried "Name" value to obtain the value of `platform`.

    If the following information is returned after the command is executed, the feature retrieval package has been upgraded successfully.

    ```text
    Upgrade package successfully.
    ```

2. After completing the above operations, refer to [Step 9](#index-sdk-installation) and [Generate Operators](./user_guide.md#generating-operators) to complete the subsequent installation and configuration process for feature retrieval.

## Uninstallation<a name="ZH-CN_TOPIC_0000001698153309"></a>

> [!NOTE]
> Uninstallation involves deleting the installation directory. If other files exist in the directory, they will also be deleted. Before uninstalling, ensure that all data has been properly handled.
> Operator files must be deleted manually. When uninstalling, delete the retrieval-related operator files as well, where `{ASCEND_OPP_PATH}` is the environment variable directory set during [Installing Index SDK](#index-sdk-installation).
>
>- For Index SDK versions earlier than 5.0.0, the operator file installation directories are `${ASCEND_OPP_PATH}/op_impl` and `${ASCEND_OPP_PATH}/op_proto`.
>- For Index SDK 5.0.0 and later, the operator file installation directory is `${ASCEND_OPP_PATH}/vendors/mxIndex`.
> You can view the specific operator files with <b>./custom_opp_**_{arch}_**.run --list</b>.

**Procedure<a name="section2817182535117"></a>**

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
