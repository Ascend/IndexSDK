# 安装部署

## 安装说明

Index SDK 支持[离线安装](#离线安装)、[镜像安装](#镜像安装)、[源码安装](#源码安装)三种方式。

若采用离线安装或源码安装，请首先[安装相关依赖](#安装依赖说明)，若采用镜像安装请跳过该步骤。

**注意事项**

- 对于第三方的开源软件，如果该版本中存在漏洞，需要及时根据开源版本中的对应说明进行修复和更新。
- （可选）Ascend Docker Runtime的安装请参考《MindCluster 集群调度用户指南》的“安装 \> 安装部署 \> 手动安装 \> [Ascend Docker Runtime](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/installation_guide/03_installation/manual_installation/02_ascend_docker_runtime.md)”章节。
- （可选）Index SDK已支持虚拟化环境，可在虚拟化环境下进行Index SDK的业务部署及运行，具体环境部署操作请参见《MindCluster 集群调度用户指南》的“使用 \> [虚拟化实例特性指南](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/usage/virtual_instance/menu_virtual_instance.md)”章节。

## 安装依赖说明

### 安装 NPU 驱动固件和 CANN

请参考《[CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)》，使用 CANN 9.0.0 及对应驱动版本完成 NPU 驱动固件与 CANN 的安装。

> [!NOTE]
> 安装 CANN 和安装 Index SDK 的用户需为同一用户，建议为普通用户。

### 安装其他依赖

#### 其他依赖

|依赖名称|版本建议|获取建议|
|--|--|--|
|gcc|>=7.5.0|建议通过获取源码包编译安装|
|cmake|>=3.24.0|建议通过包管理安装，安装命令参考如下。<br>`sudo apt-get install -y cmake`<br>若包管理中的版本不符合最低版本要求，可自行通过源码方式安装。|
|Python|3.9/3.10/3.11/3.12|建议通过获取源码包编译安装|

参考如下命令，检查是否已安装 GCC、CMake 等依赖软件。

```bash
gcc --version
cmake --version
python3 --version
```

若分别返回如下信息，说明相应软件已安装（以下回显仅为示例，请以实际情况为准）。

```bash
gcc 7.5.0
cmake version 3.24.0
Python 3.9.11
```

#### Python 依赖

Python 安装好后，pip 所需依赖名称、对应版本及获取建议请参见下表：

|依赖名称|版本建议|获取建议|
|--|--|--|
|numpy|>=1.25.0|安装命令参考如下。<br>```pip3 install "numpy>=1.25.0"```<br>|
|decorator|>=5.2.1|安装命令参考如下。<br>```pip3 install "decorator>=5.2.1"```<br>|
|sympy|>=1.14|安装命令参考如下。<br>```pip3 install "sympy>=1.14"```<br>|
|cffi|>=1.15.1|安装命令参考如下。<br>```pip3 install "cffi>=1.15.1"```<br>|
|pyyaml|无|安装命令参考如下。<br>```pip3 install pyyaml```<br>|
|pathlib2|无|安装命令参考如下。<br>```pip3 install pathlib2```<br>|
|protobuf|无|安装命令参考如下。<br>```pip3 install protobuf```<br>|
|scipy|无|安装命令参考如下。<br>```pip3 install scipy```<br>|
|requests|无|安装命令参考如下。<br>```pip3 install requests```<br>|
|attrs|无|安装命令参考如下。<br>```pip3 install attrs```<br>|
|psutil|无|安装命令参考如下。<br>```pip3 install psutil```<br>|
|faiss-cpu|1.13.2|安装命令参考如下。<br>```pip3 install faiss-cpu==1.13.2```<br>|

#### 安装 OpenBLAS

推荐用户使用对应版本的 OpenBLAS，在此处仅提供 OpenBLAS v0.3.10 的安装参考，具体安装步骤请以实际使用的 OpenBLAS 版本和环境为准。

**操作步骤**

1. 下载 OpenBLAS v0.3.10 源码压缩包并解压。

    ```bash
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
    tar -xf OpenBLAS-0.3.10.tar.gz
    ```

2. 进入 OpenBLAS 目录。

    ```bash
    cd OpenBLAS-0.3.10
    ```

3. 编译安装。

    ```bash
    make FC=gfortran USE_OPENMP=1 -j
    # 默认将OpenBLAS安装在/opt/OpenBLAS目录下
    make install
    # 或执行如下命令可以安装在指定路径
    # make PREFIX=/your_install_path install
    ```

4. 配置库路径的环境变量。

    ```bash
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
    # 配置/etc/profile
    vim /etc/profile
    # 在/etc/profile中添加export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
    source /etc/profile
    ```

5. 验证是否安装成功。

    ```bash
    cat /opt/OpenBLAS/lib/cmake/openblas/OpenBLASConfigVersion.cmake | grep 'PACKAGE_VERSION "'
    ```

    如果正确显示软件的版本信息，则表示安装成功。

#### 安装 Faiss

**安装须知**

- 安装 Faiss 之前，请先完成上一节 OpenBLAS 的安装。
- Index SDK 构建脚本默认构建基于 Faiss 1.10.x 的单版本业务动态库。如果需要使用 IVFRaBitQ/RaBitQ 等依赖 Faiss 1.14 的特性，可指定构建基于 Faiss 1.14.1 的单版本业务动态库（使用时需对应链接 Faiss 1.14.1 的业务动态库、头文件和 libfaiss.so）；如果需要同时兼容 Faiss 1.10.x 和 Faiss 1.14.1，可在构建时开启多版本业务动态库共存。使用非 IVFRaBitQ/RaBitQ 特性且需要兼容老环境时，可选择 Faiss 1.10.x 版本业务动态库。
- 推荐将不同 Faiss 版本安装到相互独立的目录，例如 `/usr/local/faiss/faiss1.10.0` 和 `/usr/local/faiss/faiss1.14.1`。不建议通过覆盖 `/usr/local/lib/libfaiss.so` 切换版本，用户编译和运行程序时应通过 `-I`、`-L` 和 `LD_LIBRARY_PATH` 显式选择需要的 Faiss 版本。
- 此处仅提供 Faiss v1.10.0 的安装参考，具体安装步骤请以实际 Faiss 版本和环境为准。

> [!NOTE]
>
> - 如果是 ARM 平台，编译安装 Faiss 前请根据 gcc 版本适配 Faiss 源码。
> - ARM 平台上，部分旧版本的 gcc（如4.8.5等）不支持直接编译 Faiss 1.10.0 版本，部分旧版本的编译器不支持“simdlib\_neon.h”的相关实现，需要改用默认 CPU 上的 SIMD 实现，使用该方法时功能可以正常运行，但是部分 Index 算法（IVF 类、SQ 类等）会出现较大性能退化。推荐使用 gcc7.5.0 进行编译和安装，高于 gcc9.5.0 版本可能出现兼容性问题。

**操作步骤**

1. 下载 Faiss 源码包并解压。

    ```bash
    # Faiss 1.10.0
    wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz
    tar -xf v1.10.0.tar.gz && cd faiss-1.10.0/faiss
    ```

2. 创建 install\_faiss\_sh.sh 脚本。

    ```bash
    vi install_faiss_sh.sh
    ```

3. 在 install\_faiss\_sh.sh 脚本中写入如下内容。

    ```bash
    # modify source code
    # 步骤1：修改Faiss源码
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
    # 步骤2：Faiss编译配置
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    # 步骤3：编译安装
    cd build && make -j && make install
    ```

4. 按 `Esc` 键，输入 `:wq!`，按 `Enter` 保存并退出编辑。
5. 下载 Faiss 源码压缩包并解压安装。

    ```bash
    bash install_faiss_sh.sh
    ```

    > [!NOTE]
    > - 编译该 Faiss 1.10.0 需要 CMake 的版本不低于 CMake 3.24.0，如果编译 Faiss 时提示 CMake 版本过低，请参考[编译 Faiss 1.10.0 时，CMake 出现报错信息](./07_faq.md#编译faiss-1100时cmake出现报错信息)解决。
    > - Faiss 默认安装目录为 `/usr/local/lib`，如需指定安装目录，例如 `install_path=/usr/local/faiss/faiss1.10.0`，则在 CMake 编译配置中添加 `-DCMAKE_INSTALL_PREFIX=${install_path}` 选项即可。
    >
    > ```bash
    > install_path=/usr/local/faiss/faiss1.10.0
    > cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${install_path}
    > ```
    >
    > - 使用 IVFRaBitQ/RaBitQ 特性时，需要额外安装 Faiss 1.14.1。建议使用独立安装目录，例如 `/usr/local/faiss/faiss1.14.1`，并在编译配置中设置 `-DCMAKE_INSTALL_PREFIX=/usr/local/faiss/faiss1.14.1`。

6. 配置系统库查找路径的环境变量。

    动态链接依赖 Faiss 的程序在运行时需要知道 Faiss 动态库所在路径，需要在 Faiss 的库目录加入 `LD_LIBRARY_PATH` 环境变量。

    ```bash
    # 配置/etc/profile
    vim /etc/profile
    # 在 /etc/profile 中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    # /usr/local/lib 是 Faiss 的安装目录，如果安装在其他目录下，将 /usr/local/lib 替换为Faiss实际安装路径
    # 部分操作系统和环境中，Faiss 可能会安装在其他目录下。例如在 CentOS 操作系统下，路径为 /usr/local/lib64
    source /etc/profile
    cd ..
    ```

7. 验证是否安装成功。

    ```bash
    cat /usr/local/share/faiss/faiss-config-version.cmake |grep 'PACKAGE_VERSION "'
    ```

    如果正确显示软件的版本信息，则表示安装成功。

> [!NOTE]
> 如果在 openEuler 系统中编译 Faiss 后报错，请参见[链接 libfaiss.so 时，返回 undefined reference 错误](./07_faq.md#链接libfaissso时返回undefined-reference错误)解决。

#### 安装 AscendMiniOs(可选)

除了前述依赖外，还需要根据是否需要使用 ILFlat 算法选择安装**开放态场景包**。

- 如不需要，跳过此步骤。
- 如需要，请先下载 [Ascend-cann-device-sdk 安装包](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)。

```bash
unzip Ascend-cann-device-sdk_{version}_linux-{arch}.zip
# 解压得到 CANN-runtime-*-minios.{arch}.run
./CANN-runtime-*-minios.{arch}.run --devel --install-path=/usr/local/AscendMiniOs
./CANN-runtime-*-minios.{arch}.run --run --install-path=/usr/local/AscendMiniOSRun
```

## 安装方式

### 离线安装

请从[获取链接](https://www.hiascend.com/zh/developer/download/community/result?module=sdk%2Bcann)下载 Index SDK 特征检索软件包（Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run）。

**安装须知**

- 安装和运行 Index SDK 的用户，需要满足：
    - 安装和运行 Index SDK 的用户需为同一用户，且必须与安装 CANN 为同一用户，否则存在运行生成算子时访问 CANN 的权限问题。
    - 安装和运行 Index SDK 的用户建议为普通用户。Index SDK 依赖于 CANN 包的低权限用户的动态库，使用 root 用户运行程序时，存在链接的动态库被低权限用户篡改的安全风险。
    - 安装包所在目录、安装目标目录的属主必须为安装用户。
    - 安装 Index SDK 时，必须保证有 `~` 目录且安装用户对该目录有读、写权限。

- 特征检索以二进制共享库形式发布，软件包在本地用户自定义路径通过 run 包安装。

**安装步骤**

1. 以软件包的安装用户登录安装环境。
2. 将软件包上传到安装环境的任意路径下（如：`/home/work/FeatureRetrieval`）并进入软件包所在路径。
3. 增加对软件包的可执行权限。

    ```bash
    chmod u+x Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run
    ```

4. 执行如下命令，校验软件包的一致性和完整性。

    ```bash
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --check
    ```

    若显示如下信息，说明软件包已通过校验。

    ```bash
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
    ```

5. 创建软件包的安装路径。
    - **若用户未指定安装路径**，软件会默认安装到以下路径：
        - 若使用 root 用户安装，默认安装路径为：`/usr/local/Ascend`。
        - 若使用非 root 用户安装，则默认安装路径为：`${HOME}/Ascend`，${HOME} 指用户目录。
    - **若用户想指定安装路径**，需要先创建安装路径。以安装路径 `/home/work/FeatureRetrieval` 为例：

        ```bash
        mkdir -p /home/work/FeatureRetrieval
        ```

6. 执行安装命令。

    ```bash
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=<path> --platform=<npu_type>
    ```

详细参数如下：

|参数名称|说明|
|--|--|
|--help \| -h|查询帮助信息。|
|--info|查询包构建信息。|
|--list|查询文件列表。|
|--check|查询包完整性。|
|--quiet \| -q|可选参数，表示静默安装。使用该参数，减少人机交互的信息的打印。|
|--nox11|废弃接口，无实际作用。|
|--noexec|解压软件包到当前目录，但不执行安装脚本。配套--extract=\<path>使用，格式为：--noexec --extract=\<path>。|
|--extract=\<path>|解压软件包中文件到指定目录。可配套--noexec、--install、--upgrade之一参数使用。|
|--tar arg1 [arg2 ...]|对软件包执行tar命令，使用tar后面的参数作为命令的参数。例如执行--tar xvf命令，解压run安装包的内容到当前目录。|
|--version|查询安装包Index SDK版本。|
|--install|特征检索软件包安装操作命令。|
|--install-path=*\<path>*|（可选）自定义特征检索软件包安装根目录。配置的路径必须以/或~开头，路径取值仅支持大小写字母、数字、-_./字符。<br>若不指定，将安装到默认路径下：<ul><li>若使用root用户安装，默认安装路径为：/usr/local/Ascend。</li><li>若使用非root用户安装，则默认安装路径为：\$\{HOME}/Ascend，${HOME}指用户目录。</li></ul>若通过该参数指定了安装目录，该目录其他用户不能有写权限，如果指定普通用户安装，安装目录属主必须为当前安装用户。|
|--upgrade|特征检索软件包升级操作命令，将特征检索升级到安装包所包含的Index SDK版本。|
|--platform=*\<npu_type>*|对应昇腾AI处理器类型。<ul><li>使用<term>Atlas 推理系列产品</term>请输入310P。</li><li>使用Atlas 800I A3 超节点服务器请输入“A3”。</li><li>使用<term>Atlas A2 推理系列产品</term>，请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。</li></ul>|
|--faiss-version=*\<version>*|（可选）多版本run包用于选择安装后激活的Faiss ABI版本；单版本run包用于校验用户选择和包内版本是否一致。支持“1.10”、“1.10.0”、“faiss1.10”、“1.14”、“1.14.1”和“faiss1.14”等取值，默认值为包内默认版本。选择“1.10”时，激活基于Faiss 1.10.x构建的业务动态库和头文件；选择“1.14”时，激活基于Faiss 1.14.1构建的业务动态库和头文件。单版本run包无需设置该参数，如果设置为包内不包含的版本，安装会报错退出。|

> [!NOTE]
> 以下参数未展示在 `--help` 参数中，用户请勿直接使用。
>
> - --xwin：使用xwin模式运行。
> - --phase2：要求执行第二步动作。

**多版本 run 包**

Index SDK 多版本 run 包在安装时支持选择激活不同 Faiss ABI 版本的业务动态库：

```bash
# 多版本run包：激活Faiss 1.10.x版本业务动态库，适用于需要兼容老业务环境的场景
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=<path> --platform=<npu_type> --faiss-version=1.10

# 多版本run包：激活Faiss 1.14.1版本业务动态库，适用于使用IVFRaBitQ/RaBitQ特性的场景
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --install --install-path=<path> --platform=<npu_type> --faiss-version=1.14
```

多版本 run 包安装完成后，软件包会根据 `--faiss-version` 配置以下软链接：

```bash
mxIndex/host/lib/libascendfaiss.so -> faiss1.1x/libascendfaiss.so
mxIndex/host/lib/libascendsearch.so -> faiss1.1x/libascendsearch.so
mxIndex/include/faiss -> faiss1.1x/faiss
mxIndex/include/ascend -> faiss/ascend
```

> [!NOTE]
> run 包只负责提供或切换 Index SDK 业务动态库和头文件，不会安装或替换用户环境中的 libfaiss.so。用户编译、运行应用程序时，需要将对应版本 Faiss 的 include 和 lib 目录加入编译参数和 `LD_LIBRARY_PATH`。使用 IVFRaBitQ/RaBitQ 特性时，应选择 Faiss 1.14.1；使用非 IVFRaBitQ/RaBitQ 特性且需要兼容老环境时，可选择 Faiss 1.10.x。
> 单版本 run 包只包含一个 Faiss ABI 版本，安装脚本会校验 `--faiss-version` 和包内版本是否一致。如果用户为 Faiss 1.10.x 单版本包指定 `--faiss-version=1.14`，或为 Faiss 1.14.1 单版本包指定 `--faiss-version=1.10`，安装会报错退出。

如果应用程序直接包含 Faiss 头文件或调用 Faiss 接口，例如 `faiss::read_index`、`faiss::write_index` 或 `faiss::IndexIVFRaBitQ`，还需要在编译和运行时显式选择与 `--faiss-version` 一致的 Faiss 版本。以安装路径 `/home/work/FeatureRetrieval` 为例：

```bash
# 非IVFRaBitQ/RaBitQ业务场景，使用Faiss 1.10.x
g++ test.cpp -I/home/work/FeatureRetrieval/mxIndex/include -I/usr/local/faiss/faiss1.10.0/include \
    -L/home/work/FeatureRetrieval/mxIndex/host/lib -L/usr/local/faiss/faiss1.10.0/lib \
    -lascendfaiss -lascendsearch -lfaiss
export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib:/usr/local/faiss/faiss1.10.0/lib:$LD_LIBRARY_PATH

# IVFRaBitQ/RaBitQ业务场景，使用Faiss 1.14.1
g++ test.cpp -I/home/work/FeatureRetrieval/mxIndex/include -I/usr/local/faiss/faiss1.14.1/include \
    -L/home/work/FeatureRetrieval/mxIndex/host/lib -L/usr/local/faiss/faiss1.14.1/lib \
    -lascendfaiss -lascendsearch -lfaiss
export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib:/usr/local/faiss/faiss1.14.1/lib:$LD_LIBRARY_PATH
```

命令执行后返回如下信息，则表示特征检索包安装成功。

```bash
Install package successfully.
```

### 镜像安装

请参照 [Index SDK 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2)完成特征检索的容器化部署。

### 源码安装

在进行源码编译安装时，除了前述依赖外，还需要根据是否需要使用 ILFlat 算法选择安装**开放态场景包**。

- 如不需要，请在编译前将 `feature_retrieval/src/ascendfaiss/CMakeLists.txt` 中 `BUILD_ASCENDDEVICE` 选项设为 `OFF`，并注释38行 `# ASCEND_MINIOS_HOME`；
- 如需要，请先下载 [Ascend-cann-device-sdk 安装包](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)

```bash
unzip Ascend-cann-device-sdk_{version}_linux-{arch}.zip
# 解压得到 CANN-runtime-*-minios.{arch}.run
./CANN-runtime-*-minios.{arch}.run --devel --install-path=/usr/local/AscendMiniOs
./CANN-runtime-*-minios.{arch}.run --run --install-path=/usr/local/AscendMiniOSRun
```

进入 `build` 目录执行以下命令编译：

```bash
bash build.sh
```

- 默认构建 Faiss 1.10.x 单版本 run 包，即 `MULTI_FAISS_PACKAGE=OFF DEFAULT_FAISS_ABI=faiss1.10 bash build/build.sh`。该包只包含基于 Faiss 1.10.x 构建的业务动态库和头文件，安装时无需指定 `--faiss-version`。
- 如需构建 Faiss 1.14.1 单版本 run 包，可执行 `MULTI_FAISS_PACKAGE=OFF DEFAULT_FAISS_ABI=faiss1.14 bash build/build.sh`。该包只包含基于 Faiss 1.14.1 构建的业务动态库和头文件，适用于使用 IVFRaBitQ/RaBitQ 特性的场景。
- 如需在同一个 run 包中同时提供 Faiss 1.10.x 和 Faiss 1.14.1 两套业务动态库，可执行 `MULTI_FAISS_PACKAGE=ON DEFAULT_FAISS_ABI=faiss1.10 bash build/build.sh` 构建多版本 run 包。该包安装时可通过 `--faiss-version` 选择安装后激活的 Faiss ABI 版本；如未指定，默认激活构建时 `DEFAULT_FAISS_ABI` 指定的版本。

生成的 run 包在 `build/output` 目录下：`Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run`，执行相应安装命令即可完成安装。

## 升级

> [!NOTE]
>
> - 升级操作涉及对安装目录的卸载再安装，如目录下存在其他文件，也会被一并删除。请在执行升级操作前，确保所有数据都已妥善处理。
> - 当从 Index SDK 的 5.0.RC2 版本的开放态部署变更为 5.0.RC2 之后版本的标准态部署时，请[卸载](#卸载)开放态部署后再次部署标准态特征检索。
> - 部署过程中，请链接 `mxIndex-_{version}/host` 目录下的动态库并重新生成算子和配置算子模型文件目录环境变量。

特征检索包升级操作参考以下命令执行，升级操作参数说明请参见下表。

```bash
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=<npu_type> --install-path=<path>
```

| 参数名 | 参数说明 |
|--|--|
| --upgrade | 特征检索软件包升级操作命令，将特征检索升级到安装包所包含的 Index SDK 版本。|
| --platform=*\<npu_type>* | 对应昇腾 AI 处理器类型。<ul><li>使用 Atlas 推理系列产品请输入 `310P`。</li><li>使用 Atlas 800I A3 超节点服务器请输入 `A3`。</li><li>使用 Atlas A2 推理系列产品，请在安装昇腾 AI 处理器的服务器执行 `npu-smi info` 命令进行查询，将查询到的 `Name` 最后一位数字删掉，即是 --platform 的取值。</li></ul> |
| --install-path=*\<path>* |（可选）自定义特征检索软件包安装根目录。如未设置，默认为 `/usr/local/Ascend`。如使用自定义目录安装，建议在升级操作时使用该参数。|
| --faiss-version=*\<version>* |（可选）多版本 run 包用于选择升级后激活的 Faiss ABI 版本；单版本 run 包用于校验用户选择和包内版本是否一致。支持“1.10”、“1.10.0”、“faiss1.10”、“1.14”、“1.14.1”和“faiss1.14”等取值，默认值为包内默认版本。如果升级后需要继续使用 IVFRaBitQ/RaBitQ 特性，请使用 Faiss 1.14.1 版本 run 包，或在多版本 run 包中指定“--faiss-version=1.14”。单版本 run 包无需设置该参数，如果设置为包内不包含的版本，升级会报错退出。 |

命令执行后返回如下信息，则表示特征检索包升级成功。

```bash
Upgrade package successfully.
```

## 卸载

> [!NOTE]
> 卸载操作涉及对安装目录的删除步骤，如目录下存在其他文件，也会被一并删除。请在执行卸载操作前，确保所有数据都已妥善处理。
> 算子文件需要用户手动进行删除。用户在卸载时请同时删除检索相关算子文件，其中 `{ASCEND_OPP_PATH}` 为安装 Index SDK 时设置的环境变量目录。
>
> - Index SDK 5.0.0 之前版本，算子文件安装目录为 `${ASCEND_OPP_PATH}/op_impl` 和 `${ASCEND_OPP_PATH}/op_proto`。
> - Index SDK 5.0.0 及之后版本，算子文件安装目录为 `${ASCEND_OPP_PATH}/vendors/mxIndex`。
> - 具体的算子文件，可以通过 `./custom_opp_*.run --list` 查看。

**操作步骤**

1. 进入安装目录 `mxIndex-{version}`。

    ```bash
    cd mxIndex-{version}
    ```

2. 进入 `script` 目录。

    ```bash
    cd script
    ```

3. 添加 `uninstall.sh` 文件可执行权限，并执行，完成卸载。

    ```bash
    chmod u+x uninstall.sh
    ./uninstall.sh
    ```
