# 安装部署<a name="ZH-CN_TOPIC_0000001506414717"></a>

## 安装说明<a name="ZH-CN_TOPIC_0000002022551553"></a>

Index SDK支持物理机部署和容器部署两种方式，本文档介绍在物理机内部署的方式。如果需要在Docker环境中部署特征检索，请参照[Index镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2)内容完成特征检索的容器化部署。

Ascend Docker Runtime的安装请参考《MindCluster  集群调度用户指南》的“安装 \> 安装部署 \> 手动安装 \> Ascend Docker Runtime”章节。

Index SDK已支持虚拟化环境，可在虚拟化环境下进行Index SDK的业务部署及运行，具体环境部署操作请参见《MindCluster 集群调度用户指南》的“使用 \> 虚拟化实例特性指南”章节。

Index SDK特征检索的使用，依赖NPU固件驱动包、CANN软件包、[OpenBLAS](https://github.com/xianyi/OpenBLAS/tree/v0.3.9)和[Faiss](https://github.com/facebookresearch/faiss/tree/v1.6.1)。需要在完成依赖软件包的安装后，再进行Index SDK的安装。具体安装流程如下：

1.  安装NPU驱动固件和CANN，参考[安装NPU驱动固件和CANN](#安装npu驱动固件和cann)。
2.  安装OpenBLAS，参考[安装OpenBLAS](#安装openblas)。
3.  安装Faiss，参考[安装Faiss](#安装faiss)。
4.  安装Index SDK，参考[获取Index SDK软件包](#获取index-sdk软件包)和[安装Index SDK](#安装index-sdk)。

**注意事项<a name="section947444412510"></a>**

-   在Docker容器环境下进行检索部署时，请确保Docker容器中检索相关的文件夹挂载正确\(“/usr/local/Ascend/driver”和“/usr/local/Ascend/develop/”等\)，否则可能导致容器内的检索编译失败。
-   对于第三方的开源软件，如果该版本中存在漏洞，需要及时根据开源版本中的对应说明进行修复和更新。


## 安装依赖<a name="ZH-CN_TOPIC_0000001506414849"></a>

### 安装NPU驱动固件和CANN<a name="ZH-CN_TOPIC_0000001456854880"></a>

**下载依赖软件包<a name="section119752030133014"></a>**



**表 1**  软件包清单

<table>
<tr>
<th>软件类型</th>
<th>软件包名称</th>
<th>获取方式</th>
</tr>
<tr>
<td>昇腾NPU驱动</td>
<td>Ascend-hdk-xxx-npu-driver_{version}_linux-{arch}.run</td>
<td rowspan="4">单击<a href="https://www.hiascend.com/developer/download/commercial/result?module=cann">获取链接</a>，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。</td>
</tr>
<tr>
<td>昇腾NPU固件</td>
<td>Ascend-hdk-xxx-npu-firmware_{version}.run</td>
</tr>
<tr>
<td>CANN软件包</td>
<td>Ascend-cann-toolkit_{version}_linux-{arch}.run<br>CANN 8.5.0及之后版本需要配套CANN算子包安装。</td>
</tr>
<tr>
<td>CANN算子包</td>
<td>Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run<br>CANN 8.5.0之前版本该包名为Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run。</td>
</tr>
</table>


> [!NOTE] 说明 
>-   \{version\}表示软件版本号。
>-   \{arch\}表示CPU架构。
>-   \{chip\_type\}表示芯片类型。
>-   Atlas A2 推理系列产品需要使用CANN 8.0.RC1及之后的版本，Ascend HDK 24.1.RC1及之后的版本。

**安装NPU驱动固件、CANN和OPS<a name="section451714713564"></a>**

1.  参考《CANN 软件安装指南》中的“安装NPU驱动和固件”章节（商用版）或“安装NPU驱动和固件”章节（社区版）安装NPU驱动固件。
2.  参考《CANN 软件安装指南》的“安装CANN”章节（商用版）或《CANN 软件安装指南》的“安装CANN”章节（社区版）安装CANN。

    > [!NOTE] 说明 
    >-   安装CANN时，为确保Index SDK正常使用，CANN的相关依赖也需要一并安装。
    >-   安装CANN和安装Index SDK的用户需为同一用户，建议为普通用户。


### 安装OpenBLAS<a name="ZH-CN_TOPIC_0000001506414813"></a>

推荐用户使用对应版本的OpenBLAS，在此处仅提供OpenBLAS v0.3.10的安装参考，具体安装步骤请以实际使用的OpenBLAS版本和环境为准。

**操作步骤<a name="section97897492564"></a>**

1.  下载OpenBLAS v0.3.10源码压缩包并解压。

    ```
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
    tar -xf OpenBLAS-0.3.10.tar.gz
    ```

2.  进入OpenBLAS目录。

    ```
    cd OpenBLAS-0.3.10
    ```

3.  编译安装。

    ```
    make FC=gfortran USE_OPENMP=1 -j
    # 默认将OpenBLAS安装在/opt/OpenBLAS目录下
    make install
    # 或执行如下命令可以安装在指定路径
    #make PREFIX=/your_install_path install
    ```

4.  配置库路径的环境变量。

    ```
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
    # 配置/etc/profile
    vim /etc/profile
    # 在/etc/profile中添加export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
    source /etc/profile
    ```

5.  验证是否安装成功。

    ```
    cat /opt/OpenBLAS/lib/cmake/openblas/OpenBLASConfigVersion.cmake | grep 'PACKAGE_VERSION "'
    ```

    如果正确显示软件的版本信息，则表示安装成功。


### 安装Faiss<a name="ZH-CN_TOPIC_0000001839589749"></a>

**安装须知<a name="section1541154012292"></a>**

-   请在安装Faiss之前，完成OpenBLAS的安装，具体安装步骤请参见[安装OpenBLAS](#安装openblas)。
-   当前发布的特征检索依赖于Faiss v1.10.0执行构建和发布，推荐用户使用对应版本的Faiss，在此处仅提供Faiss v1.10.0的安装参考，具体安装步骤请以实际Faiss版本和环境为准。

> [!NOTE] 说明 
>-   如果是ARM平台，编译安装Faiss前请根据gcc版本适配Faiss源码。
>-   ARM平台上，部分旧版本的gcc（如4.8.5等）不支持直接编译Faiss 1.10.0版本，部分旧版本的编译器不支持“simdlib\_neon.h”的相关实现，需要改用默认CPU上的SIMD实现，使用该方法时功能可以正常运行，但是部分Index算法（IVF类、SQ类等）会出现较大性能退化。推荐使用gcc7.5.0进行编译和安装，高于gcc9.5.0版本可能出现兼容性问题。

**操作步骤<a name="section94317151588"></a>**

1.  下载Faiss源码包并解压。

    ```
    # Faiss 1.10.0
    wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz
    tar -xf v1.10.0.tar.gz && cd faiss-1.10.0/faiss
    ```

2.  创建install\_faiss\_sh.sh脚本。

    ```
    vi install_faiss_sh.sh
    ```

3.  在install\_faiss\_sh.sh脚本中写入如下内容。

    ```
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
    cd ../.. && rm -f v1.10.0.tar.gz && rm -rf faiss-1.10.0
    ```

4.  按“Esc”键，输入<b>:wq!</b>，按“Enter”保存并退出编辑。
5.  下载Faiss源码压缩包并解压安装。

    ```
    bash install_faiss_sh.sh
    ```

    > [!NOTE] 说明 
    >-   编译该Faiss 1.10.0需要CMake的版本不低于CMake 3.24.0，如果编译Faiss时提示CMake版本过低，请参考[编译Faiss 1.10.0时，CMake出现报错信息](./faq.md#编译faiss-1100时cmake出现报错信息)解决。
    >-   Faiss默认安装目录为“/usr/local/lib”，如需指定安装目录，例如“install\_path=/usr/local/faiss/faiss1.10.0”，则在CMake编译配置加**-DCMAKE\_INSTALL\_PREFIX=$\{**_install\_path_**\}**选项即可。
    >    ```
    >    install_path=/usr/local/faiss/faiss1.10.0
    >    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${install_path}
    >    ```

6.  配置系统库查找路径的环境变量。

    动态链接依赖Faiss的程序在运行时需要知道Faiss动态库所在路径，需要在Faiss的库目录加入“LD\_LIBRARY\_PATH”环境变量。

    ```
    # 配置/etc/profile
    vim /etc/profile
    # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    # /usr/local/lib是Faiss的安装目录,如果安装在其他目录下,将/usr/local/lib替换为Faiss实际安装路径，部分操作系统和环境中，Faiss可能会安装在其他目录下。
    source /etc/profile
    cd ..
    ```

7.  验证是否安装成功。

    ```
    cat /usr/local/share/faiss/faiss-config-version.cmake |grep 'PACKAGE_VERSION "'
    ```

    如果正确显示软件的版本信息，则表示安装成功。

> [!NOTE] 说明 
>如果在openEuler系统中编译Faiss后报错，请参见[链接libfaiss.so时，返回undefined reference错误。](./faq.md#链接libfaissso时返回undefined-reference错误)解决。



## 获取Index SDK软件包<a name="ZH-CN_TOPIC_0000001456695124"></a>

请参考本章获取所需软件包和对应的数字签名文件。

**表 1**  软件包

|组件名称|软件包名称|获取方式|
|--|--|--|
|Index SDK|特征检索软件包|获取链接。|


**软件数字签名验证<a name="section10830205518487"></a>**

为了防止软件包在传递过程中或存储期间被恶意篡改，下载软件包时请下载对应的数字签名文件用于完整性验证。

在软件包下载之后，请参考《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请勿使用该软件包并联系华为技术支持工程师解决。

使用软件包安装/升级前，也需要按照上述过程，验证软件包的数字签名，确保软件包未被篡改。

运营商客户请访问：[https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

企业客户请访问：[https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)


## 安装Index SDK<a name="ZH-CN_TOPIC_0000001456375296"></a>

**安装须知<a name="section3134195618512"></a>**

-   安装和运行Index SDK的用户，需要满足：
    -   安装和运行Index SDK的用户需为同一用户，且必须与安装CANN为同一用户，否则存在运行生成算子时访问CANN的权限问题。
    -   安装和运行Index SDK的用户建议为普通用户。Index SDK依赖于CANN包的低权限用户的动态库，使用root用户运行程序时，存在链接的动态库被低权限用户篡改的安全风险。
    -   安装包所在目录、安装目标目录的属主必须为安装用户。
    -   安装Index SDK时必须保证有“\~”目录且安装用户对该目录有读、写权限。

-   特征检索以二进制共享库形式发布，软件包在本地用户自定义路径通过run包安装。

**安装步骤<a name="section109931516193714"></a>**

1.  以软件包的安装用户登录安装环境。
2.  将软件包上传到安装环境的任意路径下（如：“/home/work/FeatureRetrieval”）并进入软件包所在路径。
3.  增加对软件包的可执行权限。

    ```
    chmod u+x Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run
    ```

4.  执行如下命令，校验软件包的一致性和完整性。

    ```
    ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --check
    ```

    若显示如下信息，说明软件包已通过校验。

    ```
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.    
    ```

5.  创建软件包的安装路径。
    -   **若用户未指定安装路径**，软件会默认安装到软件包所在的路径。
    -   **若用户想指定安装路径**，需要先创建安装路径。以安装路径“/home/work/FeatureRetrieval”为例：

        ```
        mkdir -p /home/work/FeatureRetrieval
        ```

6.  获取安装命令。请在下方根据实际情况单击对应选项，获取Index SDK的安装命令。

    **表 1**  快速安装详细信息
    <table>
    <tr>
    <th>产品系列</th>
    <th>产品型号</th>
    <th>CPU架构</th>
    <th>是否指定安装路径</th>
    <th>安装命令</th>
    </tr>
    <tr>
    <td rowspan="8">Atlas 200/300/500 推理产品</td>
    <td rowspan="4">Atlas 300I 推理卡（型号 3000）</td>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310 <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310 <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300I 推理卡（型号 3010）</td>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310 <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310 <br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310 <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="20">Atlas 推理系列产品</td>
    <td rowspan="4">Atlas 300I Pro 推理卡</td>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300V 视频解析卡</td>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300V Pro 视频解析卡</td>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 300I Duo 推理卡</td>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas 200I SoC A1 核心板</td>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=310P <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=310P<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    <tr>
    <td rowspan="4">Atlas A2 推理系列产品</td>
    <td rowspan="4">Atlas 800I A2 推理服务器</td>
    <td rowspan="2">x86_64</td>
    <td>是</td>
    <td>./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --install-path=/home/work/FeatureRetrieval --platform=npu_type<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换；<i>npu_type</i>表示芯片名称，可在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-x86_64.run --install --platform=npu_type<br># </td>
    </tr>
    <tr>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=npu_type <br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换；<i>npu_type</i>表示芯片名称，可在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=npu_type<br># 安装路径默认为命令执行所在目录；<i>npu_type</i>表示芯片名称，可在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。</td>
    </tr>
    <tr>
    <td rowspan="2">Atlas 800I A3 超节点服务器</td>
    <td rowspan="2">Atlas 800I A3 超节点服务器</td>
    <td rowspan="2">aarch64</td>
    <td>是</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=/home/work/FeatureRetrieval --platform=A3<br># 该命令以安装路径/home/work/FeatureRetrieval为例，请根据实际情况进行替换。</td>
    </tr>
    <tr>
    <td>否</td>
    <td> ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --platform=A3<br># 安装路径默认为命令执行所在目录。</td>
    </tr>
    </table>



1.  执行安装命令，安装软件包。用户需确保整个安装过程由同一用户执行，安装路径和解压路径仅允许该用户访问。

    > [!NOTE] 说明 
    >--install安装命令同时支持输入可选参数，如[表2](#table7138521890)所示。输入不在列表中的参数可能正常安装或者报错。

2.  添加Index软件包路径的环境变量。以Index SDK的安装路径“/home/work/FeatureRetrieval”为例：

    ```
    export LD_LIBRARY_PATH=/home/work/FeatureRetrieval/mxIndex/host/lib/:$LD_LIBRARY_PATH
    ```

**相关参考<a name="section111812571483"></a>**

**表 2**  --install安装命令可选参数表<a id="table7138521890"></a>

|参数名称|说明|
|--|--|
|--help \| -h|查询帮助信息。|
|--info|查询包构建信息。|
|--list|查询文件列表。|
|--check|查询包完整性。|
|--quiet\|-q|可选参数，表示静默安装。使用该参数，减少人机交互的信息的打印。|
|--nox11|废弃接口，无实际作用。|
|--noexec|解压软件包到当前目录，但不执行安装脚本。配套--extract=\<path>使用，格式为：--noexec --extract=\<path>。|
|--extract=\<path>|解压软件包中文件到指定目录。可配套--noexec、--install、--upgrade之一参数使用。|
|--tar arg1 [arg2 ...]|对软件包执行tar命令，使用tar后面的参数作为命令的参数。例如执行--tar xvf命令，解压run安装包的内容到当前目录。|
|--version|查询安装包Index SDK版本。|
|--install|特征检索软件包安装操作命令。|
|--install-path=*\<path>*|（可选）自定义特征检索软件包安装根目录。如未设置，默认为当前命令执行所在目录。配置的路径必须以/或~开头，路径取值仅支持大小写字母、数字、-_./字符。<br>若不指定，将安装到默认路径下：<li>若使用root用户安装，默认安装路径为：/usr/local/Ascend。<li>若使用非root用户安装，则默认安装路径为：${HOME}/Ascend。<br>若通过该参数指定了安装目录，该目录other用户不能有写权限，如果指定普通用户安装，安装目录属主必须为当前安装用户。|
|--upgrade|特征检索软件包升级操作命令，将特征检索升级到安装包所包含的Index SDK版本。|
|--platform|对应昇腾AI处理器类型。<li>使用Atlas 200/300/500 推理产品请输入310。使用Atlas 推理系列产品请输入310P。<li>使用Atlas 800I A3 超节点服务器请输入“A3”。<li>使用Atlas A2 推理系列产品，请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。|


> [!NOTE] 说明
>以下参数未展示在--help参数中，用户请勿直接使用。
>-   --xwin：使用xwin模式运行。
>-   --phase2：要求执行第二步动作。


# 升级<a name="ZH-CN_TOPIC_0000001675534950"></a>

## 升级前必读<a name="ZH-CN_TOPIC_0000001649833012"></a>

**注意事项<a name="section44827188568"></a>**

-   升级操作涉及对安装目录的卸载再安装，如目录下存在其他文件，也会被一并删除。请在执行升级操作前，确保所有数据都已妥善处理。
-   当从Index SDK的5.0.RC2版本的开放态部署变更为5.0.RC2之后版本的标准态部署时，请参见[卸载](#卸载)，卸载开放态部署后再次部署标准态特征检索。
-   部署过程中，请链接“mxIndex-_\{version\}_/host”目录下的动态库并重新生成算子和配置算子模型文件目录环境变量。


## 升级操作<a name="ZH-CN_TOPIC_0000001675532836"></a>

特征检索包升级操作参考以下命令执行，升级操作参数说明请参见[表1](#table121021026102016)。

```
./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=form --install-path={mxIndex_install_path}
```

**表 1**  参数名及说明<a id="table121021026102016"></a>

|参数名|参数说明|
|--|--|
|--upgrade|特征检索软件包升级操作命令，将特征检索升级到安装包所包含的Index SDK版本。|
|--platform|对应昇腾AI处理器类型。<li>使用Atlas 200/300/500 推理产品请输入310。<li>使用Atlas 推理系列产品请输入310P。<li>使用Atlas 800I A3 超节点服务器请输入“A3”。<li>使用Atlas A2 推理系列产品，请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。|
|--install-path|（可选）自定义特征检索软件包安装根目录。如未设置，默认为当前命令执行所在目录。<br>如使用自定义目录安装，建议在升级操作时使用该参数。|


**操作步骤<a name="section1479912418555"></a>**

1.  特征检索包升级命令执行示例参见如下，以安装路径“/home/work/FeatureRetrieval”为例。

    -   对于Atlas 200/300/500 推理产品：

        ```
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=310 --install-path=/home/work/FeatureRetrieval
        ```

    -   对于Atlas 推理系列产品：

        ```
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=310P --install-path=/home/work/FeatureRetrieval
        ```

    -   对于Atlas A2 推理系列产品：

        ```
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --upgrade --platform=form --install-path=/home/work/FeatureRetrieval
        ```

        如果无法确定platform的取值，请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是platform的取值


    命令执行后返回如下信息，则表示特征检索包升级成功。

    ```
    Upgrade package successfully.
    ```

2.  完成以上操作后，请参考[步骤9](#安装index-sdk)和[生成算子](./user_guide.md#生成算子)步骤，完成特征检索后续安装配置过程。


# 卸载<a name="ZH-CN_TOPIC_0000001698153309"></a>

> [!NOTE] 说明
>卸载操作涉及对安装目录的删除步骤，如目录下存在其他文件，也会被一并删除。请在执行卸载操作前，确保所有数据都已妥善处理。
>算子文件需要用户手动进行删除。用户在卸载时请同时删除检索相关算子文件，其中\{ASCEND\_OPP\_PATH\}为[安装Index SDK](#安装index-sdk)时设置的环境变量目录。
>-   Index SDK  5.0.0之前版本，算子文件安装目录为"\$\{ASCEND_OPP_PATH}/op_impl"和"\${ASCEND_OPP_PATH}/op_proto"。
>-   Index SDK  5.0.0及之后版本，算子文件安装目录为“$\{ASCEND\_OPP\_PATH\}/vendors/mxIndex”。
>具体的算子文件，可以通过<b>./custom\_opp\_**_\{arch\}_**.run --list</b>查看。

**操作步骤<a name="section2817182535117"></a>**

1.  进入安装目录mxIndex-_\{version\}_。

    ```
    cd mxIndex-{version}
    ```

2.  进入“script”目录。

    ```
    cd script
    ```

3.  添加“uninstall.sh”文件可执行权限，并执行，完成卸载。

    ```
    chmod u+x uninstall.sh
    ./uninstall.sh
    ```

