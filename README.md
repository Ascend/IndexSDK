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
- [建议与交流](#建议与交流)

# 最新消息
- [2025.12.30]: 🚀 INDEXSDK 开源发布

# 简介
Index SDK是基于Faiss开发的昇腾NPU异构检索加速框架，针对高维空间中的海量数据，提供高性能的检索，采用与Faiss风格一致的C++语言，结合TBE，Ascendc算子开发，支持ARM和x86_64平台。
用户可以在此框架上实现面向应用场景的检索系统。

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
Index SDK版本配套和特性变更

# 环境部署
1. 安装NPU驱动固件和CANN

	| 软件类型     | 软件包名称                                           | 获取方式     |
	| ------------ | ---------------------------------------------------- | ------------ |
	| NPU驱动      | Ascend-hdk-xxx-npu-driver_{version}_linux-{arch}.run | 昇腾社区下载 |
	| NPU固件      | Ascend-hdk-xxx-npu-firmware_{version}.run            | 昇腾社区下载 |
	| CANN软件包   | Ascend-cann-toolkit_{version}_linux-{arch}.run       | 昇腾社区下载 |
	| 开放态场景包 | Ascend-cann-device-sdk_{version}_linux-{arch}.run    | 昇腾社区商用版资源申请 |

2. 安装OpenBLAS到默认路径
	```bash
	# 下载OpenBLAS v0.3.10源码压缩包并解压
	wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
	tar -xf OpenBLAS-0.3.10.tar.gz

	# 进入OpenBLAS目录
	cd OpenBLAS-0.3.10

	# 编译安装
	make FC=gfortran USE_OPENMP=1 -j
	# 默认将OpenBLAS安装在/opt/OpenBLAS目录下
	make install
	# 或执行如下命令可以安装在指定路径
	# make PREFIX=/your_install_path install

	# 配置库路径的环境变量
	ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so
	# 配置/etc/profile
	vim /etc/profile
	# 在/etc/profile中添加export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
	source /etc/profile

	# 验证是否安装成功, 如果正确显示软件的版本信息，则表示安装成功
	cat /opt/OpenBLAS/lib/cmake/openblas/OpenBLASConfigVersion.cmake | grep 'PACKAGE_VERSION "'
	```

3. 安装Faiss到 ```/usr/local/faiss```
	```bash
	# 下载Faiss源码包并解压
	wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz
	tar -xf faiss-1.10.0.tar.gz && cd faiss-1.10.0/faiss

	# 创建install_faiss_sh.sh脚本
	vi install_faiss_sh.sh
	```
	在install_faiss_sh.sh脚本中写入如下内容:
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
	cd ../.. && rm -f faiss-1.10.0.tar.gz && rm -rf faiss-1.10.0
	```
	使用脚本进行安装：
	```bash
	bash install_faiss_sh.sh

	# Faiss默认安装目录为"/usr/local/lib"，如需指定安装目录，例如"install_path=/usr/local/faiss"，则在CMake编译配置加-DCMAKE_INSTALL_PREFIX=${install_path}选项即可。
	install_path=/usr/local/faiss
	cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${install_path}

	# 配置系统库查找路径的环境变量
	vim /etc/profile
	# 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
	# /usr/local/lib是Faiss的安装目录, 如果安装在其他目录下, 将/usr/local/lib替换为Faiss实际安装路径，部分操作系统和环境中, Faiss可能会安装在其他目录下。
	source /etc/profile

	# 验证是否安装成功, 如果正确显示软件的版本信息，则表示安装成功
	cat /usr/local/share/faiss/faiss-config-version.cmake | grep 'PACKAGE_VERSION "'
	```

4. 安装Index SDK

# 编译流程
本节以CANN 8.3.RC2相关配套为例，介绍如何通过源码编译生成 Index SDK，其中NPU驱动、固件和CANN软件包可以通过昇腾社区下载，开放态场景包可以通过登录 ```https://support.huawei.com``` 搜索CANN 8.3.RC2，在相关页面申请商业版下载。

1. 编译依赖下载

	```bash
	# 依赖均下载在项目根目录，脚本会自动进行patch/编译
	cd IndexSDK

	# 项目使用定制版的makeself进行打包，需要下载makeselfv2.5和对应的patch
	git clone -b v2.5.0.x https://gitcode.com/cann-src-third-party/makeself.git makeself_patch
	git clone -b release-2.5.0 https://gitcode.com/gh_mirrors/ma/makeself.git
	```

	若需要运行测试用例，则还要下载以下源码：
	```bash
	# mockcpp
	git clone -b v2.7.x-h3 https://gitcode.com/cann-src-third-party/mockcpp.git mockcpp_patch
	git clone -b v2.7 https://gitee.com/sinojelly/mockcpp.git
	# huawei_secure_c
	git clone -b v1.1.16 https://gitee.com/openeuler/libboundscheck.git huawei_secure_c
	# googletest
	git clone -b release-1.11.0 https://gitcode.com/GitHub_Trending/go/googletest.git googletest
	```

2. 执行编译
	
	执行以下命令编译：
    ```bash
	source /path/to/Ascend/ascend-toolkit/set_env.sh
    bash build/build.sh
	```

3. 生成的 run 包在 ```build/output``` 下：```Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run```

4. 执行测试用例

	首先安装lcov2.0用于统计测试覆盖率和生成可视化报告：
	```bash
	apt update
	apt install -y libcapture-tiny-perl libdatetime-perl libtimedate-perl
	wget https://github.com/linux-test-project/lcov/releases/download/v2.0/lcov-2.0.tar.gz
	tar -xzf lcov-2.0.tar.gz && cd lcov-2.0
	make install
	```

	然后执行以下命令运行测试用例：
	```bash
	bash build/build.sh ut
	```

# 快速入门
本章节提供了一个简单的样例，帮助用户快速体验运用Index SDK进行检索的流程。

假定在Atlas推理系列产品上，有业务需要使用到暴搜（Flat）算法，底库大小为100w，维度是512维，需要检索的向量是128个，topk是10，编写一个Demo调用Index接口大致步骤如下。

## 前提条件
- 已完成安装部署操作。
- 已经生成Flat和AICPU算子。

## 操作步骤
1. 构造Demo，过程包括：
	1. Demo中引入暴搜（Flat）的头文件。
	2. 构造底库向量数据，这里用随机数生成代替。
	3. 归一化底库数据。
	4. 初始化Flat的Index。
	5. 调用接口添加底库。
	6. 调用接口进行检索。

	demo.cpp代码如下：
	```cpp
	#include <faiss/ascend/AscendIndexFlat.h>
	#include <sys/time.h>
	#include <random>
	// 获取当前时间
	inline double GetMillisecs()
	{
		struct timeval tv = {0, 0};
		gettimeofday(&tv, nullptr);
		return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
	}
	// 使用随机数构造底库数据
	void Generate(size_t ntotal, std::vector<float> &data, int seed = 5678)
	{
		std::default_random_engine e(seed);
		std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
		data.resize(ntotal);
		for (size_t i = 0; i < ntotal; ++i) {
			data[i] = static_cast<float>(255 * rCode(e) - 128);
		}
	}
	// 底库数据归一化
	void Norm(size_t total, std::vector<float> &data, int dim)
	{
		for (size_t i = 0; i < total; ++i) {
			float mod = 0;
			for (int j = 0; j < dim; ++j) {
				mod += data[i * dim + j] * data[i * dim + j];
			}
			mod = sqrt(mod);
			for (int j = 0; j < dim; ++j) {
				data[i * dim + j] = data[i * dim + j] / mod;
			}
		}
	}
	int main()
	{
		int dim = 512;
		std::vector<int> device{0};
		size_t ntotal = 1000000;
		int searchnum = 128;
		std::vector<float> features(dim * ntotal);
		int64_t resourceSize = static_cast<int64_t>(1024) * 1024 * 1024;
		int topK = 10;
		printf("Generating random numbers start!\r\n");
		Generate(ntotal, features);
		Norm(ntotal, features, dim);
		try {
			// index初始化
			faiss::ascend::AscendIndexFlatConfig conf(device, resourceSize);
			auto metricType = faiss::METRIC_INNER_PRODUCT;
			faiss::ascend::AscendIndexFlat index(dim, metricType, conf);
			index.reset();
			// add底库
			printf("add start!\r\n");
			index.add(ntotal, features.data());
			size_t tmpTotal = index.getBaseSize(0);
			if (tmpTotal != ntotal) {
				printf("------- Error -----------------\n");
				return -1;
			}
			// search
			printf("search start!\r\n");
			int loopTimes = 1;
			std::vector<float> dist(searchnum * topK, 0);
			std::vector<faiss::idx_t> label(searchnum * topK, 0);
			auto ts = GetMillisecs();
			for (int i = 0; i < loopTimes; i++) {
				index.search(searchnum, features.data(), topK, dist.data(), label.data());
			}
			auto te = GetMillisecs();
			printf("search end!\r\n");
			printf("flat, base:%lu, dim:%d, searchnum:%d, topk:%d, duration:%.3lf, QPS:%.4f\n",
				ntotal,
				dim,
				searchnum,
				topK,
				te - ts,
				1000 * searchnum * loopTimes / (te - ts));
			return 0;
		} catch(...) {
			printf("Exception caught! \r\n");
			return -1;
		}
	}
	```

2. 编译demo.cpp
	```bash
	# 以安装路径“/home/work/FeatureRetrieval”为例
	g++ --std=c++11 -fPIC -fPIE -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O3  -Wl,-z,relro,-z,now,-z,noexecstack -s -pie \
	-o demo demo.cpp \
	-I/home/work/FeatureRetrieval/mxIndex/include \
	-I/usr/local/faiss/faiss1.10.0/include \
	-I/usr/local/Ascend/driver/include \
	-I/opt/OpenBLAS/include \
	-L/home/work/FeatureRetrieval/mxIndex/host/lib \
	-L/usr/local/faiss/faiss1.10.0/lib \
	-L/usr/local/Ascend/driver/lib64 \
	-L/usr/local/Ascend/driver/lib64/driver \
	-L/opt/OpenBLAS/lib \
	-L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
	-lfaiss -lascendfaiss -lopenblas -lc_sec -lascendcl -lascend_hal -lascendsearch -lock_hmm -lacl_op_compiler
	```

3. 运行Demo，显示search end!即表示Demo运行成功。
	```bash
	./demo
	...
	search end!
	```

# 功能介绍
- 全量检索
- 近似检索
- 属性过滤
- 批量检索


# 安全声明
安全要求：使用API读取文件时，用户需要保证该文件的owner必须为自己，且权限不大于640，避免发生提权等安全问题。

## 操作系统安全加固
1. 防火墙配置：操作系统安装后，若配置普通用户，可以通过在“/etc/login.defs”文件中新增“ALWAYS_SET_PATH=yes”配置，防止越权操作。
2. 设置umask
	建议用户将宿主机和容器中的umask设置为027及以上，提高文件权限。以设置umask为027为例，具体操作如下所示:
	```bash
	# 以root用户登录服务器，编辑“/etc/profile”文件
	vim /etc/profile

	# 在“/etc/profile”文件末尾加上umask 027，保存并退出
	# 执行如下命令使配置生效
	source /etc/profile
	```

## 检索使用安全加固
- 合理规划内存

	用户需要合理规划内存使用，确保使用不要超过系统资源限制。同时，检索业务特征底库存储于昇腾AI处理器DDR内，特征维度和数量（入库或查询等操作）以及计算过程中，业务临时内存和系统临时内存的使用决定总内存占用大小，输入过大会导致设备侧内存申请失败错误。当前单个Index（底库）支持最大库容视具体昇腾AI处理器Device侧内存大小而定，业务侧需要根据实际需求规划Index个数，防止内存超限情形发生。

- OMP设置

	如果需要修改OMP相关配置，请评估系统的内存、线程数等资源限制，否则可能导致运行异常，例如可以通过设置${OMP_NUM_THREADS}设置并发量。OMP的相关设置请参考OMP官方指导。

- 接口使用

	检索接口大多采用C语言的入参形式，因此需要用户保证输入指针的长度为有效值，否则可能导致运行异常。

- 和faiss::Index的相互转换

	检索提供和faiss::Index的相互转换功能，请确保copyTo输出的faiss::Index不会被修改，否则可能导致copyFrom异常；index_ascend_to_cpu、index_int8_ascend_to_cpu、index_cpu_to_ascend、index_int8_cpu_to_ascend等接口同理。

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

# 建议与交流
欢迎大家为社区做贡献。贡献前，请先签署开放项目贡献者许可协议（CLA）。如果有任何疑问或建议，请提交GitCode Issues，我们会尽快回复。 感谢您的支持。

贡献声明
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
