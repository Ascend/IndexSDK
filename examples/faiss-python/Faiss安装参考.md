# 依赖安装

```bash
apt-get install swig
apt-get install python3.10-dev
pip3 install wheel
pip3 install build
```

注 swig 要求 4.0.1 以上版本, 由于 pip24.3 将会舍弃 egg 文件直接构建的方式，该组件安装方式修改为 python 推荐的 build and pip 方式，故需新增依赖库 wheel 和 build 依赖库。

# Faiss安装参考

根据 index 版本，选择 faiss 的安装版本，index 版本为 5.0.0 及以前版本，安装 faiss 1.7.1, index 版本为 6.0.0 以后安装 faiss 1.7.4，index 版本为 7.1.0 以后安装 faiss 1.10.0;
如果使用 26.0.0-beta.1 版本 IndexSDK 的 IVFRaBitQ 检索算法，需要安装 faiss 1.14.1 版版本。

## 安装 faiss 1.14.1 版本

### 1、源码下载

执行以下命令下载Faiss源码压缩包并解压。（编译该Faiss需要CMake的版本不低于CMake 3.24.0。）

```bash
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.14.1.tar.gz
tar -xf v1.14.1.tar.gz && cd faiss-1.14.1/faiss
```

### 2、部署安装脚本

创建安装脚本文件

```bash
vi install_faiss_sh.sh
```

在安装脚本中添加以下内容

```bash
# modify source code
# 步骤1：修改Faiss源码

sed -i "214 i\\
    \\
    virtual void search_with_filter (idx_t n, const float *x, idx_t k,\\
                                     float *distances, idx_t *labels, const void *mask = nullptr) const {}\\
" Index.h
sed -i "60 i\\
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

# modify source code end
cd ..
ls
# 步骤2：Faiss编译配置
FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.14.1
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}
# 步骤3：编译安装
cd build && make -j && make install
cd ../.. && rm -f v1.14.1.tar.gz && rm -rf faiss-1.14.1
```

按“Esc”键，输入:wq!，按“Enter”保存并退出编辑

### 3、源码编译安装

1. 执行安装脚本完成faiss安装。

   ```bash
   bash install_faiss_sh.sh
   ```

2. 配置系统库查找路径，返回上层目录。

   动态链接依赖Faiss的程序在运行时需要知道Faiss动态库所在路径，需要在Faiss的库目录加入“LD_LIBRARY_PATH”环境变量。

   ```bash
   # 配置/etc/profile
   vim /etc/profile
   # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   # /usr/local/lib是Faiss的安装目录,如果安装在其他目录下,将/usr/local/lib替换为Faiss实际安装路径（例如上文参考配置中的/usr/local/faiss/faiss1.14.1/lib），部分操作系统和环境中，faiss可能会安装在其他目录下。
   source /etc/profile
   cd ..
   ```

## 安装 faiss 1.10.0 版本

### 1、源码下载

执行以下命令下载Faiss源码压缩包并解压。（编译该Faiss需要CMake的版本不低于CMake 3.24.0。）

```bash
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.10.0.tar.gz
tar -xf v1.10.0.tar.gz && cd faiss-1.10.0/faiss
```

### 2、部署安装脚本

创建安装脚本文件

```bash
vi install_faiss_sh.sh
```

在安装脚本中添加以下内容

```bash
# modify source code
# 步骤1：修改Faiss源码

sed -i "149 i\
    \
    virtual void search_with_filter (idx_t n, const float *x, idx_t k,\
                                     float *distances, idx_t *labels, const void *mask = nullptr) const {}\
" Index.h

sed -i "217 i\
    \
utils/sorting.h\
" ../CMakeLists.txt

sed -i "30 i\
    \
    explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids);\
" IndexIDMap.h

sed -i "49 i\
    \
template <typename IndexT>\
IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids):\
    index (index),\
    own_fields (false)\
{\
    this->is_trained = index->is_trained;\
    this->metric_type = index->metric_type;\
    this->verbose = index->verbose;\
    this->d = index->d;\
    id_map = ids;\
}\
" IndexIDMap.cpp

# modify source code end
cd ..
ls
# 步骤2：Faiss编译配置
PYTHON=/usr/local/lib/python3.10
FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.10.0
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DPython_EXECUTABLE=${PYTHON} -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}
# 步骤3：编译安装
make -C build -j faiss
make -C build -j swigfaiss
cd build/faiss/python && python3 setup.py bdist_wheel
cd ../../.. && make -C build install
cd build/faiss/python && cp libfaiss_python_callbacks.so ${FAISS_INSTALL_PATH}/lib
cd dist
pip3 install faiss-1.10.0*.whl
cd ../../.. && rm -f v1.10.0.tar.gz && rm -rf faiss-1.10.0
```

按“Esc”键，输入:wq!，按“Enter”保存并退出编辑

### 3、源码编译安装

1. 执行安装脚本完成faiss安装。

   ```bash
   bash install_faiss_sh.sh
   ```

   注：安装完成以后，如果执行失败，就查看执行失败目录文件是否解压，如果没有解压就手动解压

2. 配置系统库查找路径，返回上层目录。

   动态链接依赖Faiss的程序在运行时需要知道Faiss动态库所在路径，需要在Faiss的库目录加入“LD_LIBRARY_PATH”环境变量。

   ```bash
   # 配置/etc/profile
   vim /etc/profile
   # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   # /usr/local/lib是Faiss的安装目录,如果安装在其他目录下,将/usr/local/lib替换为Faiss实际安装路径，部分操作系统和环境中，faiss可能会安装在其他目录下。
   source /etc/profile
   cd ..
   ```

## 安装 faiss 1.7.4 版本

### 1、源码下载

执行以下命令下载Faiss源码压缩包并解压。（编译该Faiss需要CMake的版本不低于CMake 3.23.1。）

```bash
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.7.4.tar.gz
tar -xf v1.7.4.tar.gz && cd faiss-1.7.4/faiss
```

### 2、部署安装脚本

创建安装脚本文件

```bash
vi install_faiss_sh.sh
```

在安装脚本中添加以下内容

```bash
# modify source code
# 步骤1：修改Faiss源码

sed -i "131 i\
    \
    virtual void search_with_filter (idx_t n, const float *x, idx_t k,\
                                     float *distances, idx_t *labels, const void *mask = nullptr) const {}\
" Index.h

sed -i "199 i\
    \
utils/sorting.h\
" ../CMakeLists.txt

sed -i "29 i\
    \
    explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids);\
" IndexIDMap.h

sed -i "38 i\
    \
template <typename IndexT>\
IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids):\
    index (index),\
    own_fields (false)\
{\
    this->is_trained = index->is_trained;\
    this->metric_type = index->metric_type;\
    this->verbose = index->verbose;\
    this->d = index->d;\
    id_map = ids;\
}\
" IndexIDMap.cpp

# modify source code end
cd ..
ls
# 步骤2：Faiss编译配置
PYTHON=/usr/local/lib/python3.10
FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.7.4
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DPython_EXECUTABLE=${PYTHON} -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}
# 步骤3：编译安装
make -C build -j faiss
make -C build -j swigfaiss
cd build/faiss/python && python3 setup.py bdist_wheel
cd ../../.. && make -C build install
cd build/faiss/python && cp libfaiss_python_callbacks.so ${FAISS_INSTALL_PATH}/lib
cd dist
pip3 install faiss-1.7.4*.whl
cd ../../.. && rm -f v1.7.4.tar.gz && rm -rf faiss-1.7.4
```

按“Esc”键，输入:wq!，按“Enter”保存并退出编辑

### 3、源码编译安装

1. 执行安装脚本完成faiss安装。

   ```bash
   bash install_faiss_sh.sh
   ```

   注：安装完成以后，如果执行失败，就查看执行失败目录文件是否解压，如果没有解压就手动解压

2. 配置系统库查找路径，返回上层目录。

   动态链接依赖Faiss的程序在运行时需要知道Faiss动态库所在路径，需要在Faiss的库目录加入“LD_LIBRARY_PATH”环境变量。

   ```bash
   # 配置/etc/profile
   vim /etc/profile
   # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   # /usr/local/lib是Faiss的安装目录,如果安装在其他目录下,将/usr/local/lib替换为Faiss实际安装路径，部分操作系统和环境中，faiss可能会安装在其他目录下。
   source /etc/profile
   cd ..
   ```

## 安装 faiss 1.7.1 版本

### 1、源码下载

执行以下命令下载Faiss源码压缩包并解压。（编译该Faiss需要CMake的版本不低于CMake 3.17。）

```bash
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.7.1.tar.gz
tar -xf v1.7.1.tar.gz && cd faiss-1.7.1/faiss
```

### 2、部署安装脚本

创建安装脚本文件

```bash
vi install_faiss_sh.sh
```

在安装脚本中添加以下内容

```bash
# modify source code
# 步骤1：修改Faiss源码

sed -i "118 i\
    \
    virtual void search_with_filter (idx_t n, const float *x, idx_t k,\
                                     float *distances, idx_t *labels, const void *mask = nullptr) const {}\
" Index.h

sed -i "33 i\
    \
    explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids);\
" MetaIndexes.h

sed -i "39 i\
    \
template <typename IndexT>\
IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids):\
    index (index),\
    own_fields (false)\
{\
    this->is_trained = index->is_trained;\
    this->metric_type = index->metric_type;\
    this->verbose = index->verbose;\
    this->d = index->d;\
    id_map = ids;\
}\
" MetaIndexes.cpp

# modify source code end
cd ..
ls
# 步骤2：Faiss编译配置
PYTHON=/usr/local/lib/python3.10
FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.7.1
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DPython_EXECUTABLE=${PYTHON} -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}
# 步骤3：编译安装
make -C build -j faiss
make -C build -j swigfaiss
cd build/faiss/python && python3 setup.py bdist_wheel
cd ../../.. && make -C build install
cd build/faiss/python && cp libfaiss_python_callbacks.so ${FAISS_INSTALL_PATH}/lib
cd dist
pip3 install faiss-1.7.1*.whl
cd ../../.. && rm -f v1.7.1.tar.gz && rm -rf faiss-1.7.1
```

按“Esc”键，输入:wq!，按“Enter”保存并退出编辑

### 3、源码编译安装

1. 执行安装脚本完成faiss安装。

   ```bash
   bash install_faiss_sh.sh
   ```

   注：安装完成以后，如果执行失败，就查看执行失败目录文件是否解压，如果没有解压就手动解压

2. 配置系统库查找路径，返回上层目录。

   动态链接依赖Faiss的程序在运行时需要知道Faiss动态库所在路径，需要在Faiss的库目录加入“LD_LIBRARY_PATH”环境变量。

   ```bash
   # 配置/etc/profile
   vim /etc/profile
   # 在/etc/profile中添加: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   # /usr/local/lib是Faiss的安装目录,如果安装在其他目录下,将/usr/local/lib替换为Faiss实际安装路径，部分操作系统和环境中，faiss可能会安装在其他目录下。
   source /etc/profile
   cd ..
   ```

# ascendfaiss编译

**编译前请完成Index SDK的安装 并生成aicpu和flat 512维的算子** MXINDEX_INSTALL_PATH为安装路径

1、设置软件安装路径信息，根据实际情况配置

```bash
PYTHON_INSTALL_PATH=/usr/local/lib/python3.10
FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.7.4
MXINDEX_INSTALL_PATH=/home/mxIndex #根据实际路径修改

ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
DRIVER_INATALL_PATH=/home/driver #根据实际路径修改
PYTHON_HEADER=/usr/include/python3.10
```

2、编译ascendfaiss

```bash
swig -python -c++ -Doverride= -module swig_ascendfaiss -I${PYTHON_HEADER} -I${FAISS_INSTALL_PATH}/include -I${MXINDEX_INSTALL_PATH}/include -DSWIGWORDSIZE64 -o swig_ascendfaiss.cpp swig_ascendfaiss.swig

g++ -std=c++11 -DFINTEGER=int -fopenmp -I/usr/local/include -I${ASCEND_INSTALL_PATH}/acllib/include -I${ASCEND_INSTALL_PATH}/runtime/include -I${DRIVER_INATALL_PATH}/driver/kernel/inc/driver -I${DRIVER_INATALL_PATH}/driver/kernel/libc_sec/include -fPIC -fstack-protector-all -Wall -Wreturn-type -D_FORTIFY_SOURCE=2 -g -O3 -Wall -Wextra -I${PYTHON_HEADER}/ -I${PYTHON_INSTALL_PATH}/dist-packages/numpy/core/include -I${FAISS_INSTALL_PATH}/include -I${MXINDEX_INSTALL_PATH}/include -c swig_ascendfaiss.cpp -o swig_ascendfaiss.o

g++ -std=c++11 -shared -fopenmp -L${ASCEND_INSTALL_PATH}/acllib/lib64 -L${ASCEND_INSTALL_PATH}/runtime/lib64 -L${DRIVER_INATALL_PATH}/lib64/driver -L${DRIVER_INATALL_PATH}/driver/lib64/common -L${DRIVER_INATALL_PATH}/driver/lib64/driver -L${FAISS_INSTALL_PATH}/lib -Wl,-rpath-link=${ASCEND_INSTALL_PATH}/acllib/lib64:${ASCEND_INSTALL_PATH}/runtime/lib64:${DRIVER_INATALL_PATH}/driver/lib64:${DRIVER_INATALL_PATH}/driver/lib64/common:${DRIVER_INATALL_PATH}/driver/lib64/driver -L/usr/local/lib -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s -o _swig_ascendfaiss.so swig_ascendfaiss.o -L${MXINDEX_INSTALL_PATH}/host/lib -lascendfaiss -lfaiss -lascend_hal -lacl_retr -lascendcl -lc_sec -lopenblas

python3.10 -m build
```

进入dist文件夹, 使用pip安装生成的ascendfaiss*.whl文件:

```bash
cd dist
pip3 install ascendfaiss*.whl
```

设置环境变量，执行测试用例：

```bash
cd ..
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 test_ascend_index_flat.py
```
