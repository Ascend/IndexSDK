<p id="MemoryBridge用户手册"></p>

# Memory Bridge 用户手册
# 目录
* [环境准备](#环境准备)
  * [Ascendtookkit](#Ascendtookkit)
  * [openssl1.1.1+(DT用)](#openssl1.1.1+DT用)
  * [Python3.7+,<3.10(DT用)](#Python3.7+DT用)
  * [GCC7.3+](#GCC7.3+)
  * [CMake3.13+](#CMake3.13+)
* [代码下载](#代码下载)
* [编译构建](#编译构建)
  * [方法一：使用build.sh进行构建](#方法一：使用build.sh进行构建)
  * [方法二：使用cmake](#方法二：使用cmake)
  * [清理构建过程的临时文件](#清理构建过程的临时文件)
* [DT](#DT)
  * [编译运行DT](#编译运行DT)
* [Topo侦测](#Topo侦测)
  * [使用说明](#使用说明)
    * [Options说明](#Options说明)
    * [其他约束说明](#其他约束说明)
    * [FAQ](#FAQ)
<p id="环境准备"></p>

# 环境准备
<p id="Ascendtoolkit"></p>

## Ascend toolkit
  ```bash
  # 固件与驱动下载：
  https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.3.RC2&driver=Ascend+HDK+25.3.RC1.2

  # cann下载：
  https://www.hiascend.com/developer/download/community/result?module=pt+cann
  
  # Ascend toolkit版本需要与CANN版本统一
  # 前往包所在路径
  chmod +x Ascend-cann-toolkit_{version}_linux-x86_64.run
  ./Ascend-cann-toolkit_{version}_linux-x86_64.run --install
  reboot
  ```

<p id="openssl1.1.1+DT用"></p>

## openssl 1.1.1+ (DT用)
  ```bash
  wget https://www.openssl.org/source/openssl-1.1.1c.tar.gz
  tar -zxvf openssl-1.1.1c.tar.gz
  cd /usr/local/openssl/openssl-1.1.1c
  ./config --prefix=/usr/local/openssl
  ./config -t
  make && make install //如果make命令执行不了，安装GCC
  cd /usr/local
  ldd /usr/local/openssl/bin/openssl //设置依赖文件目录

  //设置新配置
  ln -s /usr/local/openssl/bin/openssl /usr/bin/openssl
  ln -s /usr/local/openssl/include/openssl /usr/include/openssl
  echo "/usr/local/openssl/lib" >> /etc/ld.so.conf
  ldconfig -v
  rm /usr/bin/openssl //删除软连接
  ```

<p id="Python3.7+DT用"></p>

## Python 3.7+, <3.10 (DT用)

<p id="GCC7.3+"></p>

## GCC 7.3+

<p id="CMake3.13+"></p>

## CMake 3.13+

<p id="代码下载"></p>

# 代码

```bash
vsa_hpp\3rdparty\hcps\3rdparty\memory_bridge
```

<p id="编译构建"></p>

# 编译构建
<p id="方法一：使用build.sh进行构建"></p>

## 方法一： 使用build.sh进行构建
```bash
# bash build.sh
```
<p id="方法二：使用cmake"></p>

## 方法二： 使用cmake
```bash
# cd build
# cmake ..
-- The CXX compiler identification is GNU 7.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/local/bin/g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/yuguojun/memory_bridge/build
# make -j 20
[ 50%] Building CXX object src/ock/hmm/CMakeFiles/ock_hmm.dir/mgr/HeteroMemoryLocation.cpp.o
[100%] Linking CXX shared library libock_hmm.so
[100%] Built target ock_hmm
```
<p id="清理构建过程的临时文件"></p>

## 清理构建过程的临时文件
```bash
# bash build.sh -t clean
```
* 执行`bash build.sh -h`可以查看帮助信息
<p id="DT"></p>

# DT
<p id="编译运行DT"></p>

## 编译运行DT
```bash
# cd tests/
# bash run_dt.sh
```
* 覆盖率报告生成在 build/gcovr_report

<p id="Topo侦测"></p>

# Topo侦测
* **`TopoDetect`**
<p id="使用说明"></p>

## 使用说明
<p id="Options说明"></p>

### Options说明
|option|是否必填|缺省值|最小值|最大值|有效值|单位|其他说明|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|-m|是|-|-|-|PARALLEL<br>SERIAL|-|PARALLEL:多卡并发检测<br>SERIAL:多卡串行检测|
|-d|是|-|-|-|DecvideId:<CpuIdRange>,...|-|指定检测device与cpu core<br>Example:-d 0:1,2,4-5|
|-t|否|HOST_TO_DEVICE|-|-|HOST_TO_DEVICE<br>HOST_TO_HOST<br>DEVICE_TO_DEVICE<br>DEVICE_TO_HOST|-|-|
|-p|否|2|1|5|[1,5]|线程数|单卡线程数|
|-n|否|2|1|60|[1,60]|秒|单卡检测时间|
|-s|否|64|2|2048|[2,2048]|MB|每个传输包大小|
|-h|否|-|-|-|-|-|输出helpMssage|

<p id="其他约束说明"></p>

### 其他约束说明
* **错误码：0代表成功，0~255供可执行程序返回码使用， 10000~20000 HMM 内部使用， 100000~1000000 ACL模块使用**
* **传入参数DeviceId请检查与物理设备数是否匹配**
* **传入参数CpuIdRange不允许大于设备最大同时可运行线程数**
* **传入参数CpuIdRange允许重复值，重复值在运行时会被自动忽略，但在结果列表中不体现区间合并**
```bash
Example: 
./TopoDetect -m SERIAL -d 0:1,1
./TopoDetect -m SERIAL -d 0:1,1-5
./TopoDetect -m SERIAL -d 0:1-5,2-8
```
* **传入参数CpuIdRange不允许错误的范围区间,如区间尾大于区间头**
```bash
Error Example: 
./TopoDetect -m SERIAL -d 0:5-1
```
* **传输模式参数为ALL时，工具将会依次运行HOST_TO_DEVICE、HOST_TO_HOST、DEVICE_TO_DEVICE、DEVICE_TO_HOST四次检测，总运行时间4*单卡检测时间**
* **检测模式为串行(SERIAL)时，实际运行时间为单卡检测时间*参与检测device数**
* **当检测时间和单包大小值过大时（如60s，1024M）出现运行超时场景，可以减少检测时间或单包大小**

<p id="FAQ"></p>

### FAQ
* **mode输入不合法**
    * 错误信息
      Invalid mode[xx]. with parameter -m, valid value is <PARALLEL|SERIAL>
    * 错误原因
      -m入参错误。请检查输入是否合法，-m参数有效值为PARALLEL|SERIAL
    * 错误信息
      Lack input with parameter -m, Only the 'PARALLEL' or 'SERIAL' mode is supported.
    * 错误原因
      缺少-m参数。请检查输入，-m参数为必填项
* **单卡线程数输入不合法**
    * 错误信息
      Invalid thread number[xx]. with parameter -p, valid range is[1,5] default(2)
    * 错误原因
      -p入参错误。请检查输入是否合法，-p参数有效值范围[1,5]
* **单卡测试时间输入不合法**
    * 错误信息
      Invalid test time[xx]. with parameter -n, valid range is[1,60] default(2)
    * 错误原因
      -n入参错误。请检查输入是否合法，-n参数有效值范围[1,60]
* **传输包大小输入不合法**
    * 错误信息
      Invalid packet size[xx]. with parameter -n, valid range is[2,2048] default(64)
    * 错误原因
      -s入参错误。请检查输入是否合法，-s参数有效值范围[2,2048]
* **指定设备ID或CpuId错误**
    * 错误信息
      Lack input with parameter -d. Please type in deviceId and cpuId.
    * 错误原因
      缺少-d参数。请检查输入，deviceId与cpuId是必填参数
    * 错误信息
      运行失败，错误码107001
    * 错误原因
      -d参数传入的DeviceId不存在，请核查物理环境设备卡数量
    * 错误信息
      The CPU core id(xxx) is too large.
    * 错误原因
      -d参数传入CpuId超出core总数，请核查物理环境cpu数及core数
    * 错误信息
      The start value(xx) is greater than the end value(xx).
    * 错误原因
      -d参数传入CpuIdRange区间尾大于区间头，请检查入参