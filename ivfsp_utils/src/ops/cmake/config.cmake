# 算子部署目录结构如下：
#├── opp      //算子库目录
#│   ├── op_impl
#│       ├── built-in
#│       ├── custom
#│           ├── ai_core
#│                ├── tbe
#│                    ├── config
#│                        ├── ${soc_version}     //昇腾AI处理器类型
#│                            ├── aic-${soc_version}-ops-info.json     //TBE自定义算子信息库
#│                    ├── custom_impl               //TBE自定义算子实现代码文件
#│                        ├── xx.py
#│           ├── vector_core   //此目录预留，无需关注
#│           ├── cpu          //AI CPU自定义算子实现库及算子信息库所在目录
#│                ├── aicpu_kernel
#│                    ├── custom_impl
#│                        ├── libcust_aicpu_kernels.so   //AI CPU算子实现库文件
#│                ├── config
#│                    ├── cust_aicpu_kernel.json         //AI CPU算子信息库
#│   ├── op_proto
#│       ├── built-in
#│       ├── custom
#│           ├── libcust_op_proto.so    //自定义算子原型库文件

# Set compiling options
ADD_COMPILE_OPTIONS(-std=c++11  -fPIC  -fstack-protector-all  -Wall  -Wreturn-type  -D_FORTIFY_SOURCE=2  -O2)
SET(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS}  -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack  -s")
SET(CMAKE_EXE_LINKER_FLAGS      "${CMAKE_EXE_LINKER_FLAGS}  -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack  -pie  -s")
INCLUDE_DIRECTORIES(${ASCEND_TOOLKIT_PATH}/include)

# Set run package name, this package is used to install OPP.
SET(RUN_TARGET custom_opp_${CMAKE_SYSTEM_PROCESSOR}.run)

# Parser
SET(INI_2_JSON_PY  ${CMAKE_CURRENT_LIST_DIR}/util/parse_ini_to_json.py)

# Set local package paths
SET(OP_PROTO_TARGET          cust_op_proto_ascendsearch)
SET(OP_PROTO_TARGET_OUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/op_proto/custom/)
SET(AIC_OP_INFO_CFG_OUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/op_impl/custom/ai_core/tbe/config/)

SET(AICPU_KERNEL_TARGET        cust_aicpu_kernels_mxindex_ascendsearch)
SET(AICPU_OP_INFO_CFG_OUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/op_impl/custom/cpu/config)
SET(AICPU_OP_IMPL_OUT_DIR      ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/op_impl/custom/cpu/aicpu_kernel/custom_impl/)
