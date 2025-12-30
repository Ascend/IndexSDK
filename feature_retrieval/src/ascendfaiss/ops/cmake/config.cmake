# 算子部署目录结构如下：
#└── opp    //算子库目录
#    └── vendors          //自定义算子所在目录
#        ├── config.ini   //自定义算子优先级配置文件
#        └── mxIndex      //检索自定义算子目录
#            ├── op_impl
#            │   ├── ai_core
#            │   │   └── tbe
#            │   │       ├── config
#            │   │       │   ├── ${soc_version}     //昇腾AI处理器类型
#            │   │       │   │   └── aic-${soc_version}-ops-info.json     //TBE自定义算子信息库
#            │   │       └── mxIndex_impl           //TBE自定义算子实现代码文件
#            │   │           ├── XXXX.py
#            │   ├── cpu                            //AI CPU自定义算子实现库及算子信息库所在目录
#            │   │   ├── aicpu_kernel
#            │   │   │   └── impl
#            │   │   │       └── libcust_aicpu_kernels_mxindex.so  //AI CPU自定义算子实现库文件
#            │   │   └── config
#            │   │       └── cust_aicpu_kernel.json                //AI CPU自定义算子信息库文件
#            │   └── vector_core   //此目录预留，无需关注
#            └── op_proto           //自定义算子原型库所在目录
#                └── libcust_op_proto.so

# 注意注意：修改算子安装目录或者构建方式时，注意build/build.sh脚本中ivfsp中算子的处理，需要保持一致

# Set compiling options
ADD_COMPILE_OPTIONS(-std=c++11  -fPIC  -fstack-protector-all  -Wall  -Wreturn-type  -D_FORTIFY_SOURCE=2  -O2)
SET(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS}  -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack  -s")
SET(CMAKE_EXE_LINKER_FLAGS      "${CMAKE_EXE_LINKER_FLAGS}  -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack  -pie  -s")
INCLUDE_DIRECTORIES(${ASCEND_TOOLKIT_PATH}/atc/include)

# Set vendor name
SET(VENDOR_NAME mxIndex)

# Set run package name, this package is used to install OPP.
SET(RUN_TARGET custom_opp_${CMAKE_SYSTEM_PROCESSOR}.run)

# Parser
SET(INI_2_JSON_PY  ${CMAKE_CURRENT_LIST_DIR}/util/parse_ini_to_json.py)
SET(ASENDC_IMPL_BUILD_PY ${CMAKE_CURRENT_LIST_DIR}/util/ascendc_impl_build.py)
SET(GEN_VERSION_FINO_SH ${CMAKE_CURRENT_LIST_DIR}/util/gen_version_info.sh)
SET(MERGE_JSON_PY  ${CMAKE_CURRENT_LIST_DIR}/util/merge_json.py)

SET(OPP_INSTALL_DIR          ${CMAKE_CURRENT_BINARY_DIR}/makepkg/packages/vendors/${VENDOR_NAME})
# Set local package paths
SET(OP_PROTO_TARGET          cust_op_proto)
SET(OP_PROTO_TARGET_OUT_DIR  ${OPP_INSTALL_DIR}/op_proto/)
SET(AIC_OP_INFO_CFG_OUT_DIR  ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/config/)

SET(AICPU_KERNEL_TARGET        cust_aicpu_kernels_mxindex)
SET(AICPU_OP_INFO_CFG_OUT_DIR  ${OPP_INSTALL_DIR}/op_impl/cpu/config)
SET(AICPU_OP_IMPL_OUT_DIR      ${OPP_INSTALL_DIR}/op_impl/cpu/aicpu_kernel/impl/)
