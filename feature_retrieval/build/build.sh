#!/bin/bash
# -------------------------------------------------------------------------
# This file is part of the IndexSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# IndexSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

set -e
gcc_version=$1
enable_hitest=$2
build_ascend_search=true
build_ock=true
build_disk_search=true
readonly PROJECT_ROOT_FOLDER=${PWD}/..
readonly PROJECT_ZIP_PATH="${PROJECT_ROOT_FOLDER}"/output
readonly PROJECT_SRC_PATH="${PROJECT_ROOT_FOLDER}"/src/ascendfaiss
readonly ASCEND_SEARCH_HOME="${PROJECT_ROOT_FOLDER}"/secondparty/ascendsearch
readonly DISK_SEARCH_HOME="${PROJECT_ROOT_FOLDER}"/secondparty/disksearch
readonly OCK_HOME="${PROJECT_ROOT_FOLDER}"/secondparty
readonly VERSION=$(sed -n 's/^version:[[:space:]]*//p' "${PROJECT_ROOT_FOLDER}"/../../ci/config/config.ini)
readonly ARCH="$(uname -m)"
OP_PROTO_TARGET_OUT_DIR="${PROJECT_SRC_PATH}/ops/build/makepkg/packages/vendors/mxIndex/op_proto/"
AICPU_OP_IMPL_OUT_DIR="${PROJECT_SRC_PATH}/ops/build/makepkg/packages/vendors/mxIndex/op_impl/cpu/aicpu_kernel/impl/"

info() { echo -e "[INFO ] $1" ; }
error() { echo >&2 -e "[ERROR ] $1" ; }

install_redction_train_shell()
{
    reduction_install_dir="${PROJECT_ZIP_PATH}"/feature_retrieval/tools/train/reduction/
    [ -d "${reduction_install_dir}" ] || mkdir -p "${reduction_install_dir}"

    cp "$1"/DR/* "${reduction_install_dir}"
}

build_train()
{
    if [ "${build_ascend_search}" = "true" -a -d "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}" ]; then
        # 拷贝ivfsp训练码本需要的so到train目录
        ivfsp_train_dir="${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}/train"
        mkdir -p "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/train/
        cp "${ivfsp_train_dir}"/SP.py "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/train/
        cp "${ivfsp_train_dir}"/train.py "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/train/ivfsp_train_codebook.py
        cp "${ivfsp_train_dir}"/train.sh "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/train/ivfsp_train_codebook_example.sh
        install_redction_train_shell "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"
    fi

    if [ "${build_ascend_search}" = "true" -a -d "${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}" ]; then
        # 拷贝mix-search训练码本需要的py到train目录
        mix_train_dir="${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}/train"
        cp "${mix_train_dir}"/*.py "${PROJECT_ZIP_PATH}/feature_retrieval/tools/train/"
    fi
}

prepare_zip_src()
{
    [ -d "${PROJECT_ZIP_PATH}"/feature_retrieval/modelpath ] || mkdir -p "${PROJECT_ZIP_PATH}"/feature_retrieval/modelpath
    [ -d "${PROJECT_ZIP_PATH}"/feature_retrieval/ops ] || mkdir "${PROJECT_ZIP_PATH}"/feature_retrieval/ops
    [ -d "${PROJECT_ZIP_PATH}"/feature_retrieval/script ] || mkdir "${PROJECT_ZIP_PATH}"/feature_retrieval/script
    [ -d "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib ] || mkdir -p "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib
    cp "${PROJECT_SRC_PATH}"/script/uninstall.sh "${PROJECT_ZIP_PATH}"/feature_retrieval/script
    cp -r "${PROJECT_SRC_PATH}"/tools "${PROJECT_ZIP_PATH}"/feature_retrieval

    if [ "${build_ascend_search}" = "true" ]; then
        ascend_search_so_file="${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}/../lib/libascendsearch.so"
        cp "${ascend_search_so_file}" "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib/
    fi

    if [ "${build_ascend_search}" = "true" -a -d "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}" ]; then
        # 拷贝ivfsp的文件到${PROJECT_ZIP_PATH}/feature_retrieval下面
        ivfsp_generate_model_file="${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}/tools/ivfsp_generate_model.py"
        ivfsp_generate_pyacl_model_file="${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}/tools/ivfsp_generate_pyacl_model.py"
        ivfsp_aicpu_generate_model_file="${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}/tools/aicpu_generate_model.py"
        cp "${ivfsp_generate_model_file}" "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/
        cp "${ivfsp_generate_pyacl_model_file}" "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/
        cp "${ivfsp_aicpu_generate_model_file}" "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/ivfsp_aicpu_generate_model.py
    fi

    if [ "${build_ascend_search}" = "true" -a -d "${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}" ]; then
        # 拷贝mix-search的文件到${PROJECT_ZIP_PATH}/feature_retrieval下面
        mix_generate_model_file="${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}/tools/vstar_generate_models.py"
        cp "${mix_generate_model_file}" "${PROJECT_ZIP_PATH}"/feature_retrieval/tools/vstar_generate_models.py
    fi

    cd "${PROJECT_ZIP_PATH}"/feature_retrieval
    ls -al
    ln -sf faiss/ascend include/ascend
    cd -
    build_train
}

make_zip_src()
{
    cd "${PROJECT_ZIP_PATH}"
    tar -zcf "$1".tar.gz feature_retrieval
}

prepare_disksearch()
{
    cp -r "${ASCEND_SEARCH_HOME}"/"${npu_device_type}"/lib "${DISK_SEARCH_HOME}"/
}

# 根据x86/arm, gcc版本, NPU_TYPE来提取ivfsp需要的
# 检查依赖的ivfsp的包是否存在，以及解压到指定的ivfsp_home目录
prepare_ascendsearch_with_npu_type()
{
    npu_device_type=$1
    ASCEND_SEARCH_VERSION="${VERSION}_gcc7.3.0"
    if [ "${gcc_version}" = "Gcc4" ]; then
        ASCEND_SEARCH_VERSION="${VERSION}_gcc4.8.5"
    fi

    ascend_search_package_name="Ascend-ascendsearch_${ASCEND_SEARCH_VERSION}-linux-${ARCH}-${npu_device_type}.tar.gz"
    ascend_search_package_file="${ASCEND_SEARCH_HOME}/../${ascend_search_package_name}"

    if [ ! -f "${ascend_search_package_file}" ]; then
        error "AscendSearch package: ${ascend_search_package_file}, is not exist!"
        return 1
    fi

    info "AscendSearch package: ${ascend_search_package_file} exist!"
    if [ ! -d "${ASCEND_SEARCH_HOME}"/"${npu_device_type}" ]; then
        mkdir -p "${ASCEND_SEARCH_HOME}"/${npu_device_type}
    else
        cd "${ASCEND_SEARCH_HOME}"/.. && rm -rf ascendsearch/${npu_device_type}/*
    fi

    if [ ! -d "${DISK_SEARCH_HOME}" ]; then
        mkdir -p "${DISK_SEARCH_HOME}"
    else
        cd "${DISK_SEARCH_HOME}"/.. && rm -rf disksearch/*
    fi

    tar -xf ${ascend_search_package_file} -C "${ASCEND_SEARCH_HOME}"/${npu_device_type}

    if [ "${build_disk_search}" = "true" ]; then
        prepare_disksearch
    fi

    return 0
}

prepare_ascendsearch()
{
    if [ "${build_ascend_search}" != "true" ]; then
        return 0
    fi

    cd "${ASCEND_SEARCH_HOME}"/.. && rm -rf ascendsearch/*
    prepare_ascendsearch_with_npu_type 310 || true
    prepare_ascendsearch_with_npu_type 310P || true
}

# 拷贝描述文件
cp_ivfsp_ops_ini()
{
    if [ "${build_ascend_search}" != "true" ]; then
        return 0
    fi

    # DEFAULT_IVFSP_HOME_WITH_NPU_TYPE
    if [ -d "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}" ]; then
        # 拷贝tik/aicpu描述文件、tik算子文件
        ivfsp_tbe_aicore_ascend310=(distance_compute_qc.ini distance_ivf_sp_int_l2.ini distance_ivf_sp_int_l2_mins.ini \
            distance_masked_ivf_sp_int_l2_mins.ini ivf_cid_filter3.ini matmul_compute.ini fp_to_fp16.ini)
        ivfsp_tbe_aicore_ascend310p=(corr_compute.ini distance_compute_qc.ini distance_ivf_sp_int_l2.ini \
            distance_ivf_sp_int_l2_mins.ini distance_masked_ivf_sp_int_l2_mins.ini ivf_cid_filter3.ini matmul_compute.ini fp_to_fp16.ini)
        ivfsp_tbe_aicore_ascend710=(corr_compute.ini distance_compute_qc.ini distance_ivf_sp_int_l2.ini \
            distance_ivf_sp_int_l2_mins.ini distance_masked_ivf_sp_int_l2_mins.ini ivf_cid_filter3.ini matmul_compute.ini fp_to_fp16.ini)
        for ivfsp_tbe_aicore_file in ${ivfsp_tbe_aicore_ascend310[@]}
        do
            cp "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"/ops/tbe/op_info_cfg/ai_core/ascend310/"${ivfsp_tbe_aicore_file}" \
                "${PROJECT_SRC_PATH}"/ops/tbe/op_info_cfg/ai_core/ascend310/"${ivfsp_tbe_aicore_file}"
        done

        for ivfsp_tbe_aicore_file in ${ivfsp_tbe_aicore_ascend310p[@]}
        do
            cp "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"/ops/tbe/op_info_cfg/ai_core/ascend310p/"${ivfsp_tbe_aicore_file}" \
                "${PROJECT_SRC_PATH}"/ops/tbe/op_info_cfg/ai_core/ascend310p/"${ivfsp_tbe_aicore_file}"
        done

        for ivfsp_tbe_aicore_file in ${ivfsp_tbe_aicore_ascend710[@]}
        do
            cp "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"/ops/tbe/op_info_cfg/ai_core/ascend710/"${ivfsp_tbe_aicore_file}" \
                "${PROJECT_SRC_PATH}"/ops/tbe/op_info_cfg/ai_core/ascend710/"${ivfsp_tbe_aicore_file}"
        done

        ivfsp_tbe_impl_files=(corr_compute.py distance_compute_qc.py distance_ivf_sp_int_l2.py \
            distance_ivf_sp_int_l2_mins.py distance_masked_ivf_sp_int_l2_mins.py ivf_cid_filter3.py matmul_compute.py fp_to_fp16.py)
        for ivfsp_tbe_impl_file in ${ivfsp_tbe_impl_files[@]}
        do
            cp "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"/ops/tbe/impl/"${ivfsp_tbe_impl_file}" \
                "${PROJECT_SRC_PATH}"/ops/tbe/impl/"${ivfsp_tbe_impl_file}"
        done

        ivfsp_aicpu_files=(topk_ivfsp_l1.ini topk_multisearch_ivf.ini topk_spsq.ini topk_multisearch_ivf_v2.ini \
            topk_ivf_sp.ini transdata_raw_sp.ini transdata_shaped_sp.ini vec_l2sqr_sp.ini)
        for ivfsp_aicpu_file in ${ivfsp_aicpu_files[@]}
        do
            cp "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"/ops/cpukernel/op_info_cfg/aicpu_kernel/"${ivfsp_aicpu_file}" \
                "${PROJECT_SRC_PATH}"/ops/cpukernel/op_info_cfg/aicpu_kernel/"${ivfsp_aicpu_file}"
        done

        return 0
    else
        return 1
    fi
}

cp_ivfsp_opsso_with_npu_type()
{
    if [ "${build_ascend_search}" != "true" ]; then
        return 0
    fi

    npu_device_type=$1
    if [ -d "${ASCEND_SEARCH_HOME}/${npu_device_type}/IVFSP" ]; then
        # 拷贝ivfsp的aicore接口so文件、aicpu算子so文件、tik/aicpu描述文件、tik算子文件
        ivfsp_aicore_so_file="${ASCEND_SEARCH_HOME}/${npu_device_type}/IVFSP/ops/libcust_op_proto_ascendsearch.so"
        ivfsp_aicpu_so_file="${ASCEND_SEARCH_HOME}/${npu_device_type}/IVFSP/ops/libcust_aicpu_kernels_ascendsearch.so"
        mkdir -p "${OP_PROTO_TARGET_OUT_DIR}"
        mkdir -p "${AICPU_OP_IMPL_OUT_DIR}"
        cp "${ivfsp_aicore_so_file}" "${OP_PROTO_TARGET_OUT_DIR}"/libcust_op_proto_ascendsearch.so
        cp "${ivfsp_aicpu_so_file}" "${AICPU_OP_IMPL_OUT_DIR}"/libcust_aicpu_kernels_ascendsearch.so
        return 0
    else
        return 1
    fi
}

cp_mix_ops_project()
{
    # 拷贝mix_search的算子源文件
    cp -r "${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}"/ops/ascend_c/AscendcOps/op_host/*.h "${PROJECT_SRC_PATH}"/ops/ascendc/op_host/
    cp -r "${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}"/ops/ascend_c/AscendcOps/op_host/*.cpp "${PROJECT_SRC_PATH}"/ops/ascendc/op_host/
    cp -r "${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}"/ops/ascend_c/AscendcOps/op_kernel/*.cpp "${PROJECT_SRC_PATH}"/ops/ascendc/op_kernel/
    cp -r "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}"/ops/cpukernel/op_info_cfg/aicpu_kernel/ivf_* "${PROJECT_SRC_PATH}"/ops/cpukernel/op_info_cfg/aicpu_kernel/
}


clean_ivfsp_opsso_with_npu_type()
{
    if [ "${build_ascend_search}" != "true" ]; then
        return 0
    fi

    [ -f "${OP_PROTO_TARGET_OUT_DIR}"/libcust_op_proto_ascendsearch.so ] && rm "${OP_PROTO_TARGET_OUT_DIR}"/libcust_op_proto_ascendsearch.so
    [ -f "${AICPU_OP_IMPL_OUT_DIR}"/libcust_aicpu_kernels_ascendsearch.so ] && rm "${AICPU_OP_IMPL_OUT_DIR}"/libcust_aicpu_kernels_ascendsearch.so
}

build_ops()
{
    npu_device_type=$1
    mkdir -p "${PROJECT_ZIP_PATH}"/feature_retrieval/ops/
    cd "${PROJECT_SRC_PATH}"/ops && rm -rf build && mkdir ./build
    cp_ivfsp_opsso_with_npu_type ${npu_device_type} || true
    cp_mix_ops_project ${npu_device_type}
    cmake -B build -DNPU_TYPE=${npu_device_type} -DASCEND_SEARCH=${build_ascend_search} && cd build && make -j
    cp *.run "${PROJECT_ZIP_PATH}"/feature_retrieval/ops
    make_zip_src ${RELEASE_PKG_NAME}-${npu_device_type}
    clean_ivfsp_opsso_with_npu_type "${npu_device_type}" || true
}

install_ock()
{
    cp -P "${OCK_HOME}"/vsa/lib/libock_vsa_hpp.so* "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib/
    cp -P "${OCK_HOME}"/hcps/lib/libock_hcps_pier.so* "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib/
    cp -P "${OCK_HOME}"/memory_bridge/lib/libock_hmm.so* "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib/
}

install_disk_search()
{
    cp -P "${DISK_SEARCH_HOME}"/lib/libdisksearch_opgs.so "${PROJECT_ZIP_PATH}"/feature_retrieval/host/lib/
}

build_install()
{
    if [ -d "${build_binary_dir}" ]; then
        rm -rf "${build_binary_dir}"/* && cd "${build_binary_dir}"
    else
        mkdir "${build_binary_dir}" && cd "${build_binary_dir}"
    fi

    is_build_ascend_search=OFF
    # 在cmake文件中会用到不区分npu_type的${ivfsp_home_dir}/lib和${ivfsp_home_dir}/include
    if [ "${build_ascend_search}" = "true" -a -d "${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}" -a -d "${DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE}" ]; then
        is_build_ascend_search=ON
    fi
        
    cmake -S ${PROJECT_SRC_PATH} -DPROTOBUF_HOME="${PROJECT_ROOT_FOLDER}"/../opensource/opensource/temp/"${protobuf_dir}" \
        -DPROTOBUF_AARCH64_HOME="${PROJECT_ROOT_FOLDER}"/../opensource/opensource/temp/protobuf_aarch64 \
        -DBUILD_OPS=OFF -DCMAKE_INSTALL_PREFIX="${PROJECT_ZIP_PATH}"/feature_retrieval \
        -DBUILD_ASCEND_SEARCH="${is_build_ascend_search}" -DASCEND_SEARCH_HOME="${DEFAULT_IVFSP_HOME_WITH_NPU_TYPE}/.." \
        -DBUILD_OCK="${build_ock}" -DOCK_HOME="${OCK_HOME}"

    if [ "${enable_hitest}" = "hitest" ];then
        export PATH=$PATH:/opt/hitest/linux_avatar_arm_64:/opt/hitest/linux_avatar_x86_64
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hitest/linux_avatar_arm_64:/opt/hitest/linux_avatar_x86_64
        cmake . -DHITEST=ON
    fi

    make -j && cmake --install .

    prepare_zip_src

    if [ "${build_ock}" = "true" ]; then
        install_ock
    fi

    if [ "${build_disk_search}" = "true" ]; then
        install_disk_search
    fi

    cp_ivfsp_ops_ini
    build_ops 310
    build_ops 310P
    build_ops 910B

    rm -rf "${PROJECT_ZIP_PATH}"/feature_retrieval
}

main()
{
    cd "${PROJECT_ROOT_FOLDER}"/build
    find ./ -name "*.sh" -exec chmod +x {} +
    find ./ -name "*.sh" -exec dos2unix {} +

    cd "${PROJECT_SRC_PATH}"/
    find ./ -name "*.sh" -exec dos2unix {} +
    find ./ -name "*.sh" -exec chmod +x {} +

    if [ "${gcc_version}" = "Gcc4" ];then
        protobuf_dir="protobuf_gcc4"
        build_binary_dir="${PROJECT_ROOT_FOLDER}/build_gcc4"
        RELEASE_PKG_NAME=Ascend-featureretrieval_${VERSION}_gcc4.8.5-linux-${ARCH}
    else
        build_binary_dir="${PROJECT_ROOT_FOLDER}/build_gcc7"
        RELEASE_PKG_NAME=Ascend-featureretrieval_${VERSION}_gcc7.3.0-linux-${ARCH}
    fi

    [ ! -d "${ASCEND_SEARCH_HOME}" ] && mkdir -p "${ASCEND_SEARCH_HOME}"
    prepare_ascendsearch
    ascend_search_home_310="${ASCEND_SEARCH_HOME}/310"
    ascend_search_home_310P="${ASCEND_SEARCH_HOME}/310P"
    if [ -d "${ascend_search_home_310P}" ]; then
        readonly DEFAULT_IVFSP_HOME_WITH_NPU_TYPE="${ascend_search_home_310P}/IVFSP"
        readonly DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE="${ascend_search_home_310P}/mix-index"
    elif [ -d "${ascend_search_home_310}" ]; then
        readonly DEFAULT_IVFSP_HOME_WITH_NPU_TYPE="${ascend_search_home_310}/IVFSP"
        readonly DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE="${ascend_search_home_310}/mix-index"
    else
        readonly DEFAULT_IVFSP_HOME_WITH_NPU_TYPE="${ascend_search_home_310P}/IVFSP"
        readonly DEFAULT_MIX_SEARCH_HOME_WITH_NPU_TYPE="${ascend_search_home_310P}/mix-index"
    fi

    build_install
}

main