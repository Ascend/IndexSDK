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

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly TOP_DIR=${CUR_DIR}/..
readonly MAKESELF_SCRIPT=${TOP_DIR}/makeself/makeself.sh
readonly MAKESELF_HEADER=${TOP_DIR}/makeself/makeself-header.sh
readonly Gcc4_PATH="/opt/rh/devtoolset-7/root/usr/bin"
readonly Gcc7_PATH="/opt/rh/devtoolset-index/root/usr/bin"
readonly FAISS_110_HOME="${FAISS_110_HOME:-/usr/local/faiss/faiss1.10.0}"
readonly FAISS_114_HOME="${FAISS_114_HOME:-/usr/local/faiss/faiss1.14.1}"
readonly DEFAULT_FAISS_ABI="${DEFAULT_FAISS_ABI:-faiss1.10}"
readonly MULTI_FAISS_PACKAGE="${MULTI_FAISS_PACKAGE:-OFF}"
readonly FAISS_VARIANT_WORK_DIR="${TOP_DIR}/build/faiss_variants"
readonly FAISS_COMBINED_OUTPUT_DIR="${TOP_DIR}/build/faiss_combined_output"

cd "${TOP_DIR}"
find ./ -name "*.sh" -exec dos2unix {} \;
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

function set_env()
{
    export CC=$GCC_HOME/bin/gcc
    export CXX=$GCC_HOME/bin/g++
    export PATH=$GCC_HOME/bin:/usr/local/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
}

function build_Retrieval()
{
    local gcc_type="$1"
    local faiss_home="${2:-${FAISS_110_HOME}}"
    export FAISS_HOME="${faiss_home}"
    echo "build Retrieval with FAISS_HOME=${FAISS_HOME}"
    cd "${TOP_DIR}" && cp -rf "vstar_great_impl/mix-index/publish_union.sh" ./
    chmod +x publish_union.sh
    ./publish_union.sh "${TOP_DIR}/vstar_great_impl" "${gcc_type}"
    rm -f "${TOP_DIR}/publish_union.sh"
    mkdir -p "${TOP_DIR}/build/output"
    mv "${TOP_DIR}"/Ascend-ascendsearch* "${TOP_DIR}/build/output"
    echo "build Retrieval success"
}

function build_vsa()
{
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    cd "${TOP_DIR}/vsa_hpp"
    chmod +x build.sh && ./build.sh
    echo "build vsa_hpp success"
}

function build_feature_retrieval()
{
    local gcc_type="$1"
    local faiss_home="${2:-${FAISS_110_HOME}}"
    local faiss_package_mode="${3:-}"
    local faiss_package_supported="${4:-}"
    local faiss_package_default="${5:-}"
    local build_ops="${6:-ON}"
    export FAISS_HOME="${faiss_home}"
    export BUILD_FEATURE_RETRIEVAL_OPS="${build_ops}"
    if [ -n "${faiss_package_mode}" ]; then
        export FAISS_PACKAGE_MODE="${faiss_package_mode}"
        export FAISS_PACKAGE_SUPPORTED="${faiss_package_supported}"
        export FAISS_PACKAGE_DEFAULT="${faiss_package_default}"
    else
        unset FAISS_PACKAGE_MODE FAISS_PACKAGE_SUPPORTED FAISS_PACKAGE_DEFAULT
    fi
    echo "build feature_retrieval with FAISS_HOME=${FAISS_HOME}"
    if [ ! -d "${TOP_DIR}/feature_retrieval/src/ascendfaiss/ops/cmake/util/makeself" ]; then
        mkdir -p "${TOP_DIR}/feature_retrieval/src/ascendfaiss/ops/cmake/util/makeself"
    fi
    cp -rf "${MAKESELF_SCRIPT}" "${MAKESELF_HEADER}" "${TOP_DIR}/feature_retrieval/src/ascendfaiss/ops/cmake/util/makeself/"

    if [ ! -d "${TOP_DIR}/feature_retrieval/secondparty" ]; then
        mkdir -p "${TOP_DIR}/feature_retrieval/secondparty"
    fi
    cp -rf "${TOP_DIR}"/build/output/Ascend-ascendsearch*.tar.gz "${TOP_DIR}/feature_retrieval/secondparty/"
    cp -rf "${TOP_DIR}/vsa_hpp/install/"* "${TOP_DIR}/feature_retrieval/secondparty/"

    cd "${TOP_DIR}/feature_retrieval/build" && bash build.sh "${gcc_type}"
    echo "build feature_retrieval success"
}

function validate_default_faiss_abi()
{
    if [ "${DEFAULT_FAISS_ABI}" != "faiss1.10" ] && [ "${DEFAULT_FAISS_ABI}" != "faiss1.14" ]; then
        echo "[ERROR] DEFAULT_FAISS_ABI only supports faiss1.10 or faiss1.14, current: ${DEFAULT_FAISS_ABI}"
        exit 1
    fi
}

function get_default_faiss_home()
{
    if [ "${DEFAULT_FAISS_ABI}" = "faiss1.14" ]; then
        echo "${FAISS_114_HOME}"
    else
        echo "${FAISS_110_HOME}"
    fi
}

function copy_faiss_variant_files()
{
    local src_pkg="$1"
    local dst_pkg="$2"
    local variant="$3"

    mkdir -p "${dst_pkg}/host/lib/${variant}"
    mkdir -p "${dst_pkg}/include/${variant}"

    for so_file in libascendfaiss.so libascendsearch.so; do
        if [ -e "${src_pkg}/host/lib/${so_file}" ]; then
            cp -P "${src_pkg}/host/lib/${so_file}" "${dst_pkg}/host/lib/${variant}/"
        else
            echo "[ERROR] ${src_pkg}/host/lib/${so_file} does not exist."
            exit 1
        fi
    done

    if [ -d "${src_pkg}/include/faiss" ]; then
        cp -a "${src_pkg}/include/faiss" "${dst_pkg}/include/${variant}/"
    else
        echo "[ERROR] ${src_pkg}/include/faiss does not exist."
        exit 1
    fi
}

function activate_default_faiss_variant_in_pkg()
{
    local pkg_dir="$1"
    local variant="${DEFAULT_FAISS_ABI}"

    if [ ! -d "${pkg_dir}/include/${variant}/faiss" ]; then
        echo "[ERROR] ${pkg_dir}/include/${variant}/faiss does not exist."
        exit 1
    fi

    rm -rf "${pkg_dir}/include/faiss" "${pkg_dir}/include/ascend"
    ln -s "${variant}/faiss" "${pkg_dir}/include/faiss"
    ln -s "faiss/ascend" "${pkg_dir}/include/ascend"

    for so_file in libascendfaiss.so libascendsearch.so; do
        if [ ! -f "${pkg_dir}/host/lib/${variant}/${so_file}" ]; then
            echo "[ERROR] ${pkg_dir}/host/lib/${variant}/${so_file} does not exist."
            exit 1
        fi
        rm -f "${pkg_dir}/host/lib/${so_file}"
        ln -s "${variant}/${so_file}" "${pkg_dir}/host/lib/${so_file}"
    done

    cat > "${pkg_dir}/faiss_versions.info" << EOF
mode:multi
supported:faiss1.10,faiss1.14
default:${variant}
faiss1.10:${FAISS_110_HOME}
faiss1.14:${FAISS_114_HOME}
EOF
}

function combine_faiss_variant_package()
{
    local faiss110_tar="$1"
    local faiss114_tar="$2"
    local out_tar="$3"
    local combine_root="$4"
    local package_stem
    package_stem=$(basename "${out_tar}" .tar.gz)
    local combine_dir="${combine_root}/${package_stem}"
    local faiss110_dir="${combine_dir}/faiss1.10"
    local faiss114_dir="${combine_dir}/faiss1.14"
    local out_dir="${combine_dir}/out"
    local base_dir="${faiss110_dir}"

    rm -rf "${combine_dir}"
    mkdir -p "${faiss110_dir}" "${faiss114_dir}" "${out_dir}"
    tar -xzf "${faiss110_tar}" -C "${faiss110_dir}"
    tar -xzf "${faiss114_tar}" -C "${faiss114_dir}"
    if [ "${DEFAULT_FAISS_ABI}" = "faiss1.14" ]; then
        base_dir="${faiss114_dir}"
    fi
    cp -a "${base_dir}/feature_retrieval" "${out_dir}/"

    local dst_pkg="${out_dir}/feature_retrieval"
    copy_faiss_variant_files "${faiss110_dir}/feature_retrieval" "${dst_pkg}" "faiss1.10"
    copy_faiss_variant_files "${faiss114_dir}/feature_retrieval" "${dst_pkg}" "faiss1.14"
    activate_default_faiss_variant_in_pkg "${dst_pkg}"

    mkdir -p "$(dirname "${out_tar}")"
    (
        cd "${out_dir}"
        tar -zcf "${out_tar}" feature_retrieval
    )
}

function combine_faiss_variant_packages()
{
    local gcc_type="$1"
    local work_dir="${FAISS_VARIANT_WORK_DIR}/${gcc_type}"
    local combine_root="${work_dir}/combined_work"

    mkdir -p "${FAISS_COMBINED_OUTPUT_DIR}"
    for faiss110_tar in "${work_dir}/faiss1.10"/*.tar.gz; do
        if [ ! -f "${faiss110_tar}" ]; then
            echo "[ERROR] no faiss1.10 feature_retrieval package found in ${work_dir}/faiss1.10"
            exit 1
        fi
        local tar_name
        tar_name=$(basename "${faiss110_tar}")
        local faiss114_tar="${work_dir}/faiss1.14/${tar_name}"
        if [ ! -f "${faiss114_tar}" ]; then
            echo "[ERROR] matching faiss1.14 feature_retrieval package does not exist: ${faiss114_tar}"
            exit 1
        fi
        combine_faiss_variant_package "${faiss110_tar}" "${faiss114_tar}" "${FAISS_COMBINED_OUTPUT_DIR}/${tar_name}" "${combine_root}"
    done
}

function build_feature_retrieval_multi_faiss()
{
    local gcc_type="$1"
    local work_dir="${FAISS_VARIANT_WORK_DIR}/${gcc_type}"
    local variant_configs=("faiss1.10:${FAISS_110_HOME}" "faiss1.14:${FAISS_114_HOME}")

    rm -rf "${work_dir}"
    mkdir -p "${work_dir}/faiss1.10" "${work_dir}/faiss1.14"

    for variant_config in "${variant_configs[@]}"; do
        local variant="${variant_config%%:*}"
        local faiss_home="${variant_config#*:}"

        if [ ! -d "${faiss_home}" ]; then
            echo "[ERROR] ${variant} FAISS_HOME does not exist: ${faiss_home}"
            exit 1
        fi

        echo "build ${gcc_type} package variant ${variant} with FAISS_HOME=${faiss_home}"
        local build_ops="OFF"
        if [ "${variant}" = "${DEFAULT_FAISS_ABI}" ]; then
            build_ops="ON"
        fi
        rm -rf "${TOP_DIR}/build/output"
        rm -rf "${TOP_DIR}/feature_retrieval/output"
        rm -rf "${TOP_DIR}/feature_retrieval/secondparty"

        build_Retrieval "${gcc_type}" "${faiss_home}"
        build_feature_retrieval "${gcc_type}" "${faiss_home}" "" "" "" "${build_ops}"

        if ! compgen -G "${TOP_DIR}/feature_retrieval/output/*.tar.gz" > /dev/null; then
            echo "[ERROR] no feature_retrieval package generated for ${variant}"
            exit 1
        fi
        mv "${TOP_DIR}/feature_retrieval/output"/*.tar.gz "${work_dir}/${variant}/"
    done

    combine_faiss_variant_packages "${gcc_type}"
}

function build_release_for_gcc()
{
    local gcc_type="$1"

    if [ "${MULTI_FAISS_PACKAGE}" = "ON" ]; then
        build_vsa
        build_feature_retrieval_multi_faiss "${gcc_type}"
    else
        local faiss_home
        faiss_home=$(get_default_faiss_home)
        if [ ! -d "${faiss_home}" ]; then
            echo "[ERROR] ${DEFAULT_FAISS_ABI} FAISS_HOME does not exist: ${faiss_home}"
            exit 1
        fi
        echo "build single faiss package variant ${DEFAULT_FAISS_ABI} with FAISS_HOME=${faiss_home}"
        build_Retrieval "${gcc_type}" "${faiss_home}" &
        local retrieval_pid=$!
        build_vsa &
        local vsa_pid=$!
        local build_failed=0
        wait "${retrieval_pid}" || build_failed=1
        wait "${vsa_pid}" || build_failed=1
        if [ "${build_failed}" -ne 0 ]; then
            echo "[ERROR] build_Retrieval or build_vsa failed."
            exit 1
        fi
        build_feature_retrieval "${gcc_type}" "${faiss_home}" "single" "${DEFAULT_FAISS_ABI}" "${DEFAULT_FAISS_ABI}"
    fi
}

function run_ut()
{
    rm -rf "${TOP_DIR}/feature_retrieval/opensource/mockcpp"
    rm -rf "${TOP_DIR}/feature_retrieval/secondparty/huawei_secure_c"
    cd "${TOP_DIR}/feature_retrieval/ut"
    chmod +x *.sh
    ./config_env.sh && ./ci_run.sh -c on -a on
}

function clean()
{
    set +e
    cd "${TOP_DIR}/vsa_hpp" && chmod +x build.sh && ./build.sh -t clean
    cd "${TOP_DIR}/vsa_hpp/3rdparty/hcps" && chmod +x build.sh && ./build.sh -t clean
    cd "${TOP_DIR}/vsa_hpp/3rdparty/hcps/3rdparty/memory_bridge" && chmod +x build.sh && ./build.sh -t clean
    rm -rf "${TOP_DIR}/build/output"
    rm -rf "${FAISS_VARIANT_WORK_DIR}"
    rm -rf "${FAISS_COMBINED_OUTPUT_DIR}"
    rm -rf "${TOP_DIR}/feature_retrieval/output"
    rm -rf "${TOP_DIR}/feature_retrieval"/build_gcc*
    rm -rf "${TOP_DIR}/feature_retrieval/secondparty"
    set -e
}

function package()
{
    if compgen -G "${FAISS_COMBINED_OUTPUT_DIR}/*.tar.gz" > /dev/null; then
        rm -rf "${TOP_DIR}/feature_retrieval/output"
        mkdir -p "${TOP_DIR}/feature_retrieval/output"
        cp -f "${FAISS_COMBINED_OUTPUT_DIR}"/*.tar.gz "${TOP_DIR}/feature_retrieval/output/"
    fi
    bash "${CUR_DIR}/package.sh"
}

function main()
{
    validate_default_faiss_abi
    clean
    if [ -d "${Gcc4_PATH}" ]; then
        echo "build with Gcc4 in ${Gcc4_PATH}..."
        export GCC_HOME=/opt/rh/devtoolset-7/root/usr
        set_env
        build_release_for_gcc Gcc4
    fi
    if [ -d "${Gcc7_PATH}" ]; then
        echo "build with Gcc7 in ${Gcc7_PATH}..."
        export GCC_HOME=/opt/rh/devtoolset-index/root/usr
        set_env
        build_release_for_gcc Gcc7
    fi
    if [ ! -d "${Gcc4_PATH}" ] && [ ! -d "${Gcc7_PATH}" ]; then
        echo "build with system default Gcc..."
        build_release_for_gcc Gcc7
    fi
    package
}

BUILDTYPE=$1
bash ${CUR_DIR}/install_deps.sh ${BUILDTYPE}
if [ "${BUILDTYPE}" = "ut" ]; then
    run_ut
else
    main
fi
