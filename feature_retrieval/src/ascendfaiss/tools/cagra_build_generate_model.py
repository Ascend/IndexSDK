#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the IndexSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

IndexSDK is licensed under Mulan PSL v2.
-------------------------------------------------------------------------
"""

import argparse
import os

import common as utils
from common import OpJsonGenerator

# Keep these graph-build limits synchronized with ops/ascendc/op_host/cagra_build.cpp and
# ops/ascendc/op_kernel/cagra_build_kernel.h. A5_CORE_NUM must also match tools/common.py.
A5_CORE_NUM = 56
MAX_BUILD_DIM = 3072
MAX_INTERMEDIATE_DEGREE = 128
MAX_SAMPLE_DEGREE = 32
SIZE_ENTRY_COUNT = 4
MAX_DATA_SIZE = 0x7FFFFFFF


def parse_args():
    parser = argparse.ArgumentParser(description='generate AICore models for CAGRA graph building')
    utils.op_common_parse(
        parser, "--cores", 'core_num', A5_CORE_NUM, int, f"AI Core number (A5 default: {A5_CORE_NUM})"
    )
    utils.op_common_parse(parser, "-d", 'dim', 128, int, f"vector dimension in range [1, {MAX_BUILD_DIM}]")
    utils.op_common_parse(parser, "-n", 'data_size', 10000, int, "number of base vectors")
    utils.op_common_parse(parser, "-i", 'intermediate_degree', MAX_INTERMEDIATE_DEGREE, int, "NN-Descent graph degree")
    utils.op_common_parse(parser, "-graph", 'graph_degree', 64, int, "final CAGRA graph degree")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "process ID used in model names")
    utils.op_common_parse(
        parser,
        "-t",
        'npu_type',
        "Ascend950PR",
        str,
        "NPU type. Ascend950PR is canonical; Ascend950 is accepted as an alias.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.dim <= 0 or args.dim > MAX_BUILD_DIM:
        raise ValueError(f"CAGRA graph builder requires dim in range [1, {MAX_BUILD_DIM}]")
    if args.data_size <= 1 or args.data_size > MAX_DATA_SIZE:
        raise ValueError(f"data_size must be in range [2, {MAX_DATA_SIZE}]")
    if not 0 < args.graph_degree <= args.intermediate_degree < args.data_size:
        raise ValueError("degrees must satisfy 0 < graph_degree <= intermediate_degree < data_size")
    if args.intermediate_degree > MAX_INTERMEDIATE_DEGREE:
        raise ValueError(f"current CAGRA graph builder requires intermediate_degree <= {MAX_INTERMEDIATE_DEGREE}")
    if args.core_num != A5_CORE_NUM:
        raise ValueError(f"Ascend950 CAGRA graph build currently requires --cores {A5_CORE_NUM}")


def generate_operators(dim, data_size, intermediate_degree, graph_degree):
    sample_degree = min(MAX_SAMPLE_DEGREE, intermediate_degree)
    data_shape = [data_size, dim]
    intermediate_shape = [data_size, intermediate_degree]
    sample_shape = [data_size, sample_degree]
    sizes_shape = [data_size, SIZE_ENTRY_COUNT]
    count_shape = [data_size]
    graph_shape = [data_size, graph_degree]
    operators = []

    generator = OpJsonGenerator("CagraNndInit")
    generator.add_input("ND", data_shape, "float32")
    generator.add_output("ND", intermediate_shape, "uint32")
    generator.add_output("ND", intermediate_shape, "float32")
    generator.add_attr("intermediate_degree", "required", "int", intermediate_degree)
    operators.append(generator.generate_obj())

    generator = OpJsonGenerator("CagraNndSampleReverse")
    generator.add_input("ND", intermediate_shape, "uint32")
    for _ in range(SIZE_ENTRY_COUNT):
        generator.add_output("ND", sample_shape, "uint32")
    generator.add_output("ND", sizes_shape, "uint32")
    operators.append(generator.generate_obj())

    generator = OpJsonGenerator("CagraNndLocalJoin")
    generator.add_input("ND", data_shape, "float32")
    for _ in range(SIZE_ENTRY_COUNT):
        generator.add_input("ND", sample_shape, "uint32")
    generator.add_input("ND", sizes_shape, "uint32")
    generator.add_output("ND", intermediate_shape, "uint32")
    generator.add_output("ND", intermediate_shape, "float32")
    generator.add_output("ND", count_shape, "uint32")
    generator.add_attr("candidate_degree", "required", "int", intermediate_degree)
    operators.append(generator.generate_obj())

    generator = OpJsonGenerator("CagraNndUpdate")
    generator.add_input("ND", intermediate_shape, "uint32")
    generator.add_input("ND", intermediate_shape, "float32")
    generator.add_input("ND", sample_shape, "uint32")
    generator.add_input("ND", sizes_shape, "uint32")
    generator.add_input("ND", intermediate_shape, "uint32")
    generator.add_input("ND", intermediate_shape, "float32")
    generator.add_input("ND", count_shape, "uint32")
    generator.add_output("ND", intermediate_shape, "uint32")
    generator.add_output("ND", intermediate_shape, "float32")
    generator.add_output("ND", [1], "uint32")
    operators.append(generator.generate_obj())

    generator = OpJsonGenerator("CagraPruneReverse")
    generator.add_input("ND", intermediate_shape, "uint32")
    generator.add_output("ND", graph_shape, "uint32")
    generator.add_output("ND", graph_shape, "uint32")
    generator.add_output("ND", count_shape, "uint32")
    generator.add_attr("output_degree", "required", "int", graph_degree)
    operators.append(generator.generate_obj())

    generator = OpJsonGenerator("CagraMerge")
    generator.add_input("ND", graph_shape, "uint32")
    generator.add_input("ND", graph_shape, "uint32")
    generator.add_input("ND", count_shape, "uint32")
    generator.add_output("ND", graph_shape, "uint32")
    operators.append(generator.generate_obj())

    return operators


def compile_operator(operator, index, process_id, config_path, soc_version):
    op_name = operator['op']
    model_name = f"ascendc_cagra_graph_build_{index}_{op_name.lower()}_pid{process_id}"
    file_path = os.path.join(config_path, f"{model_name}.json")
    utils.generate_op_config([operator], file_path)

    output_path = './op_models'
    before = {
        file_name: os.stat(os.path.join(output_path, file_name)).st_mtime_ns
        for file_name in os.listdir(output_path)
        if file_name.endswith('.om')
    }
    utils.atc_model(model_name, soc_version)
    generated = []
    for file_name in os.listdir(output_path):
        if not file_name.endswith('.om') or f"_{op_name}_" not in file_name:
            continue
        modified_time = os.stat(os.path.join(output_path, file_name)).st_mtime_ns
        if file_name not in before or before[file_name] != modified_time:
            generated.append(file_name)
    if not generated:
        raise RuntimeError(f"ATC did not generate an OM model for {op_name}")
    print(f"  generated {op_name}: {', '.join(sorted(generated))}")
    return file_path


def generate_models():
    args = parse_args()
    validate_args(args)
    utils.set_env()
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    utils.get_core_num_by_npu_type(args.core_num, args.npu_type)

    config_path = utils.get_config_path('.')
    operators = generate_operators(
        args.dim,
        args.data_size,
        args.intermediate_degree,
        args.graph_degree,
    )
    config_files = []
    for index, operator in enumerate(operators):
        config_files.append(compile_operator(operator, index, args.process_id, config_path, soc_version))

    print("CAGRA graph-build models generated successfully")
    print(f"  N={args.data_size}, D={args.dim}, intermediate_degree={args.intermediate_degree}")
    print(f"  graph_degree={args.graph_degree}, cores={args.core_num}")
    print(f"  soc_version={soc_version}")
    print(f"  configs={len(config_files)} files under {config_path}")
    print("  output=./op_models")


if __name__ == '__main__':
    generate_models()
