#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the IndexSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

IndexSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os
import argparse
import common as utils
from common import OpJsonGenerator


def arg_parse():
    """
    Parse arguments to the operator model
    """

    parser = argparse.ArgumentParser(
        description='generate aicore operator model for CAGRA graph building')

    utils.op_common_parse(parser, "-d", 'dim', 128, int, "Dimension of data vectors")
    utils.op_common_parse(parser, "-n", 'data_size', 10000, int, "Number of data points")
    utils.op_common_parse(parser, "-g", 'graph_degree', 32, int, "Graph degree (K value)")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Process ID for operator naming")
    utils.op_common_parse(parser, "-t", 'npu_type', "310P", str, "NPU type, 310 / 310P / 910B1 / 910B2 / 910B3 / 910B4 / 910_{NPU Name} / Ascend950PR. 310P by default")

    return parser.parse_args()


def generate_preprocess_data_json(dim, data_size, file_path):
    preprocess_obj = []
    generator = OpJsonGenerator("PreprocessData")
    
    # 输入：原始数据(float32), 维度(int32), 列表偏移量(uint64)
    generator.add_input("ND", [data_size, dim], "float32")
    generator.add_input("ND", [1], "int32")
    generator.add_input("ND", [1], "uint64")
    
    # 输出：预处理后数据(float16), L2范数(float32)
    generator.add_output("ND", [data_size, dim], "float16")
    generator.add_output("ND", [data_size], "float32")
    
    preprocess_obj.append(generator.generate_obj())
    utils.generate_op_config(preprocess_obj, file_path)


def generate_add_reverse_edges_json(data_size, graph_degree, file_path):
    reverse_edges_obj = []
    generator = OpJsonGenerator("AddReverseEdges")
    
    # 输入：图邻居列表(int32), 取样数量(uint64), 边数量(int32)
    generator.add_input("ND", [data_size, graph_degree], "int32")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [data_size], "int32")
    
    # 输出：反向边后的邻居列表(int32), 边数量(int32)
    generator.add_output("ND", [data_size, graph_degree], "int32")
    generator.add_output("ND", [data_size], "int32")
    
    reverse_edges_obj.append(generator.generate_obj())
    utils.generate_op_config(reverse_edges_obj, file_path)


def generate_local_join_json(dim, data_size, graph_degree, file_path):
    local_join_obj = []
    generator = OpJsonGenerator("LocalJoin")
    
    # 输入：新图邻居列表(int32), 新图反向邻居列表(int32), 新图前向边数量(int32), 新图反向边数量(int32),
    #       旧图邻居列表(int32), 旧图反向邻居列表(int32), 旧图前向边数量(int32), 旧图反向边数量(int32),
    #       取样图度数(uint64), 预处理数据(float16), 数据维度(int32),
    #       新生成图度数(uint64), 锁数组(int32), L2范数值(float32)
    generator.add_input("ND", [data_size, graph_degree], "int32")
    generator.add_input("ND", [data_size, graph_degree], "int32")
    generator.add_input("ND", [data_size], "int32")
    generator.add_input("ND", [data_size], "int32")
    generator.add_input("ND", [data_size, graph_degree], "int32")
    generator.add_input("ND", [data_size, graph_degree], "int32")
    generator.add_input("ND", [data_size], "int32")
    generator.add_input("ND", [data_size], "int32")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [data_size, dim], "float16")
    generator.add_input("ND", [1], "int32")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [data_size * ((graph_degree + 31) // 32)], "int32")
    generator.add_input("ND", [data_size], "float32")
    # 输出：新生成的图邻居列表(int32), 对应的距离值(float32)
    generator.add_output("ND", [data_size, graph_degree], "int32")
    generator.add_output("ND", [data_size, graph_degree], "float32")
    
    local_join_obj.append(generator.generate_obj())
    utils.generate_op_config(local_join_obj, file_path)


def generate_prune_json(data_size, graph_degree, file_path):
    prune_obj = []
    generator = OpJsonGenerator("KernPrune")
    
    # 输入：KNN图(uint64), 图节点数(uint64), 输入图度数(uint64),
    #       输出图度数(uint64), 批处理大小(uint64), 当前批次ID(uint64)
    generator.add_input("ND", [data_size, graph_degree], "uint64")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [1], "uint64")
    
    # 输出：绕行次数(uint8), 可直连的边数(uint32), 统计信息(uint64)
    generator.add_output("ND", [data_size, graph_degree], "uint8")
    generator.add_output("ND", [data_size], "uint32")
    generator.add_output("ND", [2], "uint64")
    
    prune_obj.append(generator.generate_obj())
    utils.generate_op_config(prune_obj, file_path)


def generate_make_rev_graph_json(data_size, graph_degree, file_path):
    make_rev_graph_obj = []
    generator = OpJsonGenerator("KernMakeRevGraph")
    
    # 输入：目标节点列表(uint64), 图节点数(uint64), 反向图度数(uint64)
    generator.add_input("ND", [data_size], "uint64")
    generator.add_input("ND", [1], "uint64")
    generator.add_input("ND", [1], "uint64")
    
    # 输出：反向图邻居列表(uint64), 每节点的反向边计数(uint32)
    generator.add_output("ND", [data_size, graph_degree], "uint64")
    generator.add_output("ND", [data_size], "uint32")
    
    make_rev_graph_obj.append(generator.generate_obj())
    utils.generate_op_config(make_rev_graph_obj, file_path)


def generate_all_cagra_operators_offline_model():
    utils.set_env()
    args = arg_parse()
    
    process_id = args.process_id
    dim = args.dim
    data_size = args.data_size
    graph_degree = args.graph_degree
    
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)
    
    # 生成数据预处理算子
    op_name1 = f"ascendc_preprocess_data_op_pid{process_id}"
    file_path1 = os.path.join(config_path, f"{op_name1}.json")
    generate_preprocess_data_json(dim, data_size, file_path1)
    utils.atc_model(op_name1, soc_version)
    
    # 生成添加反向边算子
    op_name2 = f"ascendc_add_reverse_edges_op_pid{process_id}"
    file_path2 = os.path.join(config_path, f"{op_name2}.json")
    generate_add_reverse_edges_json(data_size, graph_degree, file_path2)
    utils.atc_model(op_name2, soc_version)
    
    # 生成本地连接算子
    op_name3 = f"ascendc_local_join_op_pid{process_id}"
    file_path3 = os.path.join(config_path, f"{op_name3}.json")
    generate_local_join_json(dim, data_size, graph_degree, file_path3)
    utils.atc_model(op_name3, soc_version)
    
    # 生成剪枝算子
    op_name4 = f"ascendc_prune_op_pid{process_id}"
    file_path4 = os.path.join(config_path, f"{op_name4}.json")
    generate_prune_json(data_size, graph_degree, file_path4)
    utils.atc_model(op_name4, soc_version)
    
    # 生成反向图算子
    op_name5 = f"ascendc_make_rev_graph_op_pid{process_id}"
    file_path5 = os.path.join(config_path, f"{op_name5}.json")
    generate_make_rev_graph_json(data_size, graph_degree, file_path5)
    utils.atc_model(op_name5, soc_version)

if __name__ == '__main__':
    generate_all_cagra_operators_offline_model()