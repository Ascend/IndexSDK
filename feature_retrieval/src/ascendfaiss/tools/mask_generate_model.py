#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
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
import math
from multiprocessing import Pool

import common as utils
from common import OpJsonGenerator

_BLOCK_SIZE = 262144
_MAX_TOKEN_VALUE = 300000


def arg_parse():
    """
    Parse arguements to the operator model
    """
    parser = argparse.ArgumentParser(
        description='generate aicore ts masks operator model')

    utils.op_common_parse(parser, "-token", 'max_token_cnt', 2500, int, "Max token num, default 2500")
    utils.op_common_parse(parser, "-p", 'process_id', 0, int, "Number of process_id, default 0")
    utils.op_common_parse(parser, "-pool", 'pool_size', 16, int, "Number of pool_size, default 16")
    utils.op_common_parse(parser, "-t", 'npu_type', "310P", str, "NPU type, only support 310P. 310P by default")
    return parser.parse_args()


def generate_ts_masks_json(max_token_num, file_path):
    # write ts masks op json
    ts_masks_json_obj = []
    for block_count in (1, 60, 64):
        generator = OpJsonGenerator("DistanceMaskGenerator")
        generator.add_input("ND", [8], "int32")
        generator.add_input("ND", [(max_token_num + 7) // 8 * 2], "uint8")
        generator.add_input("ND", [_BLOCK_SIZE * block_count], "int32")
        generator.add_input("ND", [_BLOCK_SIZE * block_count], "int32")
        generator.add_input("ND", [_BLOCK_SIZE * block_count * 2], "uint8")
        generator.add_output("ND", [(_BLOCK_SIZE * block_count + 7) // 8], "uint8")
        obj = generator.generate_obj()
        ts_masks_json_obj.append(obj)
    utils.generate_op_config(ts_masks_json_obj, file_path)


def generate_ts_extra_masks_json(max_token_num, file_path):
    # write ts extra masks op json
    ts_extra_masks_json_obj = []
    for block_count in (1, 60, 64):
        generator = OpJsonGenerator("DistanceMaskGeneratorWithExtra")
        generator.add_input("ND", [8], "int32")
        generator.add_input("ND", [(max_token_num + 7) // 8 * 2], "uint8")
        generator.add_input("ND", [_BLOCK_SIZE * block_count], "int32")
        generator.add_input("ND", [_BLOCK_SIZE * block_count], "int32")
        generator.add_input("ND", [_BLOCK_SIZE * block_count * 2], "uint8")
        generator.add_input("ND", [(_BLOCK_SIZE * block_count + 7) // 8], "uint8")
        generator.add_output("ND", [(_BLOCK_SIZE * block_count + 7) // 8], "uint8")
        obj = generator.generate_obj()
        ts_extra_masks_json_obj.append(obj)
    utils.generate_op_config(ts_extra_masks_json_obj, file_path)


def generate_ts_batch_masks_json(max_token_num, batch, file_path):
    # write ts masks op json
    ts_batch_masks_json_obj = []
    generator = OpJsonGenerator("DistanceBatchMaskGenerator")
    generator.add_input("ND", [batch, 8], "int32")
    generator.add_input("ND", [batch, (max_token_num + 7) // 8 * 2], "uint8")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE * 2], "uint8")
    generator.add_output("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    ts_batch_masks_json_obj.append(generator.generate_obj())
    utils.generate_op_config(ts_batch_masks_json_obj, file_path)


def generate_ts_batch_val_masks_json(max_token_num, batch, file_path):
    # write ts val masks op json
    ts_batch_val_masks_json_obj = []
    generator = OpJsonGenerator("DistanceBatchValMaskGenerator")
    generator.add_input("ND", [batch, 8], "int32")
    generator.add_input("ND", [batch, (max_token_num + 7) // 8 * 2], "uint8")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE * 2], "uint8")
    generator.add_input("ND", [batch, 16], "int16")
    generator.add_input("ND", [_BLOCK_SIZE], "int16")
    generator.add_output("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    ts_batch_val_masks_json_obj.append(generator.generate_obj())
    utils.generate_op_config(ts_batch_val_masks_json_obj, file_path)


def generate_ts_batch_extra_masks_json(max_token_num, batch, file_path, use_base_mask: bool):
    ts_extra_batch_masks_json_obj = []
    generator = OpJsonGenerator("DistanceBatchMaskGeneratorWithExtraAndBaseMask") if use_base_mask \
        else OpJsonGenerator("DistanceBatchMaskGeneratorWithExtra")
    generator.add_input("ND", [batch, 8], "int32")
    generator.add_input("ND", [batch, (max_token_num + 7) // 8 * 2], "uint8")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE * 2], "uint8")
    generator.add_input("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    generator.add_input("ND", [8], "int32")
    if use_base_mask:
        generator.add_input("ND", [1, (_BLOCK_SIZE + 7) // 8], "uint8")
    generator.add_output("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    ts_extra_batch_masks_json_obj.append(generator.generate_obj())
    utils.generate_op_config(ts_extra_batch_masks_json_obj, file_path)


def generate_ascendc_batch_masks_json(max_token_num, batch, file_path):
    # write ts masks op json
    ascendc_batch_masks_json_obj = []
    generator = OpJsonGenerator("AscendcDistanceBatchMaskGenerator")
    generator.add_input("ND", [batch, 8], "int32")
    generator.add_input("ND", [batch, (max_token_num + 7) // 8 * 2], "uint8")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE * 2], "uint8")
    generator.add_output("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    ascendc_batch_masks_json_obj.append(generator.generate_obj())
    utils.generate_op_config(ascendc_batch_masks_json_obj, file_path)


def generate_ascendc_batch_extra_masks_json(max_token_num, batch, file_path):
    ascendc_extra_batch_masks_json_obj = []
    generator = OpJsonGenerator("AscendcDistanceBatchMaskGeneratorWithExtra")
    generator.add_input("ND", [batch, 8], "int32")
    generator.add_input("ND", [batch, (max_token_num + 7) // 8 * 2], "uint8")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE], "int32")
    generator.add_input("ND", [_BLOCK_SIZE * 2], "uint8")
    generator.add_input("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    generator.add_input("ND", [8], "int32")
    generator.add_output("ND", [batch, (_BLOCK_SIZE + 7) // 8], "uint8")
    ascendc_extra_batch_masks_json_obj.append(generator.generate_obj())
    utils.generate_op_config(ascendc_extra_batch_masks_json_obj, file_path)


def generate_ts_masks_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    max_token_cnt = args.max_token_cnt
    if args.npu_type != "310P":
        raise RuntimeError("NPU type only support 310P now!")
    if args.max_token_cnt > _MAX_TOKEN_VALUE or args.max_token_cnt < 1:
        raise ValueError("input variable token {0}, should be in range[1, {1}]".format(max_token_cnt, _MAX_TOKEN_VALUE))
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    batch_sizes = (256, 128, 64, 48, 36, 32, 30, 24, 20, 18, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1)

    ts_batch_masks_generate_op_name = "ts_batch_masks_generate_op{}_batch{}_pid{}"
    ts_batch_extra_masks_generate_op_name = "ts_batch_extra_masks_generate_op{}_batch{}_pid{}"
    ts_batch_extra_and_base_masks_generate_op_name = "ts_batch_extra_and_base_masks_generate_op{}_batch{}_pid{}"
    ts_batch_val_masks_generate_op_name = "ts_batch_val_masks_generate_op{}_batch{}_pid{}"

    ts_masks_generate_op_name = "ts_masks_generate_op{}_pid{}"
    ts_extra_masks_generate_op_name = "ts_extra_masks_generate_op{}_pid{}"

    map_args = []
    ts_masks_generate_op_name_ = ts_masks_generate_op_name.format(max_token_cnt, process_id)
    file_path = os.path.join(config_path, '{}.json'.format(ts_masks_generate_op_name_))
    generate_ts_masks_json(max_token_cnt, file_path)

    ts_extra_masks_generate_op_name_ = ts_extra_masks_generate_op_name.format(max_token_cnt, process_id)
    extra_file_path = os.path.join(config_path, '{}.json'.format(ts_extra_masks_generate_op_name_))
    generate_ts_extra_masks_json(max_token_cnt, extra_file_path)

    map_args.append((ts_masks_generate_op_name_, soc_version))
    map_args.append((ts_extra_masks_generate_op_name_, soc_version))

    for batch in batch_sizes:
        ts_batch_masks_generate_op_name_ = ts_batch_masks_generate_op_name.format(max_token_cnt, batch, process_id)
        file_path = os.path.join(config_path, '{}.json'.format(ts_batch_masks_generate_op_name_))
        generate_ts_batch_masks_json(max_token_cnt, batch, file_path)

        ts_batch_val_masks_generate_op_name_ = ts_batch_val_masks_generate_op_name.format(max_token_cnt,
                                                                                          batch, process_id)
        file_path = os.path.join(config_path, '{}.json'.format(ts_batch_val_masks_generate_op_name_))
        generate_ts_batch_val_masks_json(max_token_cnt, batch, file_path)

        ts_batch_extra_masks_generate_op_name_ = ts_batch_extra_masks_generate_op_name.format(max_token_cnt,
                                                                                              batch, process_id)
        extra_file_path = os.path.join(config_path, '{}.json'.format(ts_batch_extra_masks_generate_op_name_))
        generate_ts_batch_extra_masks_json(max_token_cnt, batch, extra_file_path, False)

        ts_batch_extra_and_base_masks_generate_op_name_ = \
            ts_batch_extra_and_base_masks_generate_op_name.format(max_token_cnt, batch, process_id)
        basemask_file = os.path.join(config_path, '{}.json'.format(ts_batch_extra_and_base_masks_generate_op_name_))
        generate_ts_batch_extra_masks_json(max_token_cnt, batch, basemask_file, True)

        map_args.append((ts_batch_masks_generate_op_name_, soc_version))
        map_args.append((ts_batch_extra_masks_generate_op_name_, soc_version))
        map_args.append((ts_batch_extra_and_base_masks_generate_op_name_, soc_version))
        map_args.append((ts_batch_val_masks_generate_op_name_, soc_version))

    utils.run_generate_model_task(args, map_args)  


def generate_ascendc_masks_offline_model():
    utils.set_env()
    args = arg_parse()
    process_id = args.process_id
    max_token_cnt = args.max_token_cnt
    if args.max_token_cnt > _MAX_TOKEN_VALUE or args.max_token_cnt < 1:
        raise ValueError("input variable token {0}, should be in range[1, {1}]".format(max_token_cnt, _MAX_TOKEN_VALUE))
    soc_version = utils.get_soc_version_from_npu_type(args.npu_type)
    work_dir = '.'
    config_path = utils.get_config_path(work_dir)

    batch_sizes = (256, 128, 64, 48, 36, 32, 30, 24, 20, 18, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1)

    ascendc_batch_masks_generate_op_name = "ascendc_batch_masks_generate_op{}_batch{}_pid{}"
    ascendc_batch_extra_masks_generate_op_name = "ascendc_batch_extra_masks_generate_op{}_batch{}_pid{}"

    map_args = []
    for batch in batch_sizes:
        ascendc_batch_masks_generate_op_name_ = ascendc_batch_masks_generate_op_name.format(max_token_cnt, batch, 
                                                                                            process_id)
        file_path = os.path.join(config_path, '{}.json'.format(ascendc_batch_masks_generate_op_name_))
        generate_ascendc_batch_masks_json(max_token_cnt, batch, file_path)

        ascendc_batch_extra_masks_generate_op_name_ = ascendc_batch_extra_masks_generate_op_name.format(max_token_cnt,
                                                                                              batch, process_id)
        extra_file_path = os.path.join(config_path, '{}.json'.format(ascendc_batch_extra_masks_generate_op_name_))
        generate_ascendc_batch_extra_masks_json(max_token_cnt, batch, extra_file_path, False)

        map_args.append((ascendc_batch_masks_generate_op_name_, soc_version))
        map_args.append((ascendc_batch_extra_masks_generate_op_name_, soc_version))

    utils.run_generate_model_task(args, map_args)  

if __name__ == '__main__':
    generate_ts_masks_offline_model()
    generate_ascendc_masks_offline_model()
