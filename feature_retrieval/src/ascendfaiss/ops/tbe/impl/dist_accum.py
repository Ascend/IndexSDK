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
from te import tik
from mxIndex_impl.common import set_soc_info

AICORE_USE = 2


def dist_accum(input_x, output_y, output_z, kernel_name="dist_accum"):
    """
    accumulate_op with aicore vector unit
    :param input_x: tuple, (num_features, pq_m), dytpe=float16
    :param output_y: tuple, (num_fetures, ), dtype=float16
    :param output_z: tuple, (16, ), dtype=uint16
    :param kernel_name: name of op
    :return: Tik instance
    """

    shape_x = input_x.get("shape")
    shape_y = output_y.get("shape")
    shape_z = output_z.get("shape")

    if shape_x[0] != shape_y[0]:
        raise RuntimeError("feature_num of input and output must be the same")
    if shape_z[0] != 16:
        raise RuntimeError("output flag num must be 16")

    c0 = 16
    # set repeat_time in vcadd instruction
    repeat_time = 128
    num_feature, pq_m = shape_x
    if pq_m > 128 or pq_m % c0 != 0:
        raise RuntimeError("PQ_M must be a multiple of \
                            16 and not greater than 128")

    set_soc_info()
    tik_instance = tik.Tik()
    input_x = tik_instance.Tensor("float16", shape_x,
                                  name="input_gm", scope=tik.scope_gm)
    output_y = tik_instance.Tensor("float16", shape_y,
                                   name="output_gm", scope=tik.scope_gm)
    output_z = tik_instance.Tensor("uint16", shape_z,
                                   name="output_z_gm", scope=tik.scope_gm)

    # divide data to 2 cores
    aicore_use = AICORE_USE
    num_units = num_feature // c0
    num_core_1 = (num_units // aicore_use)*c0 + num_feature % c0
    num_core_0 = num_feature - num_core_1

    # if no data allocate for core_0, use single core
    if not num_core_0:
        # keep num_core_1 since core_0 is the last core while
        # only using one core, see line_98
        num_core_0 = num_core_1
        aicore_use = 1

    # get loop times
    loop_time = num_feature // (repeat_time*aicore_use)
    t_num = 2 if loop_time >= 2 else 1

    with tik_instance.for_range(0, aicore_use,
                                block_num=aicore_use) as block_idx:
        with tik_instance.for_range(0, loop_time, thread_num=t_num) as i_loop:
            # load features from gm to ub
            input_ub = tik_instance.Tensor("float16", (repeat_time, pq_m),
                                           name="input_ub",
                                           scope=tik.scope_ubuf)
            tik_instance.tensor_mov(input_ub,
                                    input_x[block_idx*num_core_0 +
                                            repeat_time*i_loop, 0],
                                    '', 1, repeat_time*pq_m // c0, 0, 0)

            # vcadd
            res_ub = tik_instance.Tensor("float16", (repeat_time, ),
                                         name="res_ub", scope=tik.scope_ubuf)
            tik_instance.vcadd(pq_m, res_ub, input_ub, repeat_time,
                               1, 1, pq_m // c0)

            # calculate mov_block in case repeat_time is not a multiple of 16
            mov_block = repeat_time // c0 + 1 if repeat_time % c0 else repeat_time // c0
            # mov result from ub to gm
            tik_instance.tensor_mov(output_y[block_idx*num_core_0 + repeat_time*i_loop],
                                    res_ub, '', 1, mov_block, 0, 0)

        # if tail and not last core
        with tik_instance.if_scope(block_idx != aicore_use - 1):
            tail_core = num_core_0 - repeat_time*loop_time  # do not use mod op
            if tail_core and aicore_use > 1:
                # load features from gm to ub
                tail_ub = tik_instance.Tensor("float16", (tail_core, pq_m),
                                              name="tail_ub",
                                              scope=tik.scope_ubuf)
                tik_instance.tensor_mov(tail_ub,
                                        input_x[block_idx*num_core_0 +
                                                repeat_time*loop_time, 0],
                                        '', 1, tail_core*pq_m // c0, 0, 0)

                # vcadd
                tail_res = tik_instance.Tensor("float16", (tail_core, ),
                                               name="tail_res",
                                               scope=tik.scope_ubuf)
                tik_instance.vcadd(pq_m, tail_res, tail_ub, tail_core,
                                   1, 1, pq_m // c0)

                # calculate tail_block in case tail_num is not a multiple of 16
                tail_block = tail_core // c0 + 1 if tail_core % c0 else tail_core // c0

                # mov result from ub to gm
                tik_instance.tensor_mov(output_y[block_idx*num_core_0 +
                                        repeat_time*loop_time], tail_res,
                                        '', 1, tail_block, 0, 0)

        # if tail and last core
        with tik_instance.else_scope():
            tail_core_last = num_core_1 - repeat_time*loop_time
            if tail_core_last:
                # load features from gm to ub
                tail_ub = tik_instance.Tensor("float16",
                                              (tail_core_last, pq_m),
                                              name="tail_ub",
                                              scope=tik.scope_ubuf)
                tik_instance.tensor_mov(tail_ub,
                                        input_x[block_idx*num_core_0 +
                                                repeat_time*loop_time, 0],
                                        '', 1, tail_core_last*pq_m // c0, 0, 0)

                # vcadd
                tail_res = tik_instance.Tensor("float16", (tail_core_last, ),
                                               name="tail_res",
                                               scope=tik.scope_ubuf)
                tik_instance.vcadd(pq_m, tail_res, tail_ub, tail_core_last,
                                   1, 1, pq_m // c0)

                # calculate tail_block in case tail_num is not a multiple of 16
                tail_block = tail_core_last // c0 + 1 if tail_core_last % c0 else tail_core_last // c0

                # mov result from ub to gm
                tik_instance.tensor_mov(output_y[block_idx*num_core_0 +
                                        repeat_time*loop_time],
                                        tail_res, '', 1, tail_block, 0, 0)

    one = tik_instance.Scalar("uint16", "one", 1)
    flag_ub = tik_instance.Tensor("uint16", shape_z, name="flag_ub",
                                  scope=tik.scope_ubuf)
    tik_instance.tensor_mov(flag_ub, output_z, '', 1, shape_z[0] // 16, 0, 0)
    flag_ub[0].set_as(one)
    tik_instance.tensor_mov(output_z, flag_ub, '', 1, shape_z[0] // 16, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[input_x],
                          outputs=[output_y, output_z])
    return tik_instance
