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


class ComputeL2Norm:
    def __init__(self,
                 input_vectors,
                 input_transfer,
                 input_actual_num,
                 output_norm,
                 kernel_name="int8_l2_norm"):
        self.shape_vectors = input_vectors.get("shape")
        self.dtype_vectors = input_vectors.get("dtype")
        self.shape_transfer = input_transfer.get("shape")
        self.dtype_transfer = input_transfer.get("dtype")
        self.shape_actual_num = input_actual_num.get("shape")
        self.dtype_actual_num = input_actual_num.get("dtype")
        self.shape_norm = output_norm.get("shape")
        self.dtype_norm = output_norm.get("dtype")
        self.kernel_name = kernel_name

        # compute parameter
        self.vector_num, self.dim = self.shape_vectors

        # check parameter
        self.check_parameter()

        # set vector fp32 mask and fp16 mask
        self.int32_mask = 64
        self.fp16_mask = 128
        self.transfer_mask = 64
        self.vcadd_mask = 16
        self.scale = 0.01

        # set tik instance
        self.set_tik_instance()

    def check_parameter(self):
        if self.dim % 16 != 0:
            raise RuntimeError("feature dim must be a multiple of 16")

    def set_tik_instance(self):
        """
        set tik_instance
        """
        set_soc_info()
        self.tik_instance = tik.Tik()

        self.aicore_use = 2

        self.vectors_num_each_loop = max(min(128, (self.vector_num // 2 + 15) // 16 * 16), 16)

        self.set_src_dst_tensor()

    def set_src_dst_tensor(self):
        """
        set input and output tensor
        """
        # creat input tensor: input_vectors_gm, input_actual_num_gm
        # and output tensor: output_norm_gm in global buffer
        self.input_vectors_gm = self.tik_instance.Tensor(self.dtype_vectors,
                                                         self.shape_vectors,
                                                         name="input_vectors_gm", scope=tik.scope_gm)
        self.input_transfer_gm = self.tik_instance.Tensor(self.dtype_transfer,
                                                          self.shape_transfer,
                                                          name="input_transfer_gm", scope=tik.scope_gm)
        self.input_actual_num_gm = self.tik_instance.Tensor(self.dtype_actual_num,
                                                            self.shape_actual_num,
                                                            name="input_actual_num_gm",
                                                            scope=tik.scope_gm)
        self.output_norm_gm = self.tik_instance.Tensor(self.dtype_norm,
                                                       self.shape_norm,
                                                       name="output_norm_gm", scope=tik.scope_gm)

    def cal_num_each_core(self):
        """
        calculate actual code num of each core
        """
        # move actual code num from out to UB
        actual_num_ub = self.tik_instance.Tensor("uint32",
                                                 (8,),
                                                 name="actual_code_num_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(actual_num_ub,
                                    self.input_actual_num_gm,
                                    0, 1, 1, 0, 0)
        actual_num = self.tik_instance.Scalar(dtype="uint32", name="actual_code_num", init_value=0)
        actual_num.set_as(actual_num_ub[0])

        self.vectors_num_each_core = actual_num // self.aicore_use // 16 * 16
        self.vectors_num_last_core = actual_num - (self.aicore_use - 1) * self.vectors_num_each_core

    def compute_l2_norm_each_loop(self, aicore_move_offset, aicore_vectors_num, transfer_ub):
        # def compute_l2_norm_each_loop(self, aicore_move_offset, aicore_vectors_num):
        # compute xy using cube
        vectors_loop_time = aicore_vectors_num // self.vectors_num_each_loop

        with self.tik_instance.if_scope(vectors_loop_time > 0):
            with self.tik_instance.for_range(0, vectors_loop_time, thread_num=2) as loop_vectors:
                self.cube_compute_each_loop(aicore_move_offset, loop_vectors * self.vectors_num_each_loop,
                                            self.vectors_num_each_loop, transfer_ub)

        vectors_last_num = aicore_vectors_num % self.vectors_num_each_loop
        with self.tik_instance.if_scope(vectors_last_num > 0):
            self.cube_compute_each_loop(aicore_move_offset, vectors_loop_time * self.vectors_num_each_loop,
                                        vectors_last_num, transfer_ub)

    def cube_compute_each_loop(self, aicore_move_offset, vectors_move_offset, vectors_move_num, transfer_ub):
        vectors_l1 = self.tik_instance.Tensor("int8",
                                              (self.dim // 32, self.vectors_num_each_loop, 32),
                                              name="vectors_l1", scope=tik.scope_cbuf)

        with self.tik_instance.for_range(0, vectors_move_num) as i:
            # for i in range(vectors_move_num):
            self.tik_instance.data_move(vectors_l1[0, i, 0],
                                        self.input_vectors_gm[aicore_move_offset + vectors_move_offset + i, 0],
                                        0, self.dim // 32, 1, 0, self.vectors_num_each_loop - 1)

        vectors_square_l0c = \
            self.tik_instance.Tensor("int32",
                                     (self.vectors_num_each_loop // 16, self.vectors_num_each_loop, 16),
                                     name="vectors_square_l0c", scope=tik.scope_cbuf_out)
        self.tik_instance.matmul(vectors_square_l0c,
                                 vectors_l1, vectors_l1,
                                 self.vectors_num_each_loop, self.dim, self.vectors_num_each_loop)
        vectors_transfer_ub = \
            self.tik_instance.Tensor("float16",
                                     (self.vectors_num_each_loop, 16),
                                     name="vectors_square_ub", scope=tik.scope_ubuf)
        self.tik_instance.tensor_mov(vectors_transfer_ub,
                                     vectors_square_l0c,
                                     'm',
                                     self.vectors_num_each_loop // 16, 1,
                                     0, self.vectors_num_each_loop // 16, deqscale=self.scale)

        repeat_times = 16 * self.vectors_num_each_loop // self.transfer_mask
        self.tik_instance.vmul(self.transfer_mask,
                               vectors_transfer_ub,
                               vectors_transfer_ub, transfer_ub,
                               repeat_times, 1, 1, 1, 4, 4, 4)

        vectors_res_ub = self.tik_instance.Tensor("float16",
                                                  (self.vectors_num_each_loop,),
                                                  name="vectors_res_ub", scope=tik.scope_ubuf)
        self.tik_instance.vcadd(self.vcadd_mask,
                                vectors_res_ub, vectors_transfer_ub,
                                self.vectors_num_each_loop,
                                1, 1, 1)
        mask = min(self.vectors_num_each_loop, self.fp16_mask)
        repeat_times = self.vectors_num_each_loop // mask

        block_len = 16
        src_extent_size = (repeat_times - 1) * mask // 16 * block_len + mask
        wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
        wk_size = 6 * wk_size_unit
        work_ub = self.tik_instance.Tensor("float32",
                                           (wk_size,),
                                           name="work_ub", scope=tik.scope_ubuf)
        res_ub = self.tik_instance.Tensor("float16",
                                          (self.vectors_num_each_loop,),
                                          name="res_ub", scope=tik.scope_ubuf)

        self.tik_instance.vec_rsqrt_high_preci(mask,
                                               res_ub, vectors_res_ub, work_ub[0:],
                                               repeat_times, mask // 16, mask // 16)
        self.tik_instance.Tensor("int8", (1,), tik.scope_ubuf, "test3")

        self.tik_instance.data_move(self.output_norm_gm[aicore_move_offset + vectors_move_offset],
                                    res_ub,
                                    0, 1, (vectors_move_num + 15) // 16,
                                    0, 0,
                                    self.vectors_num_each_loop // 16 - 1, self.vectors_num_each_loop // 16 - 1)

    def compute_l2_norm(self):
        """
        the compute process
        """
        self.cal_num_each_core()

        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as block_index:
            transfer_ub = self.tik_instance.Tensor("float16",
                                                   (self.vectors_num_each_loop, 16),
                                                   name="transfer_ub", scope=tik.scope_ubuf)
            self.tik_instance.tensor_mov(transfer_ub,
                                         self.input_transfer_gm,
                                         'v',
                                         self.vectors_num_each_loop, 1, 0, 0)

            aicore_vectors_num = self.tik_instance.Scalar(dtype="uint32", name="aicore_vectors_num", init_value=0)
            # compute  centroids num and move offset every core
            with self.tik_instance.if_scope(block_index != self.aicore_use - 1):
                aicore_vectors_num.set_as(self.vectors_num_each_core)
            with self.tik_instance.else_scope():
                aicore_vectors_num.set_as(self.vectors_num_last_core)

            with self.tik_instance.if_scope(aicore_vectors_num > 0):
                self.compute_l2_norm_each_loop(block_index * self.vectors_num_each_core,
                                               aicore_vectors_num, transfer_ub)

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        self.compute_l2_norm()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_vectors_gm, self.input_transfer_gm, self.input_actual_num_gm],
                                   outputs=[self.output_norm_gm])

        return self.tik_instance


def int8_l2_norm(input_vectors,
                 input_transfer,
                 input_actual_num,
                 output_norm,
                 kernel_name="int8_l2_norm"):
    """
    calculating compute l2 norm

    Parameters
    ----------
    input_vectors : dict
        shape and dtype of vector
    input_transfer : dict
        shape and dtype of vector
    input_actual_num: dict
        shape and dtype of actual code num,
        shape must be (8,) and dtype must be uint32
    output_norm : dict
        shape and dtype of l2 norm
    kernel_name : str
        kernel name, default value is "l2_norm"

    Returns
    -------
    None
    """
    l2_norm = ComputeL2Norm(input_vectors, input_transfer, input_actual_num,
                            output_norm, kernel_name)
    tik_instance = l2_norm.get_tik_instance()
    return tik_instance
