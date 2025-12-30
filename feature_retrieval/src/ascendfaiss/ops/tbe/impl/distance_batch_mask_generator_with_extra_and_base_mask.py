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
from mxIndex_impl.distance_batch_mask_common import DistanceBatchMaskGenerator


def distance_batch_mask_generator_with_extra_and_base_mask(input_x0, input_x1, input_x2, input_x3, input_x4, input_x5,
    input_x6, input_x7, output_y0, kernel_name="distance_batch_mask_generator_with_extra_and_base_mask"):
    """
    距离结果掩码生成算子
    Args:
        input_x0: 查询起始时间戳+结束时间戳。
                  二维Tensor,shape=(N, 8),dtype=int32,[0]表示起始时间 [1]表示结束时间,均加负号。
        input_x1: 查询token id集,特殊格式,详见方案设计文档。
                  二维Tensor,shape=(N, DIV_UP(MAX_TOKEN_CNT, 8)*2,),dtype=uint8,
                  其中MAX_TOKEN_CNT为最大token id个数:0~(MAX_TOKEN_CNT-1)。
        input_x2: 底库特征向量对应的时间戳。
                  一维Tensor,shape=(M,),dtype=int32,M为底库向量个数。
        input_x3: 底库特征向量对应的token id除以8后的除数。
                  一维Tensor,shape=(M,),dtype=int32,M为底库向量个数。
        input_x4: 底库特征向量对应的token id除以8后的余数,特殊设计,详见方案设计文档。
                  一维Tensor,shape=(2*M,),dtype=uint8,M为底库向量个数。
        input_x5: 用户额外的Mask输入，与本算子生成的Mask进行与操作。
                  一维Tensor，shape=(N, DIV_UP(M, 8),)，dtype=uint8，M为底库向量个数。
        input_x6: 用户额外的Mask输入属性信息。
                  一维Tensor，shape=(8,)，dtype=int32。
        input_x7: 底库快速删除信息，用户调用快速删除的接口生成的掩码。
                  二维Tensor，shape=(1,  DIV_UP(M, 8))，dtype=uint8, M为底库向量个数。
        output_y0: 距离结果掩码。
                   二维Tensor,shape=(N, DIV_UP(M, 8),),dtype=uint8, M为底库向量个数。

    """
    obj = DistanceBatchMaskGenerator(input_x0, input_x1, input_x2, input_x3, input_x4, input_x5, input_x6, None, None,
                                    output_y0, kernel_name)
    obj.set_basemask(input_x7)
    
    tik_instance = obj.get_tik_instance_with_base_mask()

    return tik_instance

