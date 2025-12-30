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
import stat
import time
import logging
import struct
import warnings
from glob import glob
import acl
from tqdm import tqdm
import numpy as np
import faiss

LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

ACL_SUCCESS = 0

MAX_SIZE = 3 * 1024 * 1024 * 1024 # 3GB is the max size for a CodeBook

# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# images format
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']

# numpy data type
NPY_BOOL = 0
NPY_INT8 = 1
NPY_UINT8 = 2
NPY_INT16 = 3
NPY_UINT16 = 4
NPY_INT32 = 5
NPY_UINT32 = 6
NPY_INT64 = 7
NPY_UINT64 = 8
NPY_LONGLONG = 9
NPY_ULONGLONG = 10
NPY_FLOAT32 = 11
NPY_DOUBLE = 12
NPY_FLOAT16 = 23
NPY_TYPE2NPY_TYPE = {
    NPY_BOOL: bool,
    NPY_INT8: np.int8,
    NPY_UINT8: np.uint8,
    NPY_INT16: np.int16,
    NPY_UINT16: np.uint16,
    NPY_INT32: np.int32,
    NPY_UINT32: np.uint32,
    NPY_INT64: np.int64,
    NPY_UINT64: np.uint64,
    NPY_FLOAT32: np.float32,
    NPY_DOUBLE: np.float64,
    NPY_FLOAT16: np.float16
}


ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT8 = 4
ACL_INT16 = 6
ACL_UINT16 = 7
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_DOUBLE = 11
ACL_BOOL = 12


ACL_TYPE2NPY_TYPE = {
    ACL_FLOAT: NPY_FLOAT32,
    ACL_FLOAT16: NPY_FLOAT16,
    ACL_INT8: NPY_INT8,
    ACL_INT32: NPY_INT32,
    ACL_UINT8: NPY_UINT8,
    ACL_INT16: NPY_INT16,
    ACL_UINT16: NPY_UINT16,
    ACL_UINT32: NPY_UINT32,
    ACL_INT64: NPY_INT64,
    ACL_UINT64: NPY_UINT64,
    ACL_DOUBLE: NPY_DOUBLE,
    ACL_BOOL: NPY_BOOL
}

KMEANS_VERBOSE = False
KMEANS_MAX_ITER = 16

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
}


def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class Net(object):
    def __init__(self, device_id, model_path, is_debug=False):
        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.is_debug = is_debug
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = []
        self.output_data = []
        self.model_desc = None          # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.init_resource()


    def release_resource(self):
        if self.is_debug:
            logging.info("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        if self.is_debug:
            logging.info('Resources released successfully.')

    def init_resource(self):
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()

    def get_result(self, output_data):
        result = []

        for i, temp in enumerate(output_data):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            acl_data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            npy_data_type = ACL_TYPE2NPY_TYPE[acl_data_type]
            check_ret("acl.mdl.get_cur_output_dims", ret)
            out_dim = dims['dims']
            if self.is_debug:
                logging.info("out_dims: %s", dims)
            ptr = temp["buffer"]
            # 转化为float32类型的数据
            if "ptr_to_bytes" in dir(acl.util):
                bytes_data = acl.util.ptr_to_bytes(ptr, temp["size"])
                data = np.frombuffer(bytes_data, dtype=NPY_TYPE2NPY_TYPE[npy_data_type]).reshape(tuple(out_dim))
            else:
                data = acl.util.ptr_to_numpy(ptr, tuple(out_dim), npy_data_type)
            result.append(data)

        return result

    def run(self, batch_inputs):
        self._data_from_host_to_device(batch_inputs)
        self.forward()
        res = self._data_from_device_to_host()
        return res

    def forward(self):
        if self.is_debug:
            logging.info('execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        if self.is_debug:
            logging.info('execute stage success')

    def _get_model_info(self,):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method.get(des)
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if "bytes_to_ptr" in dir(acl.util):
                    bytes_data = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_data)
                else:
                    ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        if self.is_debug:
            logging.info("data interaction from host to device")
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        if self.is_debug:
            logging.info("data interaction from host to device success")

    def _data_from_device_to_host(self):
        if self.is_debug:
            logging.info("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        if self.is_debug:
            logging.info("data interaction from device to host success")
        result = self.get_result(res)
        return_res = [x.copy() for x in result]
        # free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        return return_res

    def _print_result(self, result):
        vals = np.array(result).flatten()
        logging.info("%s", vals)

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)


class MultiNet(object):
    def __init__(self, device_id, model_paths, is_debug=False):
        self.device_id = device_id      # int
        self.model_paths = model_paths    # string
        self.is_debug = is_debug
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = {}
        self.output_data = {}
        self.model_desc = None          # pointer when using
        self.load_input_dataset = {}
        self.load_output_dataset = {}

        self.init_resource()

    def release_resource(self):
        if self.is_debug:
            logging.info("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        if self.is_debug:
            logging.info('Resources released successfully.')

    def init_resource(self):
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_ids = []
        self.model_descs = []
        for model_idx, model_path in enumerate(self.model_paths):
            model_id, ret = acl.mdl.load_from_file(model_path)
            self.model_ids.append(model_id)
            check_ret("acl.mdl.load_from_file", ret)
            self.model_descs.append(acl.mdl.create_desc())
            self._get_model_info(model_idx)

    def get_result(self, model_idx, output_data):
        result = []

        for i, temp in enumerate(output_data):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_descs[model_idx], i)
            acl_data_type = acl.mdl.get_output_data_type(self.model_descs[model_idx], i)
            npy_data_type = ACL_TYPE2NPY_TYPE[acl_data_type]
            check_ret("acl.mdl.get_cur_output_dims", ret)
            out_dim = dims['dims']
            if self.is_debug:
                logging.info("out_dims: %s", dims)
            ptr = temp["buffer"]
            if "ptr_to_bytes" in dir(acl.util):
                bytes_data = acl.util.ptr_to_bytes(ptr, temp["size"])
                data = np.frombuffer(bytes_data, dtype=NPY_TYPE2NPY_TYPE[npy_data_type]).reshape(tuple(out_dim))
            else:
                data = acl.util.ptr_to_numpy(ptr, tuple(out_dim), npy_data_type)
            result.append(data)

        return result

    def run(self, model_idx, batch_inputs):
        self._data_from_host_to_device(model_idx, batch_inputs)
        self.forward(model_idx)
        res = self._data_from_device_to_host(model_idx)
        return res

    def forward(self, model_idx):
        if self.is_debug:
            logging.info('execute stage:')
        ret = acl.mdl.execute(self.model_ids[model_idx],
                              self.load_input_dataset.get(model_idx),
                              self.load_output_dataset.get(model_idx))
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer(model_idx)
        if self.is_debug:
            logging.info('execute stage success')

    def _get_model_info(self, model_idx):
        ret = acl.mdl.get_desc(self.model_descs[model_idx], self.model_ids[model_idx])
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_descs[model_idx])
        output_size = acl.mdl.get_num_outputs(self.model_descs[model_idx])
        self._gen_data_buffer(model_idx, input_size, des="in")
        self._gen_data_buffer(model_idx, output_size, des="out")

    def _gen_data_buffer(self, model_idx, size, des):
        func = buffer_method.get(des)
        if model_idx not in self.input_data:
            self.input_data[model_idx] = []
        if model_idx not in self.output_data:
            self.output_data[model_idx] = []
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_descs[model_idx], i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data[model_idx].append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data[model_idx].append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, model_idx, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data[model_idx] \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data[model_idx]
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data[model_idx]:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if "bytes_to_ptr" in dir(acl.util):
                    if isinstance(dataset[i], bytes):
                        bytes_data = dataset[i]
                    else:
                        bytes_data = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_data)
                else:
                    ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, model_idx, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset[model_idx] = dataset
            temp_dataset = self.input_data[model_idx]
        else:
            self.load_output_dataset[model_idx] = dataset
            temp_dataset = self.output_data[model_idx]

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, model_idx, images):
        if self.is_debug:
            logging.info("data interaction from host to device")
        # copy images to device
        self._data_interaction(model_idx, images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset(model_idx, "in")
        # load output data into model
        self._gen_dataset(model_idx, "out")
        if self.is_debug:
            logging.info("data interaction from host to device success")

    def _data_from_device_to_host(self, model_idx):
        if self.is_debug:
            logging.info("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(model_idx, res, ACL_MEMCPY_DEVICE_TO_HOST)
        if self.is_debug:
            logging.info("data interaction from device to host success")
        result = self.get_result(model_idx, res)
        return_res = [x.copy() for x in result]
        # free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        return return_res

    def _print_result(self, result):
        vals = np.array(result).flatten()
        logging.info("%s", vals)

    def _destroy_databuffer(self, model_idx):
        for dataset in [self.load_input_dataset[model_idx], self.load_output_dataset[model_idx]]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)


class SPACL(object):
    def __init__(self, k, dim, dim2, qcl2_batch_size, qc_batch_size, cb_batch_size,
                 ata_batch_size, device_id, model_dir, is_debug=False):
        self.device_id = device_id      # int
        self.model_dir = model_dir    # string
        self.is_debug = is_debug
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = {}
        self.output_data = {}
        self.model_desc = None          # pointer when using
        self.load_input_dataset = {}
        self.load_output_dataset = {}
        self.k = k
        self.dim = dim
        self.dim2 = dim2
        self.qcl2_batch_size = qcl2_batch_size
        self.qc_batch_size = qc_batch_size
        self.cb_batch_size = cb_batch_size
        self.ata_batch_size = ata_batch_size
        self.op_names = ["DistanceComputeQC", "MatmulCompute", "MatmulCompute", "CorrCompute"]
        self.modelidx2id = {}
        self.init_resource()

    def release_resource(self):
        if self.is_debug:
            logging.info("Releasing resources stage:")
        for model_id in self.model_ids:
            ret = acl.mdl.unload(model_id)
            check_ret("acl.mdl.unload", ret)
        for model_desc in self.model_descs:
            acl.mdl.destroy_desc(model_desc)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        if self.is_debug:
            logging.info('Resources released successfully.')

    def init_resource(self):
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_ids = []
        self.model_descs = []
        model2checkfunc = {
            0: self.check_qcl2_op,
            1: self.check_qc_op,
            2: self.check_cb_op,
            3: self.check_ata_op
        }
        for model_idx, op_name in enumerate(self.op_names):
            modelpaths = glob(f"{self.model_dir}/*{op_name}*.om")
            check_res = False
            for modelpath in modelpaths:
                model_id, ret = acl.mdl.load_from_file(modelpath)

                model_desc = acl.mdl.create_desc()
                ret = acl.mdl.get_desc(model_desc, model_id)
                check_res = model2checkfunc.get(model_idx)(model_desc)
                if check_res is False:
                    ret = acl.mdl.unload(model_id)
                    check_ret("acl.mdl.unload", ret)
                    if model_desc:
                        acl.mdl.destroy_desc(model_desc)
                else:
                    break
            if check_res is False:
                raise Exception(f"op {op_name} can not match!")
            self.modelidx2id[model_idx] = model_id
            self.model_ids.append(model_id)
            check_ret("acl.mdl.load_from_file", ret)
            self.model_descs.append(acl.mdl.create_desc())
            self._get_model_info(model_idx)

    def check_qcl2_op(self, model_desc):
        input_size = acl.mdl.get_num_inputs(model_desc)
        output_size = acl.mdl.get_num_outputs(model_desc)
        for i in range(input_size):
            dims_msg = acl.mdl.get_input_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.qcl2_batch_size or dims_msg["dims"][1] != self.dim:
                        return False
                else:
                    return False
            elif i == 1:
                if dims_msg["dimCount"] == 4:
                    if dims_msg["dims"][0] != int(self.k * self.dim2 // 16) or \
                            dims_msg["dims"][1] != int(self.dim // 16) or \
                            dims_msg["dims"][2] != 16 or  \
                            dims_msg["dims"][3] != 16:
                        return False
                else:
                    return False
        for i in range(output_size):
            dims_msg = acl.mdl.get_output_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.qcl2_batch_size or dims_msg["dims"][1] != self.k:
                        return False
                else:
                    return False

        return True

    def check_qc_op(self, model_desc):
        input_size = acl.mdl.get_num_inputs(model_desc)
        output_size = acl.mdl.get_num_outputs(model_desc)
        for i in range(input_size):
            dims_msg = acl.mdl.get_input_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.dim2 or dims_msg["dims"][1] != self.dim:
                        return False
                else:
                    return False
            elif i == 1:
                if dims_msg["dimCount"] == 4:
                    if dims_msg["dims"][0] != int(self.qc_batch_size // 16) or dims_msg["dims"][1] != int(
                            self.dim // 16) or dims_msg["dims"][2] != 16 or dims_msg["dims"][3] != 16:
                        return False
                else:
                    return False

        for i in range(output_size):
            dims_msg = acl.mdl.get_output_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.dim2 or dims_msg["dims"][1] != self.qc_batch_size:
                        return False
                else:
                    return False

        return True


    def check_cb_op(self, model_desc):
        input_size = acl.mdl.get_num_inputs(model_desc)
        output_size = acl.mdl.get_num_outputs(model_desc)
        for i in range(input_size):
            dims_msg = acl.mdl.get_input_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.dim or dims_msg["dims"][1] != self.dim2:
                        return False
                else:
                    return False
            elif i == 1:
                if dims_msg["dimCount"] == 4:
                    if dims_msg["dims"][0] != int(self.cb_batch_size // 16) or dims_msg["dims"][1] != int(
                            self.dim2 // 16) or dims_msg["dims"][2] != 16 or dims_msg["dims"][3] != 16:
                        return False
                else:
                    return False

        for i in range(output_size):
            dims_msg = acl.mdl.get_output_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.dim or dims_msg["dims"][1] != self.cb_batch_size:
                        return False
                else:
                    return False

        return True

    def check_ata_op(self, model_desc):
        input_size = acl.mdl.get_num_inputs(model_desc)
        output_size = acl.mdl.get_num_outputs(model_desc)
        for i in range(input_size):
            dims_msg = acl.mdl.get_input_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 4:
                    if dims_msg["dims"][1] != int(self.ata_batch_size // 16) or \
                            dims_msg["dims"][2] != self.dim or dims_msg["dims"][3] != 16:
                        return False
                else:
                    return False

        for i in range(output_size):
            dims_msg = acl.mdl.get_output_dims(model_desc, i)[0]
            if i == 0:
                if dims_msg["dimCount"] == 2:
                    if dims_msg["dims"][0] != self.dim or dims_msg["dims"][1] != self.dim:
                        return False
                else:
                    return False

        return True

    def get_result(self, model_idx, output_data):
        result = []

        for i, temp in enumerate(output_data):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_descs[model_idx], i)
            acl_data_type = acl.mdl.get_output_data_type(self.model_descs[model_idx], i)
            npy_data_type = ACL_TYPE2NPY_TYPE[acl_data_type]
            check_ret("acl.mdl.get_cur_output_dims", ret)
            out_dim = dims['dims']
            if self.is_debug:
                logging.info("out_dims: %s", dims)
            ptr = temp["buffer"]

            # 转化为float32类型的数据
            if "ptr_to_bytes" in dir(acl.util):
                bytes_data = acl.util.ptr_to_bytes(ptr, temp["size"])
                data = np.frombuffer(bytes_data, dtype=NPY_TYPE2NPY_TYPE[npy_data_type]).reshape(tuple(out_dim))
            else:
                data = acl.util.ptr_to_numpy(ptr, tuple(out_dim), npy_data_type)
            result.append(data)

        return result

    def run(self, model_idx, batch_inputs):
        self._data_from_host_to_device(model_idx, batch_inputs)
        self.forward(model_idx)
        res = self._data_from_device_to_host(model_idx)
        return res

    def forward(self, model_idx):
        if self.is_debug:
            logging.info('execute stage:')
        ret = acl.mdl.execute(self.model_ids[model_idx],
                              self.load_input_dataset.get(model_idx),
                              self.load_output_dataset.get(model_idx))
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer(model_idx)
        if self.is_debug:
            logging.info('execute stage success')

    def _get_model_info(self, model_idx):
        ret = acl.mdl.get_desc(self.model_descs[model_idx], self.model_ids[model_idx])
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_descs[model_idx])
        output_size = acl.mdl.get_num_outputs(self.model_descs[model_idx])
        self._gen_data_buffer(model_idx, input_size, des="in")
        self._gen_data_buffer(model_idx, output_size, des="out")

    def _gen_data_buffer(self, model_idx, size, des):
        func = buffer_method.get(des)
        if model_idx not in self.input_data:
            self.input_data[model_idx] = []
        if model_idx not in self.output_data:
            self.output_data[model_idx] = []
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_descs[model_idx], i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data[model_idx].append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data[model_idx].append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, model_idx, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data[model_idx] \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data[model_idx]
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data[model_idx]:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if "bytes_to_ptr" in dir(acl.util):
                    if isinstance(dataset[i], bytes):
                        bytes_data = dataset[i]
                    else:
                        bytes_data = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_data)
                else:
                    ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, model_idx, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset[model_idx] = dataset
            temp_dataset = self.input_data[model_idx]
        else:
            self.load_output_dataset[model_idx] = dataset
            temp_dataset = self.output_data[model_idx]

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, model_idx, images):
        if self.is_debug:
            logging.info("data interaction from host to device")
        # copy images to device
        self._data_interaction(model_idx, images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset(model_idx, "in")
        # load output data into model
        self._gen_dataset(model_idx, "out")
        if self.is_debug:
            logging.info("data interaction from host to device success")

    def _data_from_device_to_host(self, model_idx):
        if self.is_debug:
            logging.info("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(model_idx, res, ACL_MEMCPY_DEVICE_TO_HOST)
        if self.is_debug:
            logging.info("data interaction from device to host success")
        result = self.get_result(model_idx, res)
        return_res = [x.copy() for x in result]
        # free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        return return_res

    def _print_result(self, result):
        vals = np.array(result).flatten()
        logging.info("%s", vals)

    def _destroy_databuffer(self, model_idx):
        for dataset in [self.load_input_dataset[model_idx], self.load_output_dataset[model_idx]]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)


def save_bin(data, file):
    data.tofile(file)


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def mmap_vecs(fname, dimension, dtype):
    x = np.memmap(fname, dtype=dtype, mode='r')
    return x.reshape(-1, dimension)


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_file(file_name, dtype=None, shape=None):
    max_val_size = 100 * 1024 * 1024 * 1024  # 100G
    load_data_check(file_name, max_val_size)
    base_name = os.path.basename(file_name)
    if 'fvec' in base_name:
        data = mmap_fvecs(file_name)
    elif 'ivec' in base_name:
        data = ivecs_read(file_name)
    elif 'bvec' in base_name:
        data = mmap_bvecs(file_name)
    elif 'memmap' in base_name:
        data = np.memmap(file_name, dtype=np.float32, mode="r")
    elif 'npy' in base_name:
        data = np.load(file_name)
    elif 'bin' in base_name:
        data = np.fromfile(file_name, dtype=dtype).reshape(shape)

    return data


def recall(query_results, gt, recall_cutoffs, gt_cut=1):
    res = [0.0] * len(recall_cutoffs)
    for i, _ in enumerate(query_results):
        for j, k in enumerate(recall_cutoffs):
            res[j] += len(set.intersection(set(query_results[i][:k]), set(gt[i][:gt_cut]))) / len(set(gt[i][:gt_cut]))

    return [x / len(gt) for x in res]


class SubSpaceClusterACL:
    def __init__(self, dim, k, block_size, model_dir, batch_size=1024, update_codebook_batch_size=10240,
                 seed=10, eps=1e-10, verbose=False, codebook=None, device=0, logger=None):
        self.dim = dim
        self.k = k
        self.block_size = block_size
        self.K = self.k * self.block_size
        self.verbose = verbose
        self.device = device

        self.seed = seed
        self.eps = eps
        self.codebook = None
        self.codebook_reshape_bytes = None
        self.codebook_reshape = None

        self.logger = logger
        self.blank = np.zeros(64, ).astype(np.uint8).reshape(1, -1)
        self.version_major = 1
        self.version_medium = 0
        self.version_minor = 0

        if isinstance(codebook, np.ndarray):
            self.codebook = codebook
        else:
            self.codebook = np.random.randn(self.dim, self.K)

        self.batch_size = batch_size
        self.update_codebook_batch_size = update_codebook_batch_size
        self.models = SPACL(k=self.k,
                            dim=self.dim,
                            dim2=self.block_size,
                            qcl2_batch_size=batch_size,
                            qc_batch_size=update_codebook_batch_size,
                            cb_batch_size=update_codebook_batch_size,
                            ata_batch_size=update_codebook_batch_size,
                            model_dir=model_dir,
                            device_id=device
                            )


    def reshape_codebook(self):
        self.codebook_reshape = \
            self.codebook.T.reshape(self.K // 16, 16, self.dim // 16, 16).transpose(0, 2, 1, 3).astype(np.float16)
        self.codebook_reshape_bytes = self.codebook_reshape.tobytes()

    def codebook_kmeans_init(self, base):
        """
        Initialize the codebook with KMeans algorithm
        """
        kmeans = faiss.Kmeans(d=self.dim, k=self.k, verbose=KMEANS_VERBOSE, niter=KMEANS_MAX_ITER)
        kmeans.train(base)
        kmeans_labels = kmeans.index.search(base, 1)[1].squeeze()
        labels_unique = list(set(kmeans_labels))
        batch_base = np.zeros((self.update_codebook_batch_size, self.dim), np.float16)
        for i in tqdm(range(len(labels_unique)), desc='Codebook Initiation'):
            msk = (kmeans_labels == labels_unique[i])
            sub_base = base[msk, :]
            batch_base *= 0.
            batch_base[:min(self.update_codebook_batch_size, len(sub_base)), :] =\
                sub_base[:min(self.update_codebook_batch_size, len(sub_base)), :]
            self.update_codebook_acl(sub_base=batch_base, k_i=i)

    # 保留
    def train(self, base, num_iter=10, is_cal_loss=True, codebook_dir=None):
        batch_size = self.batch_size
        update_codebook_batch_size = self.update_codebook_batch_size
        if codebook_dir is not None and os.path.exists(f'{codebook_dir}/codebook.npy'):
            self.load_codebook(codebook_dir)
            logging.info("Load codebook from %s/codebook.npy", codebook_dir)
        base_num = base.shape[0]
        iter_ = tqdm(range(num_iter), desc='iteration:') if self.verbose else range(num_iter)
        nonzero_idx = np.zeros(base_num)
        num_batch = (base_num + batch_size - 1) // batch_size
        self.orth_codebook()
        self.codebook_kmeans_init(base)

        fix_batch_x = np.zeros((batch_size, self.dim))
        for _ in iter_:
            self.reshape_codebook()
            start_idx = 0
            loss = 0.
            st = time.time()
            train_loader = tqdm(range(num_batch), desc='train_batch')

            for batch_i in train_loader:

                batch_x = base[batch_i * batch_size:(batch_i + 1) * batch_size]

                batch_sample_num = batch_x.shape[0]
                fix_batch_x[:batch_sample_num] = batch_x
                sub_nonzero_idx = self.cal_nonzero_idx(sub_base=fix_batch_x)
                nonzero_idx[start_idx:start_idx + batch_sample_num] = sub_nonzero_idx[:batch_sample_num]

                start_idx += batch_sample_num

            choose_block_cal_time = time.time() - st
            start_time = time.time()

            batch_base = np.zeros((update_codebook_batch_size, self.dim), np.float16)
            actual_nums = np.zeros((8, 8)).astype(np.uint32)

            for id_ in tqdm(range(self.k), desc='update_codebook:'):
                msk = np.where(nonzero_idx == id_)[0]
                sub_base = base[msk, :]
                batch_base[:, :] = 0.

                batch_base[:min(update_codebook_batch_size, len(msk)), :] = \
                    sub_base[:min(update_codebook_batch_size, len(msk)), :]
                self.update_codebook_acl(sub_base=batch_base, k_i=id_)
                if is_cal_loss: # and it % 5 == 4:
                    batch_block_num = (len(msk) + update_codebook_batch_size - 1) // update_codebook_batch_size
                    for i in range(batch_block_num):
                        batch_base[:, :] = 0.

                        actual_nums[0][0] = update_codebook_batch_size
                        batch_base[:min(update_codebook_batch_size, len(msk) - i * update_codebook_batch_size), :] = \
                            sub_base[i * update_codebook_batch_size:(i + 1) * update_codebook_batch_size, :]

                        batch_base_reshape = \
                            batch_base.reshape(update_codebook_batch_size // 16,
                                               16, self.dim // 16, 16).transpose(0, 2, 1, 3)

                        sub_codes, _ = \
                            self.models.run(1, [self.codebook[:,
                                                id_ * self.block_size:(id_ + 1) *
                                                                     self.block_size].T.astype(np.float16),
                                                batch_base_reshape, actual_nums])
                        sub_codes = sub_codes.T

                        sub_codes_reshape = \
                            sub_codes[:update_codebook_batch_size].reshape(update_codebook_batch_size // 16,
                                                                           16,
                                                                           self.block_size // 16,
                                                                           16).transpose(0, 2, 1, 3)
                        actual_nums[0][0] = update_codebook_batch_size

                        cb, _ = \
                            self.models.run(2,
                                            [self.codebook[:,
                                             id_ * self.block_size:(id_ + 1) * self.block_size].astype(np.float16),
                                             sub_codes_reshape, actual_nums])

                        loss += \
                            np.sum(((batch_base -
                                     cb[:, :update_codebook_batch_size].T.astype(np.float32)) ** 2)[:,
                                   :min(update_codebook_batch_size, len(msk) - i * update_codebook_batch_size)])

            codes_update_time = time.time() - start_time
            start_time = time.time()

            codebook_update_time = time.time() - start_time
            if codebook_dir is not None:
                if not os.path.exists(codebook_dir):
                    os.mkdir(codebook_dir, mode=0o750)
                self.save_codebook(codebook_dir)
                logging.info("Save codebook to %s", codebook_dir)

            if self.verbose:
                logging.info('subspace_cluster_network loss: %f, time: %.2f, choose_block_time: %.2f, '
                             'codes_update_time: %.2f, codebook_update_time: %.2f',
                             loss / base_num, time.time() - st, choose_block_cal_time,
                             codes_update_time, codebook_update_time)

    # 保留
    def cal_nonzero_idx(self, sub_base):
        res, _ = self.models.run(0,
                                 [sub_base.astype(np.float16),
                                  self.codebook_reshape_bytes, np.zeros(self.k).astype(np.float32)])
        labels_1 = np.argmax(res, -1)
        return labels_1

    # 保留
    def orth_codebook(self):
        for k_i in tqdm(range(self.k), desc='orth:'):
            C_i = self.codebook[:, k_i * self.block_size:(k_i + 1) * self.block_size]
            U, _, _ = np.linalg.svd(C_i, full_matrices=0)
            self.codebook[:, k_i * self.block_size:(k_i + 1) * self.block_size] = U[:, :]

    # 保留
    def update_codebook(self, sub_base, k_i, return_sigma=False):
        start = time.time()
        U, S, Vh = np.linalg.svd(sub_base.T, full_matrices=0)
        end = time.time()
        logging.info("%s", str(end - start))
        sigma_num = min(len(S), self.block_size)
        self.codebook[:, k_i * self.block_size:k_i * self.block_size + sigma_num] = U[:, :self.block_size]

    def update_codebook_acl(self, sub_base, k_i, return_sigma=False):
        actual_num = np.zeros((8,), dtype=np.uint64)
        actual_num[0] = sub_base.shape[0]

        output, _ = \
            self.models.run(3,
                            [sub_base.T.reshape((1, self.dim,
                                                 int(actual_num[0] // 16), 16)).transpose(0, 2, 1, 3), actual_num])

        try:
            U, S, _ = np.linalg.svd(output.astype(np.float32), full_matrices=0)
            sigma_num = min(len(S), self.block_size)
        except np.linalg.LinAlgError as e:
            warnings.warn(f"{str(e)}; processing of the current bucket will be skipped.",
                          category=RuntimeWarning)
        else:
            self.codebook[:, k_i * self.block_size:k_i * self.block_size + sigma_num] = U[:, :self.block_size]

    def save_codebook(self, file_dir):
        codebook_name_npy = f'codebook_{self.dim}_{self.block_size}_{self.k}.npy'
        codebook_name_bin = f'codebook_{self.dim}_{self.block_size}_{self.k}.bin'
        npy_filename = os.path.join(file_dir, codebook_name_npy)
        bin_filename = os.path.join(file_dir, codebook_name_bin)

        def _save_exists():
            bin_write_exists(bin_filename, self.blank, self.codebook.astype(np.float32).T, self.version_major,
                      self.version_medium, self.version_minor, self.dim, self.block_size, self.k)
            
        def _save_not_exists():
            bin_write_not_exists(bin_filename, self.blank, self.codebook.astype(np.float32).T, self.version_major,
                      self.version_medium, self.version_minor, self.dim, self.block_size, self.k)

        if not os.path.isdir(file_dir):
            raise Exception('directory [{}] not exist'.format(file_dir))
        if islink_recursive(npy_filename):
            raise Exception('directory/file [{}] is soft link'.format(npy_filename))
        if islink_recursive(bin_filename):
            raise Exception('directory/file [{}] is soft link'.format(bin_filename))
        if not os.access(file_dir, os.W_OK):
            raise Exception('cannot write to this file [{}]'.format(file_dir))

        # save npy 文件
        np.save(npy_filename, self.codebook)
        os.chmod(npy_filename, 0o640)
        
        if not os.path.exists(bin_filename):
            _save_not_exists()
        else:
            _save_exists()
        
        return 0

    # 保留
    def load_codebook(self, file_dir):

        def _check_file(file_path, max_size):
            with open(file_path) as f:
                file_info = os.stat(f.fileno())
                # 校验文件是否是软连接
                if not os.path.isfile(file_path):
                    raise Exception("not regular file！")
                if os.path.islink(file_path):
                    raise Exception("file is a soft link！")
                # 校验文件大小
                if file_info.st_size > max_size:
                    raise Exception("file too large！")
        codebook_path_to_load = f'{file_dir}/codebook_{self.dim}_{self.block_size}_{self.k}.npy'
        _check_file(codebook_path_to_load, MAX_SIZE)
        self.codebook = np.load(codebook_path_to_load)
        self.reshape_codebook()


def islink_recursive(filename):
    file_name = filename.rstrip(os.sep)
    parts = filename.split(os.sep)
    if not parts:
        return False
    start = 2 if parts[0] == '' else 1
    for i in range(start, len(parts) + 1):
        subpath = os.sep.join(parts[0:i])
        if os.path.islink(subpath):
            return True
    return False


# 如果文件存在，保存逻辑                
def bin_write_exists(fname, blank_data, codebook_data, version_major, version_medium, version_minor,
              dim, nonzeroNum, nlist):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(fname, flags, modes), 'ab') as fout:
        file_info = os.stat(fout.fileno())
        # 校验Others的权限是否为可写或可执行
        permissions = file_info.st_mode
        others_permissions = permissions & stat.S_IRWXO
        if others_permissions & stat.S_IWOTH:
            raise ValueError(f"{fname} cannot be writable by 'Others'.")
        if others_permissions & stat.S_IXOTH:
            raise ValueError(f"{fname} cannot be exeuctable by 'Others'.")
        # 校验是否可写
        if not os.access(fname, os.W_OK):
            raise ValueError(f'Current user do not hava access to write path {fname}.')
        # 迭代校验该路径及其父目录的软连接
        if islink_recursive(fname):
            raise ValueError(f'Provided file name {fname} or one of its parent directories has softlink.')  
        # 校验属主
        if file_info.st_uid != os.geteuid():
            raise ValueError(f'The owner of {fname} and current user do not match.')
        # 校验最大大小
        if file_info.st_size > MAX_SIZE:
            raise ValueError(f'The size of {fname} exceeds maximum possible codebook size.')
        
        # 将文件原本的内容清0
        fout.seek(0)
        fout.truncate(0)
        fout.write(struct.pack('c', b"C"))
        fout.write(struct.pack('c', b"D"))
        fout.write(struct.pack('c', b"B"))
        fout.write(struct.pack('c', b"K"))
        fout.write(struct.pack('B', version_major))
        fout.write(struct.pack('B', version_medium))
        fout.write(struct.pack('B', version_minor))
        fout.write(struct.pack('i', dim))
        fout.write(struct.pack('i', nonzeroNum))
        fout.write(struct.pack('i', nlist))

        for line in blank_data:
            for c in line:
                fout.write(struct.pack('B', c))

        for line in codebook_data:
            for c in line:
                fout.write(struct.pack('f', c))

 
# 如果文件不存在，保存逻辑          
def bin_write_not_exists(fname, blank_data, codebook_data, version_major, version_medium, version_minor,
              dim, nonzeroNum, nlist):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(fname, flags, modes), 'wb') as fout:
        file_info = os.stat(fout.fileno())
        # 校验是否可写
        if not os.access(fname, os.W_OK):
            raise ValueError(f'Current user do not hava access to write path {fname}.')
        # 迭代校验该路径及其父目录的软连接
        if islink_recursive(fname):
            raise ValueError(f'Provided file name {fname} or one of its parent directories has softlink.')
        # 赋予640权限
        os.fchmod(fout.fileno(), 0o640)
        
        fout.write(struct.pack('c', b"C"))
        fout.write(struct.pack('c', b"D"))
        fout.write(struct.pack('c', b"B"))
        fout.write(struct.pack('c', b"K"))
        fout.write(struct.pack('B', version_major))
        fout.write(struct.pack('B', version_medium))
        fout.write(struct.pack('B', version_minor))
        fout.write(struct.pack('i', dim))
        fout.write(struct.pack('i', nonzeroNum))
        fout.write(struct.pack('i', nlist))

        for line in blank_data:
            for c in line:
                fout.write(struct.pack('B', c))

        for line in codebook_data:
            for c in line:
                fout.write(struct.pack('f', c))


def load_data_check(file_path, max_size):
    with open(file_path) as f:
        file_info = os.stat(f.fileno())
        # check if file is soft link
        if not os.path.isfile(file_path):
            raise Exception("not regular file!")
        if os.path.islink(file_path):
            raise Exception("file is a soft link!")
        # check size of file
        if file_info.st_size > max_size:
            raise Exception("file too large!")