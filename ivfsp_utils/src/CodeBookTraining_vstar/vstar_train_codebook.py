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

import numpy as np
import argparse
import os
import torch
from vstar_trainer import VstarTrainer


def arg_parse():
    """
    Parse arguements to generate codebook
    """

    parser = argparse.ArgumentParser(
        description='generate Vstar offline operators')

    parser.add_argument("--dataPath",
                        dest='data_path',
                        required=True,
                        type=str,
                        help="Path to Your Data")

    parser.add_argument("--codebookPath",
                        dest='codebook_path',
                        default='.',
                        type=str,
                        help="Path to Save Your Codebooks")

    parser.add_argument("--dim",
                        dest='dim',
                        default=256,
                        type=int,
                        help="Data Dimension")

    parser.add_argument("--nlistL1",
                        dest='nlist_l1',
                        default=1024,
                        type=int,
                        help="L1 Bucket Number")

    parser.add_argument("--subDimL1",
                        dest='sub_dim_l1',
                        default=32,
                        type=int,
                        help="Subspace Dimension L1")

    parser.add_argument("--device",
                        dest='device',
                        default=1,
                        type=int,
                        help="NPU Phyiscal ID")

    parser.add_argument("--batchSize",
                        dest='batch_size',
                        default=10240,
                        type=int,
                        help="BatchSize on NPU")

    parser.add_argument("--sample",
                        dest='sample',
                        default=1.0,
                        type=float,
                        help="Sample rate of base data")
    
    parser.add_argument("--useOfflineCompile",
                        action="store_true",
                        help="Whether to use offline aclnn ops from kernel package")

    return parser.parse_args()


def load_base_bin(file_path: str, dimension: int):
    """
    read data in bin format
    """
    max_base_size = 4 * 1024 * 10000000 # max file size to prevent too high CPU usage; change if needed
    file_size = os.path.getsize(file_path)
    if file_size > max_base_size:
        raise ValueError(f"Max base size exceeded; max base size supported is {max_base_size},"
                         f"your input base size is {file_size}")
    if os.path.islink(file_path): # 软连接校验
        raise ValueError("base data file path is a symbolic link.")
    if not os.path.isfile(file_path): # 存在的寻常文件校验
        raise ValueError("base data file is not a regular file.")

    print(f"Loading from file {file_path}, dimension is {dimension}")
    return np.fromfile(f'{file_path}', dtype='float32').reshape(-1, dimension).copy()


def check_param_range(param: int, valid_params: [int], param_name: str):
    if param not in valid_params:
        raise ValueError("not support {0}: {1}, should be in {2}".format(param_name, param, valid_params))


def check_iter_size(iter_size: int):
    if iter_size > 1:
        print("Warning: You have tried to do more than 1 iteration of training. Typically, 1 iteration is enough, and adding more iterations has diminishing returns.")
    if iter_size < 0 or iter_size > 10:
        raise ValueError("input variable iter {0}, should be in range[1, 10]".format(iter_size))


def check_device(device_id: int):
    if not torch.npu.is_available():
        print("Warning: You do not appear to have NPU available on this machine and local CPU will be used. Performance degrade expected.")
        return False
    possible_npu = [i for i in range(torch.npu.device_count())]
    check_param_range(device_id, possible_npu, "npu id")
    return True


def check_batch(size: int):
    correct_condition = (size > 0) and (size <= 10240)
    if not correct_condition:
        raise ValueError("not support batchsize {0}: should be between (0, 10240].".format(size))


def check_sample(sample_data: float):
    if sample_data == 1.0:
        return
    elif sample_data <= 0.0 or sample_data > 1.0:
        raise ValueError("input variable sample {0}, should be in range(0, 1].".format(sample_data))
    else:
        print("You are setting sample rate for base to be less than 1, which means the resulted codebooks might not capture the characteristics of the entire dataset.")


def check_codebook_path(path: str):
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist.")
    if os.path.islink(path):
        raise ValueError("The path to save the output codebook file cannot be a symbolic link.")
    if not os.access(path, os.W_OK):
        raise ValueError(f'Current user do not hava access to write path {path}.')


if __name__ == '__main__':
    args = arg_parse()

    # Get the value of the environment variable
    openblas_num_threads = os.environ.get('OPENBLAS_NUM_THREADS')
    if openblas_num_threads is None or int(openblas_num_threads) != 1:
        raise ValueError("Please set environmental variable 'OPENBLAS_NUM_THREADS' to 1.")

    data_path = args.data_path
    codebook_path = args.codebook_path
    dim = args.dim
    check_param_range(dim, [128, 256, 512, 1024], "dim") # 暂时仅支持dim = 128, 256, 512, 1024

    # set sub_dims
    sub_dim_l1 = args.sub_dim_l1
    if sub_dim_l1 >= dim:
        raise ValueError("sub_dim_l1 should be less than dim. Currently sub_dim_l1 = {0}, dim = {1}".format(sub_dim_l1, dim))
    check_param_range(sub_dim_l1, [32, 64, 128], "sub_dim_l1")
    sub_dim_l2 = int(sub_dim_l1/2)

    # set nlists
    nlist_l1 = args.nlist_l1
    check_param_range(nlist_l1, [256, 512, 1024], "nlist_l1")
    if dim in [128, 256, 512]:
        nlist_l2 = 32
    else:
        nlist_l2 = 16

    device = args.device
    batch_size = args.batch_size
    sample = args.sample

    check_codebook_path(codebook_path)

    if not check_device(device):
        device = -1
    check_batch(batch_size)
    check_sample(sample)

    print(f"dim = {dim}, nlistL1 = {nlist_l1}, nlistL2 = {nlist_l2}, subdimL1 = {sub_dim_l1}, subdimL2 = {sub_dim_l2}")

    print(
        f"=========================================================================\n",
        f"Currently you are using the following parameters to generate codebooks:\n",
        f"dim = {dim};\n nlistL1 = {nlist_l1};\n subspaceDimL1 = {sub_dim_l1};\n",
        f"You are using batchSize = {batch_size};\n",
        f"You are using device ID = {device}, if this value is -1 it means you don't have NPU on this machine;\n",
        f"You are using {sample * 100}% of data from file path: {data_path};\n",
        f"You are saving the codebook to file path: {codebook_path}/codebook_{dim}_{nlist_l1}_{sub_dim_l1}.bin.\n"
        f"=========================================================================\n"
    )

    base = load_base_bin(data_path, dim)

    if sample < 1.0:
        base = base[np.random.choice(len(base), int(len(base) * sample), replace=False)]

    # set whether to compile ops online
    use_offline_compile = args.useOfflineCompile
    if use_offline_compile:
        torch.npu.set_compile_mode(jit_compile=False)

    trainer = VstarTrainer(dim, nlist_l1, nlist_l2, sub_dim_l1, sub_dim_l2, device=device)
    trainer.train(base, batch_size=batch_size, index_dir=codebook_path)