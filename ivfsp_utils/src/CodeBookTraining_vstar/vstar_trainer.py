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

import random
import sys
import torch
from sklearn.cluster import KMeans
import torch_npu
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import stat
import time
import argparse
import threading
import multiprocessing
from numpy import linalg as LA
import struct


def combine_codebook(codebook1, codebook2):
    codebook11_binary = codebook1.tobytes()
    codebook12_binary = codebook2.tobytes()

    combined = codebook11_binary + codebook12_binary
    return combined


def add_verify_header():
    verify_header = b'CBSP'
    return verify_header


def add_meta_data(codebook1, codebook2):
    dim = codebook1.shape[0]
    nlist_l1 = codebook2.shape[0]
    subspacedim_l1 = codebook2.shape[1]
    nl2xsd2 = codebook2.shape[2]
    dim_4byte = struct.pack('<I', dim)
    nlist_l1_4byte = struct.pack('<I', nlist_l1)
    subspacedim_l1_4byte = struct.pack('<I', subspacedim_l1)
    nl2xsd2_4byte = struct.pack('<I', nl2xsd2)
    combined = dim_4byte + nlist_l1_4byte + subspacedim_l1_4byte + nl2xsd2_4byte
    return combined


# 迭代校验软连接
def islink_recursive(file_name):
    file_name = file_name.rstrip(os.sep)
    parts = file_name.split(os.sep)
    if not parts:
        return False
    start = 2 if parts[0] == '' else 1
    for i in range(start, len(parts) + 1):
        subpath = os.sep.join(parts[0:i])
        if os.path.islink(subpath):
            return True
    return False


# 如果文件不存在，保存逻辑
def save_not_exists(file_name, combined):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(file_name, flags, modes), 'wb') as file:
        # 校验是否可写
        if not os.access(file_name, os.W_OK):
            raise ValueError(f'Current user do not hava access to write path {file_name}.')
        # 迭代校验该路径及其父目录的软连接
        if islink_recursive(file_name):
            raise ValueError(f'Provided file name {file_name} or one of its parent directories has softlink.')
        # 赋予640权限
        os.fchmod(file.fileno(), 0o640)
        file.write(combined)
        

# 如果文件已经存在，保存逻辑
def save_exists(file_name, combined):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(file_name, flags, modes), 'ab') as file:
        file_info = os.stat(file.fileno())
        # 校验Others的权限是否为可写或可执行
        permissions = file_info.st_mode
        others_permissions = permissions & stat.S_IRWXO
        if others_permissions & stat.S_IWOTH:
            raise ValueError(f"{file_name} cannot be writable by 'Others'.")
        if others_permissions & stat.S_IXOTH:
            raise ValueError(f"{file_name} cannot be exeuctable by 'Others'.")
        # 校验是否可写
        if not os.access(file_name, os.W_OK):
            raise ValueError(f'Current user do not hava access to write path {file_name}.')
        # 迭代校验该路径及其父目录的软连接
        if islink_recursive(file_name):
            raise ValueError(f'Provided file name {file_name} or one of its parent directories has softlink.')  
        # 校验属主
        if file_info.st_uid != os.geteuid():
            raise ValueError(f'The owner of {file_name} and current user do not match.')
        # 校验最大大小
        max_size = 2 * 1024 * 1024 * 1024 # 2GB is the max size for a CodeBook
        if file_info.st_size > max_size:
            raise ValueError(f'The size of {file_name} exceeds maximum possible codebook size.')
        # 将文件原本的内容清0
        file.seek(0)
        file.truncate(0)
        file.write(combined)


###################################################################################################################


def cal_nonzero_idx(base, codebook, nlist0, subspacedim):
    res = torch.sum(
        (torch.matmul(base, codebook) ** 2).reshape(base.shape[0], nlist0, subspacedim), -1)
    labels_1 = torch.argmax(res, -1)
    return labels_1


def orth_codebook(codebook, nlist, subspacedim, device):
    for k in range(nlist):
        k_codebook = codebook[:, k * subspacedim:(k + 1) * subspacedim]
        u, _, __ = torch.linalg.svd(k_codebook.to(torch.float32), full_matrices=False)
        codebook[:, k * subspacedim:(k + 1) * subspacedim] = u[:, :].to(torch.float32)
                


def update_codebook(sub_base, codebook, k_i, subspacedim, device):
    if len(sub_base) < 100:
        print("WARNING: BucketId(%d) is too small, codebook not update!" % (k_i))
        return
    try:
        u, s, vh = torch.linalg.svd(sub_base.T.to(torch.float32), full_matrices=False)
        sigma_num = min(len(s), subspacedim)
        codebook[:, k_i * subspacedim:k_i * subspacedim + sigma_num] = u[:, :subspacedim].to(torch.float32)
    except IndexError as e:  
        print(f"IndexError: {e}")  # 处理索引错误 
        return 
    except RuntimeError as e:  # 处理运行时错误  
        print(f"RuntimeError: {e}") 
        return 
    except Exception as e:  
        print(f"updating codebook for bucket {k_i} failed unexpectedly.Training process continuing...")
        return

###################################################################################################################


class VstarTrainer:
    def __init__(
        self, dim, nlist_l1, nlist_l2, subspacedim_l1, subspacedim_l2, seed=10, eps=1e-10,
        verbose=False, device=0, logger=None
    ):
        self.dim = dim
        self.nlist_l1 = nlist_l1
        self.nlist_l2 = nlist_l2
        self.subspacedim_l1 = subspacedim_l1
        self.subspacedim_l2 = subspacedim_l2
        self.verbose = verbose
        self.device = torch.device(f'npu:{device}') if device != -1 else torch.device('cpu')
        self.seed = seed
        self.eps = eps
        np.random.seed(self.seed)
        self.codebook_l1 = torch.tensor(np.random.randn(self.dim,
                            nlist_l1 * subspacedim_l1)).to(torch.float32).to(self.device)
        self.codebook_l2 = torch.tensor(np.random.randn(nlist_l1,
                            self.subspacedim_l1, nlist_l2 * subspacedim_l2)).to(torch.float32).to(self.device)
        self.logger = logger

    def codebook_kmeans_init(self, base):
        kmeans = KMeans(n_clusters=self.nlist_l1, verbose=0, max_iter=16, init='random', n_init=1).fit(base)
        kmeans_labels = kmeans.labels_
        base_num = base.shape[0]
        base_dim = base.shape[1]
        update_size = min((base_num // self.nlist_l1 + 1023) // 1024 * 1024, 32876) # ~13312
        batch_base = torch.tensor(np.zeros((update_size, base_dim))).to(torch.float32).to(self.device)
        labels_unique = list(set(kmeans_labels))
        for i in tqdm(range(len(labels_unique)), desc='Codebook Initiation'):
            msk = (kmeans_labels == labels_unique[i])
            sub_base = base[msk, :].to(torch.float32).to(self.device)
            batch_base *= 0.
            batch_base[:min(update_size, len(sub_base)), :] = sub_base[:min(update_size, len(sub_base)), :]
            update_codebook(batch_base, self.codebook_l1, labels_unique[i], self.subspacedim_l1, self.device)

    def codebook_to_numpy(self):
        codebook = []
        for i in range(self.nlist_l1):
            codebook.append(self.codebook_l1[:, i*self.subspacedim_l1:(i+1)*self.subspacedim_l1].to('cpu').numpy())
        return np.concatenate(codebook, -1)

    def save_index(self, file_dir):
        self.save_index_bin(file_dir)

    def save_index_bin(self, real_codebook_dir):
        # save the combined CB1 and CB2 into one
        codebook1_orig = self.codebook_to_numpy()
        codebook2_orig = self.codebook_l2.to('cpu').numpy()
        codebook1 = self.codebook_to_numpy().transpose(1, 0)
        codebook2 = self.codebook_l2.to('cpu').numpy().transpose(0, 2, 1)
        combined = add_verify_header() + \
                    add_meta_data(codebook1_orig, codebook2_orig) + \
                    combine_codebook(codebook1, codebook2)
        file_path = f'{real_codebook_dir}/codebook_{self.dim}_{self.nlist_l1}_{self.subspacedim_l1}.bin'
        if (os.path.exists(file_path)):
            save_exists(file_path, combined)
        else:
            save_not_exists(file_path, combined)
            
    def train_sp(self, codebook, base, nlist, subspacedim, num_iter=10, batch_size=10240,
                 msg="L1 Train Process", l2_training=False):
        orth_codebook(codebook, nlist, subspacedim, self.device)
        iter_list = tqdm(range(num_iter), desc=msg) if (self.verbose and not l2_training) else range(num_iter)
        for _ in iter_list:
            base_num = base.shape[0]
            base_dim = base.shape[1]
            nonzero_idx = torch.LongTensor(np.zeros(base_num)).to(self.device)
            num_batch = (base_num + batch_size - 1) // batch_size
            update_size = min((base_num // nlist + 1023) // 1024 * 1024, 32876) # ~13312

            start_idx = 0
            loss = torch.tensor([0.0]).to(self.device)
            
            for batch_i in range(num_batch):
                batch_x = base[batch_i * batch_size:(batch_i + 1) * batch_size]
                batch_x = batch_x.to(torch.float32).to(self.device)
                batch_sample_num = batch_x.shape[0]
                sub_nonzero_idx = cal_nonzero_idx(batch_x, codebook, nlist, subspacedim)
                nonzero_idx[start_idx:start_idx + batch_sample_num] = sub_nonzero_idx[:]
                start_idx += batch_sample_num

            batch_base = torch.tensor(np.zeros((update_size, base_dim))).to(torch.float32).to(self.device)
            
            iter_update_codebook = tqdm(range(nlist), desc='update codebook') if not l2_training else range(nlist)
            
            for level1_idx in iter_update_codebook:
                msk = torch.where(nonzero_idx == level1_idx)[0].to('cpu')
                # get all base vectors whose L1 index == level1_idx
                sub_base = base[msk, :].to(torch.float32).to(self.device)

                batch_base *= 0. # zero out batch_base completely
                batch_base[:min(update_size, len(msk)), :] = sub_base[:min(update_size, len(msk)), :]

                # batch base should be all base vectors whose L1 index == level1_idx
                update_codebook(batch_base, codebook, level1_idx, subspacedim, self.device)


    def train(self, base, num_iter=1, batch_size=10240, is_cal_loss=True, index_dir=None):
        base = torch.from_numpy(base)
        self.codebook_kmeans_init(base)
        self.train_sp(self.codebook_l1, base, self.nlist_l1, self.subspacedim_l1,
                      num_iter=num_iter, batch_size=batch_size)

        if self.nlist_l2 > 0:
            base_num = base.shape[0]
            num_batch = (base_num + batch_size - 1) // batch_size
            start_idx = 0
            nonzero_idx = torch.LongTensor(np.zeros(base_num)).to(self.device)
            for batch_i in tqdm(range(num_batch), desc='L1 Bucket Classification'):
                batch_x = base[batch_i*batch_size:(batch_i+1)*batch_size]
                batch_x = batch_x.to(torch.float32).to(self.device)
                batch_sample_num = batch_x.shape[0]
                sub_nonzero_idx = cal_nonzero_idx(batch_x, self.codebook_l1, self.nlist_l1, self.subspacedim_l1)
                nonzero_idx[start_idx:start_idx+batch_sample_num] = sub_nonzero_idx[:]
                start_idx += batch_sample_num

            for level1_idx in tqdm(range(self.nlist_l1), desc="level2 Training"):
                msk = torch.where(nonzero_idx == level1_idx)[0].to('cpu')
                sub_base = base[msk, :].to(torch.float32).to(self.device)
                sub_base_code = torch.matmul(sub_base,
                                self.codebook_l1[:, level1_idx * \
                                self.subspacedim_l1: (level1_idx + 1) * self.subspacedim_l1]).to('cpu')
                self.train_sp(self.codebook_l2[level1_idx], sub_base_code, self.nlist_l2,
                              self.subspacedim_l2, num_iter=num_iter, batch_size=batch_size, l2_training=True)
                
        self.save_index(index_dir)