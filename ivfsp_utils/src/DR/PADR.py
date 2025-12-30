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
import random
import os
import csv
import logging
import itertools
import pickle
import time
import torch
import torch_npu
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import faiss
from faiss import normalize_L2

LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class TrainDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        img = self.images[index]
        return img

    def __len__(self):
        return len(self.images)


class AutoEncoder(nn.Module):
    def __init__(self, ratio, dim):
        super(AutoEncoder, self).__init__()
        self.ratio = ratio
        self.dim = dim
        if (self.ratio >= 2):
            self.encoder1 = nn.Sequential(
                nn.Linear(self.dim, int(self.dim / 2)),
            )
        if (self.ratio >= 4):
            self.encoder2 = nn.Sequential(
                nn.Linear(int(self.dim / 2), int(self.dim / 4)),
            )
        if (self.ratio >= 8):
            self.encoder3 = nn.Sequential(
                nn.Linear(int(self.dim / 4), int(self.dim / 8)),
            )
        if (self.ratio >= 16):
            self.encoder4 = nn.Sequential(
                nn.Linear(int(self.dim / 8), int(self.dim / 16)),
            )

    def forward(self, x):
        result = []
        result.append(self.encoder1(x))
        if self.ratio >= 4:
            result.append(self.encoder2(result[-1]))
        if self.ratio >= 8:
            result.append(self.encoder3(result[-1]))
        if self.ratio >= 16:
            result.append(self.encoder4(result[-1]))
        return result


class ModelExport(AutoEncoder):
    def __init__(self, ratio, dim):
        super(ModelExport, self).__init__(ratio, dim)
        self.ratio = ratio
        self.dim = dim
        if (self.ratio >= 2):
            self.encoder1 = nn.Sequential(
                nn.Linear(self.dim, int(self.dim / 2)),
            )
        if (self.ratio >= 4):
            self.encoder2 = nn.Sequential(
                nn.Linear(int(self.dim / 2), int(self.dim / 4)),
            )
        if (self.ratio >= 8):
            self.encoder3 = nn.Sequential(
                nn.Linear(int(self.dim / 4), int(self.dim / 8)),
            )
        if (self.ratio >= 16):
            self.encoder4 = nn.Sequential(
                nn.Linear(int(self.dim / 8), int(self.dim / 16)),
            )

    def forward(self, x):
        result = []
        result.append(self.encoder1(x))
        if self.ratio >= 4:
            result.append(self.encoder2(result[-1]))
        if self.ratio >= 8:
            result.append(self.encoder3(result[-1]))
        if self.ratio >= 16:
            result.append(self.encoder4(result[-1]))
        return result[-1]


def get_all_data(dataset_dir, train_size):
    max_val_size = 100 * 1024 * 1024 * 1024  # 100G
    load_data_check(os.path.join(os.path.join(dataset_dir, "base.npy")), max_val_size)
    load_data_check(os.path.join(os.path.join(dataset_dir, "query.npy")), max_val_size)
    load_data_check(os.path.join(os.path.join(dataset_dir, "gt.npy")), max_val_size)
    base = np.load(os.path.join(dataset_dir, "base.npy")).astype(np.float32)
    query = np.load(os.path.join(dataset_dir, "query.npy")).astype(np.float32)
    gt = np.load(os.path.join(dataset_dir, "gt.npy"))

    x = np.arange(0, base.shape[0])

    rand_idx = np.random.choice(x, train_size, replace=False)
    learn = base[rand_idx]

    logging.info("learn shape: %s", str(learn.shape))
    logging.info("base shape: %s", str(base.shape))
    logging.info("query shape: %s", str(query.shape))
    logging.info("gt shape: %s", str(gt.shape))
    return learn, base, query, gt


def get_train_data(dataset_dir, train_size):
    max_val_size = 100 * 1024 * 1024 * 1024  # 100G
    load_data_check(os.path.join(os.path.join(dataset_dir, "base.npy")), max_val_size)
    base = np.load(os.path.join(dataset_dir, "base.npy")).astype(np.float32)
    x = np.arange(0, base.shape[0])
    rand_idx = np.random.choice(x, train_size, replace=False)
    learn = base[rand_idx]
    logging.info("learn shape: %s", str(learn.shape))
    logging.info("base shape: %s", str(base.shape))
    return learn, base


def cal_acc(query, gt, I):
    t1, t10, t50, t100 = 0, 0, 0, 0
    query_shape = query.shape[0]
    for i in range(query_shape):
        if I[i][0] == gt[i][0]:
            t1 += 1
        if gt[i][0] in I[i][:10]:
            t10 += 1
        if gt[i][0] in I[i][:50]:
            t50 += 1
        if gt[i][0] in I[i][:100]:
            t100 += 1
    if query_shape == 0:
        logging.error("Query shape is zero!")
        return [0, 0, 0, 0]
    else:
        logging.info("t1: %4f, t10: %4f, t50: %4f, t100: %4f", t1 / query_shape,
                     t10 / query_shape, t50 / query_shape, t100 / query_shape)
        return [t100 / query_shape, t50 / query_shape,
                t10 / query_shape, t1 / query_shape]


def cal_recall(query, gt, I, K, k):
    query_shape = query.shape[0]
    if query_shape == 0:
        logging.error("Query shape is zero!")
    else:
        for val in k:
            if val > K:
                break
            total = 0.0
            for i in range(query_shape):
                tmp = 0.0
                for j in range(val):
                    if gt[i][j] in I[i][:K]:
                        tmp += 1
                if val != 0:
                    total += tmp / val
            logging.info("k= %d, r100: %4f", val, total / query_shape)


def construct_dataset(data, batch_size, near):
    dataset_list = []
    d = data.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(data)
    logging.info("---> Start 'construct_dataset' index Search!")
    D, I = index.search(data, near)

    logging.info("---> Start 'construct_dataset' construct dataset!")
    for i in range(data.shape[0]):
        for j in range(near):
            dataset_list.append((data[i], data[I[i][j]]))

    train_data = TrainDataset(dataset_list)
    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return loader


def train_AE(train_params):
    save_path = train_params.save_path
    learn = train_params.data_learn
    ratio = train_params.ratio
    dim = train_params.dim
    val_dataset_dir = train_params.val_dataset_dir
    device = train_params.run_device
    batch_size = train_params.train_batch_size
    learning_rate = train_params.learning_rate
    epochs = train_params.epochs
    log_stride = train_params.log_stride
    construct_neighbors = train_params.construct_neighbors
    infer_batch_size = train_params.infer_batch_size

    logging.info("===> Start construct dataset")
    dataset_loader = construct_dataset(learn, batch_size, construct_neighbors)  # cost 15s for 100000 data points

    logging.info("===> Dataset has %d batches", len(dataset_loader))
    model = AutoEncoder(ratio, dim)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if val_dataset_dir:
        best_array = []
    else:
        best = 0.
    for epoch in range(epochs):
        t1 = time.time()
        total_loss = 0
        for i, (x1, x2) in enumerate(dataset_loader):
            x1, x2 = x1.to(device), x2.to(device)

            encoder1 = model(Variable(x1))
            encoder2 = model(Variable(x2))
            loss = 0.
            for j, _ in enumerate(encoder1):
                loss = loss + criterion(F.pairwise_distance(encoder1[j], encoder2[j], p=2),
                                        F.pairwise_distance(x1, x2, p=2))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
            if i % log_stride == 0:
                logging.info("epoch: %2d, step: %2d, loss: %.6f", epoch, i, loss.data)
        t2 = time.time()
        # validation
        logging.info('===> epoch time: %s sec', (t2 - t1))
        if val_dataset_dir:
            logging.info("===> start validation for epoch %d", epoch)
            max_val_size = 10 * 1024 * 1024 * 1024 # 10G
            load_data_check(os.path.join(val_dataset_dir, "val_query.npy"), max_val_size)
            load_data_check(os.path.join(val_dataset_dir, "val_gt.npy"), max_val_size)
            load_data_check(os.path.join(val_dataset_dir, "val_base.npy"), max_val_size)
            val_query = torch.Tensor(np.load(os.path.join(val_dataset_dir, "val_query.npy"))).to(device)
            val_gt = np.load(os.path.join(val_dataset_dir, "val_gt.npy"))
            val_base = torch.Tensor(np.load(os.path.join(val_dataset_dir, "val_base.npy"))).to(device)
            val_base = model(val_base)[-1]
            val_base = val_base.to("cpu").detach().numpy()
            val_query = model(val_query)[-1]
            val_query = val_query.to("cpu").detach().numpy()
            d = val_base.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(val_base)
            D, I = index.search(val_query, 100)
            now = cal_acc(val_query, val_gt, I)
            logging.info("============================================================================")
            if epoch == 0 or best_array < now:
                best_array = now
                modeltmp = model.cpu()
                save_data_check(save_path, "best.onnx", modeltmp, "model", ratio, dim, infer_batch_size)
                model = model.to(device)
                logging.info("best model update.")
        else:
            if epoch == 0 or best > total_loss:
                best = total_loss
                modeltmp = model.cpu()
                save_data_check(save_path, "best.onnx", modeltmp, "model", ratio, dim, infer_batch_size)
                model = model.to(device)
                logging.info("best model update.")
        if epoch % 5 == 4 or epoch == 0:
            modeltmp = model.cpu()
            save_data_check(save_path, str(epoch) + ".onnx", modeltmp, "model", ratio, dim, infer_batch_size)
            model = model.to(device)


def prepare_data(val_dataset_dir, raw_data, val_num):
    logging.info("---> Getting Validation Set from %s", str(val_dataset_dir))
    x = np.arange(0, raw_data.shape[0])
    rand_idx = np.random.choice(x, val_num, replace=False)
    validation_query = raw_data[rand_idx]
    logging.info("raw data shape: %s", str(raw_data.shape))
    near = 100
    d = raw_data.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(raw_data)
    D, I = index.search(validation_query, near + 1)
    I = I[:, 1:]

    I1 = np.array(I).reshape(-1, )
    uniques = np.unique(I1)
    validation_base = raw_data[uniques]

    index.reset()
    index.add(validation_base)
    _, INew = index.search(validation_query, near)

    save_data_check(val_dataset_dir, "val_gt.npy", INew, "data")
    save_data_check(val_dataset_dir, "val_base.npy", validation_base, "data")
    save_data_check(val_dataset_dir, "val_query.npy", validation_query, "data")


def islink_recursive(filename):
    filename.rstrip(os.sep)
    parts = filename.split(os.sep)
    if not parts:
        return False
    start = 2 if parts[0] == '' else 1
    for i in range(start, len(parts) + 1):
        subpath = os.sep.join(parts[0:i])
        if os.path.islink(subpath):
            return True
    return False


def save_data_check(file_dir, suffix, target, mode, ratio=None, data_dim=None, infer_batch=None):
    filename = os.path.join(file_dir, suffix)

    def _save():
        if mode == "data":
            np.save(filename, target)
        elif mode == "model":
            pretrained_dict = {'state_dict': target.state_dict()}
            model_export = ModelExport(ratio, data_dim)
            model_dict = model_export.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model_export.load_state_dict(model_dict)

            dummy_input = torch.randn(infer_batch, data_dim)
            export_onnx_file = filename
            input_names = ["actual_input_1"]
            output_names = ["output1"]
            torch.onnx.export(model_export, dummy_input, export_onnx_file, input_names=input_names,
                              output_names=output_names, opset_version=11)
        else:
            raise Exception("mode not support!")
        os.chmod(filename, 0o640)

    if not os.path.isdir(file_dir):
        raise Exception('directory [{}] not exist'.format(file_dir))
    if islink_recursive(filename):
        raise Exception('directory/file [{}] is soft link'.format(filename))
    if not os.access(file_dir, os.W_OK):
        raise Exception('cannot write to this file [{}]'.format(file_dir))

    if not os.path.exists(filename):
        _save()
        return 0

    cur_proc_uid = os.getuid()
    owner_uid = os.stat(filename).st_uid
    if owner_uid != cur_proc_uid:
        raise Exception('not owner of the file [{}]'.format(filename))
    _save()
    return 0


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
