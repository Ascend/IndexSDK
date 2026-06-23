# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# ruff: noqa: F401
# pylint: disable=E0606
import sys
import inspect
import logging
import numpy as np

import faiss

if faiss.__version__ == "1.10.0":
    from faiss.class_wrappers import handle_Index
    from faiss.class_wrappers import replace_method

if faiss.__version__ == "1.7.4":
    from faiss.class_wrappers import handle_Index
    from faiss.class_wrappers import replace_method

if faiss.__version__ == "1.7.1":
    from faiss import handle_Index
    from faiss import replace_method

logger = logging.getLogger(__name__)
try:
    from .swig_ascendfaiss import (
        FloatVector,
        DoubleVector,
        ByteVector,
        CharVector,
        UInt64Vector,
        LongVector,
        FaissIdxVector,
    )
    from .swig_ascendfaiss import IntVector, AscendIndex
    from .swig_ascendfaiss import Index, index_ascend_to_cpu
    from .swig_ascendfaiss import AscendIndexInt8, AscendIndexFlatL2
    from .swig_ascendfaiss import AscendIndexFlatConfig, index_cpu_to_ascend
    from .swig_ascendfaiss import AscendIndexInt8FlatConfig, AscendIndexFlat
    from .swig_ascendfaiss import swig_ptr, AscendIndexInt8Flat
    from .swig_ascendfaiss import IDSelectorBatch, IDSelector
    from .swig_ascendfaiss import IDSelectorRange, index_int8_ascend_to_cpu
    from .swig_ascendfaiss import index_int8_cpu_to_ascend
except ImportError as e:
    logger.error("Loading ascendfaiss error")
    logger.error(e)


def handle_index_int8(cur_class):
    def replacement_add(self, x):
        n, d = x.shape
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_assign(self, x, k):
        n, d = x.shape
        distances = np.empty((n, k), dtype=np.float32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x), k, swig_ptr(distances), swig_ptr(labels))
        return labels

    def replacement_train(self, x):
        n, d = x.shape
        self.train_c(n, swig_ptr(x))

    def replacement_update_centroids(self, x):
        n, d = x.shape
        self.updateCentroids_c(n, swig_ptr(x))

    def replacement_search(self, x, k):
        n, d = x.shape

        distances = np.empty((n, k), dtype=np.float32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x), k, swig_ptr(distances), swig_ptr(labels))
        return distances, labels

    def replacement_remove_ids(self, x):
        if isinstance(x, IDSelector):
            return self.remove_ids_c(x)
        sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    replace_method(cur_class, 'add', replacement_add)
    replace_method(cur_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(cur_class, 'assign', replacement_assign)
    replace_method(cur_class, 'train', replacement_train)
    replace_method(cur_class, 'search', replacement_search)
    replace_method(cur_class, 'updateCentroids', replacement_update_centroids)
    replace_method(cur_class, 'remove_ids', replacement_remove_ids)


this_module = sys.modules[__name__]

for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    if inspect.isclass(obj):
        target_class = obj
        if issubclass(target_class, Index):
            handle_Index(target_class)
        if issubclass(target_class, AscendIndexInt8):
            handle_index_int8(target_class)


def replace_destructor(the_class):
    original_del = the_class.__del__

    def replacement_del(self):
        if original_del is not None:
            original_del(self)

    the_class.__del__ = replacement_del


def index_cpu_to_ascend_py(devices, index, co=None):
    vdev = IntVector()
    for i in devices:
        vdev.push_back(i)
    index = index_cpu_to_ascend(vdev, index, co)
    return index
