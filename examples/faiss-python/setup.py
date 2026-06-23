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
from __future__ import print_function
import os
import shutil
import logging
from setuptools import setup


DESCRIPTION = """FAISS is a library for efficient similarity search and clustering of dense
vectors on Ascend."""
check_fpath = os.path.join("_swig_ascendfaiss.so")
if not os.path.exists(check_fpath):
    logging.warning("Could not find %s", check_fpath)


# make the faiss python package dir
shutil.rmtree("ascendfaiss", ignore_errors=True)
os.mkdir("ascendfaiss")

shutil.copyfile("ascendfaiss.py", "ascendfaiss/__init__.py")
shutil.copyfile("swig_ascendfaiss.py", "ascendfaiss/swig_ascendfaiss.py")
shutil.copyfile("_swig_ascendfaiss.so", "ascendfaiss/_swig_ascendfaiss.so")

setup(
    name='ascendfaiss',
    version='1.0.0',
    description='A library for efficient similarity search and clustering of dense vectors',
    long_description=DESCRIPTION,
    author='Huawei',
    author_email='',
    license='MIT',
    keywords='search nearest neighbors',
    install_requires=['numpy'],
    packages=['ascendfaiss'],
    include_package_data=True,
    package_data={
        'ascendfaiss': ['*.so'],
    },
    zip_safe=False,
)
