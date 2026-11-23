# -------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All Rights Reserved.
#
# IndexSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

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
