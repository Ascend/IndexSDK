#!/bin/bash
# -------------------------------------------------------------------------
# This file is part of the IndexSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

# Bootstrap the development environment - add extra files needed to run configure. 
# Note autoreconf should do what this file achieves, but it has a bug when working with automake!
# The latest config.guess and config.sub should be copied into Tools/config.
# This script will ensure the latest is copied from your autotool installation.

${ACLOCAL-aclocal}
${AUTOMAKE-automake} --add-missing --copy --force-missing
${AUTOCONF-autoconf}
