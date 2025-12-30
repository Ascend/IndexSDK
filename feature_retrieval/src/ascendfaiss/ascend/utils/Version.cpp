/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */


#include <fstream>
#include <iostream>
#include <iomanip>
#include <climits>

#include "acl/acl.h"
#include "common/utils/LogUtils.h"

#include "ascend/utils/Version.h"

namespace faiss {
namespace ascend {
namespace {
inline bool GetAbsPath(const std::string &path, std::string &resolvedPath)
{
    char tmpPath[PATH_MAX + 1] = { 0x00 };
    if ((path.size() > PATH_MAX) || (realpath(path.c_str(), tmpPath) == nullptr)) {
        APP_LOG_ERROR("Failed to get canonicalize path.\n");
        return false;
    }
    resolvedPath = std::string(tmpPath);
    return true;
}
}

std::string GetVersionInfo()
{
    std::string versionInfo = "/mxIndex/version.info";
    std::string mxIndexHome = "/usr/local";
    const char* mxIndexEnv = std::getenv("MX_INDEX_HOME");
    if (mxIndexEnv) {
        mxIndexHome = mxIndexEnv;
    }

    std::string filePath = mxIndexHome + std::string(versionInfo);
    std::string resolvedPath = "";
    bool ret = GetAbsPath(filePath, resolvedPath);
    if (!ret) {
        printf("Invalid version.info filePath.\n");
        return "";
    }
    std::string checkInfo = "/version.info";
    if (resolvedPath.length() < versionInfo.length() ||
        resolvedPath.substr(resolvedPath.length() - checkInfo.length()) != checkInfo) {
        printf("Invalid version.info resolvedPath.\n");
        return "";
    }

    std::ifstream file(resolvedPath);
    if (!file.is_open()) {
        printf("Invalid version.info resolvedPath.\n");
        return "";
    }

    char version[LINE_MAX];
    file.getline(version, LINE_MAX);
    if (file.rdstate() == std::ios_base::failbit) {
        printf("get version.info failed.\n");
        return "";
    }

    return version;
}
} // ascend
} // faiss