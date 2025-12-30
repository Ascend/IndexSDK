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


#ifndef ASCEND_COMMON_UTILS_H
#define ASCEND_COMMON_UTILS_H

#include <climits>
#include <pwd.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "acl/acl.h"

#define CREATE_UNIQUE_PTR(type, args...) \
    std::unique_ptr<type>(new type(args))

namespace ascendSearch {
    class CommonUtils {
    public:
        static std::string RealPath(const std::string &inPath)
        {
            std::string result = inPath;

            // 1. replace ~/ with /home/user/
            if (inPath.size() >= 2 && inPath[0] == '~' && inPath[1] == '/') { // >=2 means inPath startswith '~/'
                struct passwd *pw = getpwuid(getuid());
                if (pw == nullptr || pw->pw_dir == nullptr) {
                    return std::string();
                }
                std::string homedir(pw->pw_dir);
                homedir.append(inPath.substr(1));
                result = std::move(homedir);
            }

            // 2. realpath
            if (result.size() > PATH_MAX) {
                return std::string();
            }
            char realPath[PATH_MAX] = {0};
            char *ptr = realpath(result.c_str(), realPath);
            if (ptr == nullptr) {
                return std::string();
            }
            result = std::string(realPath);

            return result;
        }

        static bool CheckPathValid(const std::string &path)
        {
            if (path.find("/home/") != 0 && path.find("/root/") != 0) {
                return false;
            }
            if (access(path.c_str(), R_OK) != EOK) {
                return false;
            }
            return true;
        }

        static void AclInputBufferDelete(std::vector<const aclDataBuffer *> *distOpInput)
        {
            for (auto &item : *distOpInput) {
                aclDestroyDataBuffer(item);
            }
            delete distOpInput;
        }

        static void AclOutputBufferDelete(std::vector<aclDataBuffer *> *distOpOutput)
        {
            for (auto &item : *distOpOutput) {
                aclDestroyDataBuffer(item);
            }
            delete distOpOutput;
        }
    };
}

#endif // ASCEND_COMMON_UTILS_H
