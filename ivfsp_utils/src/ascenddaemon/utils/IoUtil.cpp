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


#include "ascenddaemon/utils/IoUtil.h"

namespace ascendSearch {

namespace {
const auto MAX_DATAFILE_SIZE = 56L * 1024L * 1024L * 1024L;  // needs adjustment?
}

void initializeFileDescription(int &fd, const std::string &fname)
{
    std::string realPath = fname;
    // 系统调用方式
    struct stat st;
    // 路径长度校验; 255 is Linux Limit
    if (fname.length() > 255) { ASCEND_THROW_MSG("Path too long!\n"); }
    // 路径编码校验
    for (uint32_t j = 0; j < fname.length(); j++) {
        if (!isValidCode(fname[j]) && !isInWhiteList(fname[j])) {
            ASCEND_THROW_MSG("Invalid Path!\n");
        }
    }
    // 判断是否为绝对路径，如果不是，获取绝对路径，最后存在realPath中
    if (fname[0] != '/' && !(fname.length() > 1 && fname[0] == '~' && fname[1] == '/')) {
        char buffer [PATH_MAX];
        if (getcwd(buffer, sizeof(buffer)) != nullptr) {
            std::string absPath = buffer;
            realPath = absPath + "/" + fname;
        } else { ASCEND_THROW_MSG("Failed to retrieve absolute path!\n"); }
    }

    if (lstat(realPath.c_str(), &st) == 0) { // 文件路径已经存在
        fd = open(realPath.c_str(), O_NOFOLLOW | O_WRONLY);
        ASCEND_THROW_IF_NOT_MSG(fd >= 0, // 文件未打开成功，无需关闭
            "fname or one of its parent directory is a softlink, or error encoutered during opening fname.\n");
        // 父目录进行校验，如果校验失败，不会移除现有的路径
        parentDirCheckAndRemove(realPath, fd, false);

        if (fstat(fd, &st) != 0) {
            closeFileDescription(fd);
            ASCEND_THROW_MSG("Cannot get stats from file description.\n");
        }
        if (!S_ISREG(st.st_mode)) { // 检查普通文件属性
            closeFileDescription(fd);
            ASCEND_THROW_MSG("File is not a regular file.\n");
        }
        if (st.st_size > MAX_DATAFILE_SIZE) { // 校验文件大小
            closeFileDescription(fd);
            ASCEND_THROW_MSG("File exceeds maximum size.\n");
        }
        if (st.st_uid != geteuid()) { // 校验文件属主
            closeFileDescription(fd);
            ASCEND_THROW_MSG("Not File Owner.\n");
        }
    } else { // 文件路径不存在
        // 以期待创建一个新文件的方式打开文件，并设置640权限（如果任何父路径为软连接，不会打开或创建）
        fd = open(realPath.c_str(), O_NOFOLLOW | O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
        ASCEND_THROW_IF_NOT_MSG(fd >= 0, // 文件未打开成功，无需关闭
            "fname or one of its parent directory is a softlink, or error encoutered during opening fname.\n");
        // 父目录进行校验，如果校验失败则移除已经被创建的路径
        parentDirCheckAndRemove(realPath, fd, true);
    }
}

bool checkLinkRec(const std::string &realPathFunc)
{
    // 逐级递归查看是否有软连接
    struct stat saveStatTmp;
    std::string tmpPath = realPathFunc;
    int tmpLast = static_cast<int>(tmpPath.rfind('/'));
    while (tmpLast != -1) {
        if ((lstat(tmpPath.c_str(), &saveStatTmp) != 0) || S_ISLNK(saveStatTmp.st_mode)) {
            return true;
        }
        tmpPath = tmpPath.substr(0, tmpLast);
        tmpLast = static_cast<int>(tmpPath.rfind('/'));
    }
    return false;
}

void parentDirCheckAndRemove(const std::string &realPath, int &fd, bool removeDir)
{
    struct stat stTmp;
    int lastPos = static_cast<int>(realPath.rfind('/'));
    std::string realPathParent = realPath.substr(0, lastPos);
    if ((lstat(realPathParent.c_str(), &stTmp) == 0) && S_ISDIR(stTmp.st_mode)) { // 如果父目录存在
        if (checkLinkRec(realPathParent)) { // 校验父目录软连接
            closeFileDescription(fd);
            if (removeDir) { // 如果文件已经通过文件句柄被创造，移除
                remove(realPath.c_str());
            }
            ASCEND_THROW_MSG("parent dir has symbolic link or does not exist!");
        }
    } else {  // 如果父目录不存在
        closeFileDescription(fd);
        if (removeDir) {
            remove(realPath.c_str());
        }
        ASCEND_THROW_MSG("parent directory not exist");
    }
}

FSPIOWriter::FSPIOWriter(const std::string &fname) : name(fname)
{
    initializeFileDescription(fd, name);
}

FSPIOWriter::~FSPIOWriter()
{
    closeFileDescription(fd);
}

FSPIOReader::FSPIOReader(const std::string &fname) : name(fname)
{
    fd = open(name.c_str(), O_NOFOLLOW | O_RDONLY);
    ASCEND_THROW_IF_NOT_MSG(fd >= 0, // 文件未打开成功，无需关闭
        "fname or one of its parent directory is a softlink, or error encoutered during opening fname.\n");
}

size_t FSPIOReader::GetFileSize() const
{
    struct stat statbuf;
    ASCEND_THROW_IF_NOT_FMT(fstat(fd, &statbuf) != -1, "Failed to retrieve file information for %s.\n", name.c_str());
    return statbuf.st_size;
}

FSPIOReader::~FSPIOReader()
{
    closeFileDescription(fd);
}

}