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


#ifndef OCK_HCPS_PIER_FUN_GUARD_UTILS_H
#define OCK_HCPS_PIER_FUN_GUARD_UTILS_H
namespace ock {
namespace utils {

/*
@brief 样例： utils::FunGuard<std::function<int32_t()>, std::function<int32_t()>> ...
*/
template <typename AllocFunT = int (*)(), typename FreeFunT = int (*)()>
class FunGuard {
public:
    explicit FunGuard(AllocFunT allocFun, FreeFunT freeFunc) : releaseControl(false), freeFun(freeFunc)
    {
        allocFun();
    }
    ~FunGuard()
    {
        if (!releaseControl) {
            freeFun();
        }
    }
    void ReleaseControl(void)
    {
        releaseControl = true;
        freeFun();
    }

private:
    bool releaseControl;
    FreeFunT freeFun;
};

/*
@brief 样例： utils::FunGuardWithRet<std::function<int32_t()>, std::function<int32_t()>, int32_t> ...
*/
template <typename AllocFunT, typename FreeFunT, typename AllocFunRetT = typename AllocFunT::result_type>
class FunGuardWithRet {
public:
    explicit FunGuardWithRet(AllocFunT allocFun, FreeFunT freeFunc) : releaseControl(false), freeFun(freeFunc)
    {
        allocRet = allocFun();
    }
    ~FunGuardWithRet()
    {
        if (!releaseControl) {
            freeFun();
        }
    }
    const AllocFunRetT &AllocRet(void) const
    {
        return allocRet;
    }
    AllocFunRetT &AllocRet(void)
    {
        return allocRet;
    }
    void ReleaseControl(void)
    {
        releaseControl = true;
        freeFun();
    }

private:
    bool releaseControl;
    FreeFunT freeFun;
    AllocFunRetT allocRet{};
};
}  // namespace utils
}  // namespace ock
#endif