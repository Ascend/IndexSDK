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

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

class VstarBaseAddMatMul {
public:
    __aicore__ inline VstarBaseAddMatMul(){};
    TPipe pipe;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                GM_ADDR usrWorkspace, const VstarBaseAddMatMulTilingData *__restrict tiling);
    __aicore__ inline void Process();

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
    MatmulType<TPosition::GM, CubeFormat::ND, half, true>,
    MatmulType<TPosition::GM, CubeFormat::ND, float>> matmulObj;

    TCubeTiling cube_tiling;

    GlobalTensor<float > aGlobal;
    GlobalTensor<float > bGlobal;
    GlobalTensor<float > cGlobal;
    GlobalTensor<half > a16Global;
    GlobalTensor<half > b16Global;
    GlobalTensor<half > a16WorkspaceGlobal;
    GlobalTensor<half > b16WorkspaceGlobal;
    LocalTensor<float> baseFP32Local;
    LocalTensor<half> baseFP16UBLocal;
    LocalTensor<float> coodesBookFP32Local;
    LocalTensor<half> coodesBookFP16UBLocal;

    TQue<QuePosition::VECIN, 1> queCastVECIN;
    TQue<QuePosition::VECOUT, 1> queCastVECOUT;

    uint32_t core_id;
    uint32_t nb;
    uint32_t dim;
    uint32_t nList;
    uint32_t subDim;
    uint64_t MMFormatUb1;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t each_M;
    uint32_t each_N;
    uint32_t loop_M;
    uint32_t loop_N;
    uint32_t last_M;
    uint32_t last_N;
    uint32_t MTaskCore;
    uint32_t NTaskCore;

    // split data by tiling info from MultiCoreMatmulTiling
    uint64_t mm_a_offset;
    uint64_t mm_b_offset;
    uint64_t mm_c_offset;
    uint64_t mm_a_size;
    uint64_t mm_b_size;
};

__aicore__ inline void VstarBaseAddMatMul::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR usrWorkspace,
                                                const VstarBaseAddMatMulTilingData *__restrict tiling)
{
    core_id = GetBlockIdx();
    nb = tiling->nb;
    dim = tiling->dim;
    nList = tiling->nList;
    subDim = tiling->subDim;
    aicNum = tiling->aicNum;
    aivNum = tiling->aivNum;
    MMFormatUb1 = tiling->MMFormatUb1;

    if (core_id < (nb % aivNum)) {
        each_M = tiling->each_M_extra;
        loop_M = tiling->loop_M_extra;
        last_M = tiling->last_M_extra;
        MTaskCore = tiling->MTaskCore_extra;
    } else {
        each_M = tiling->each_M_regular;
        loop_M = tiling->loop_M_regular;
        last_M = tiling->last_M_regular;
        MTaskCore = tiling->MTaskCore_regular;
    }

    if (core_id < (nList * subDim) % aivNum) {
        each_N = tiling->each_N_extra;
        loop_N = tiling->loop_N_extra;
        last_N = tiling->last_N_extra;
        NTaskCore = tiling->NTaskCore_extra;
    } else {
        each_N = tiling->each_N_regular;
        loop_N = tiling->loop_N_regular;
        last_N = tiling->last_N_regular;
        NTaskCore = tiling->NTaskCore_regular;
    }

    cube_tiling = tiling->cube_tiling;
    auto n_num = Ceil(cube_tiling.N, cube_tiling.singleCoreN);
    auto m_num = Ceil(cube_tiling.M, cube_tiling.singleCoreM);
    auto core_n_id = core_id % n_num;
    auto core_m_id = core_id / n_num;
    mm_a_offset = core_m_id * cube_tiling.singleCoreM * cube_tiling.Ka;
    mm_b_offset = core_n_id * cube_tiling.singleCoreN * cube_tiling.Ka;
    mm_c_offset = core_m_id * cube_tiling.singleCoreM * cube_tiling.N +
        core_n_id * cube_tiling.singleCoreN;
    mm_a_size = (core_m_id != (m_num - 1)) ?
        cube_tiling.singleCoreM : (cube_tiling.M - core_m_id * (cube_tiling.singleCoreM));
    mm_b_size = (core_n_id != (n_num - 1)) ?
        cube_tiling.singleCoreN : (cube_tiling.N - core_n_id * (cube_tiling.singleCoreN));

    uint32_t max_item = max(each_M, each_N);
    pipe.InitBuffer(queCastVECIN, 1, max_item * dim * sizeof(float));
    pipe.InitBuffer(queCastVECOUT, 1, max_item * dim * sizeof(half));

    if (core_id < (nb % aivNum)) {
        aGlobal.SetGlobalBuffer((__gm__ float *)a + core_id * tiling->MTaskCore_extra * dim);
        a16Global.SetGlobalBuffer((__gm__ half *)usrWorkspace + core_id * tiling->MTaskCore_extra * dim);
    } else {
        uint32_t regularId = core_id - nb % aivNum;
        aGlobal.SetGlobalBuffer((__gm__ float *)a + (nb % aivNum) * tiling->MTaskCore_extra * dim +
            regularId * tiling->MTaskCore_regular * dim);
        a16Global.SetGlobalBuffer((__gm__ half *)usrWorkspace + (nb % aivNum) * tiling->MTaskCore_extra * dim +
            regularId * tiling->MTaskCore_regular * dim);
    }

    if (core_id < (nList * subDim) % aivNum) {
        bGlobal.SetGlobalBuffer((__gm__ float *)b + core_id * tiling->NTaskCore_extra * dim);
        b16Global.SetGlobalBuffer((__gm__ half *)usrWorkspace + nb * dim + core_id * tiling->NTaskCore_extra * dim);
    } else {
        uint32_t regularId = core_id - (nList * subDim) % aivNum;
        bGlobal.SetGlobalBuffer((__gm__ float *)b + ((nList * subDim) % aivNum) * tiling->NTaskCore_extra * dim +
                                regularId * tiling->NTaskCore_regular * dim);
        b16Global.SetGlobalBuffer((__gm__ half *)usrWorkspace + nb * dim +
            ((nList * subDim) % aivNum) * tiling->NTaskCore_extra * dim + regularId * tiling->NTaskCore_regular * dim);
    }

    cGlobal.SetGlobalBuffer((__gm__ float *)c);
    a16WorkspaceGlobal.SetGlobalBuffer((__gm__ half *)usrWorkspace);
    b16WorkspaceGlobal.SetGlobalBuffer((__gm__ half *)usrWorkspace + nb*dim);
}

__aicore__ inline void VstarBaseAddMatMul::Process()
{
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj);

    for (int iM = 0; iM<loop_M; iM++) {
        uint32_t startM = iM * each_M;
        uint32_t endM = min(startM + each_M, MTaskCore);
        baseFP32Local = queCastVECIN.AllocTensor<float>();
        baseFP16UBLocal = queCastVECOUT.AllocTensor<half>();
        uint32_t MLen = endM - startM;
        pipe_barrier(PIPE_ALL);
        DataCopy(baseFP32Local, aGlobal[iM * each_M * dim], MLen * dim);
        pipe_barrier(PIPE_ALL);
        Cast(baseFP16UBLocal, baseFP32Local, RoundMode::CAST_NONE, MLen * dim);
        pipe_barrier(PIPE_ALL);
        DataCopy(a16Global[iM * each_M * dim], baseFP16UBLocal, MLen * dim);
        pipe_barrier(PIPE_ALL);
        queCastVECIN.FreeTensor(baseFP32Local);
        queCastVECOUT.FreeTensor(baseFP16UBLocal);
        pipe_barrier(PIPE_ALL);
    }

    for (int iN = 0; iN<loop_N; iN++) {
        uint32_t startN = iN * each_N;
        uint32_t endN = min(startN+each_N, NTaskCore);
        coodesBookFP32Local  = queCastVECIN.AllocTensor<float>();
        coodesBookFP16UBLocal  = queCastVECOUT.AllocTensor<half>();
        uint32_t NLen = endN - startN;
        pipe_barrier(PIPE_ALL);
        DataCopy(coodesBookFP32Local, bGlobal[iN * each_N * dim], NLen * dim);
        pipe_barrier(PIPE_ALL);
        Cast(coodesBookFP16UBLocal, coodesBookFP32Local, RoundMode::CAST_NONE, NLen * dim);
        pipe_barrier(PIPE_ALL);
        DataCopy(b16Global[iN * each_N * dim], coodesBookFP16UBLocal, NLen * dim);
        pipe_barrier(PIPE_ALL);
        queCastVECIN.FreeTensor(coodesBookFP32Local);
        queCastVECOUT.FreeTensor(coodesBookFP16UBLocal);
        pipe_barrier(PIPE_ALL);
    }
    SyncAll();

    TBuf<> tmpMMFormatUb1;
    LocalTensor<uint8_t> mmformatUb1;
    pipe.InitBuffer(tmpMMFormatUb1,  MMFormatUb1);
    mmformatUb1 = tmpMMFormatUb1.Get<uint8_t>(MMFormatUb1);

    matmulObj.Init(&cube_tiling);
    matmulObj.SetLocalWorkspace(mmformatUb1);
    matmulObj.SetTensorA(a16WorkspaceGlobal[mm_a_offset]);
    matmulObj.SetTensorB(b16WorkspaceGlobal[mm_b_offset], true);
    matmulObj.SetTail(mm_a_size, mm_b_size, cube_tiling.Ka);
    matmulObj.IterateAll(cGlobal[mm_c_offset]);
    matmulObj.End();
    SyncAll();
}

extern "C" __global__ __aicore__ void vstar_base_add_mat_mul(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);

    GET_TILING_DATA(tiling_data, tiling);
    const VstarBaseAddMatMulTilingData *__restrict tiling_device = &tiling_data;
    VstarBaseAddMatMul op;
    op.Init(a, b, c, usrWorkspace, tiling_device);
    op.Process();
}