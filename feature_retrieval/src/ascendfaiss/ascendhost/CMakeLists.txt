CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(ascendhost)

INCLUDE(${CMAKE_CURRENT_LIST_DIR}/../cmake/utils.cmake)

IF (NOT DEFINED ASCEND_HOME)
    SET(ASCEND_HOME  /usr/local/Ascend  CACHE  STRING "")
ENDIF()
IF (NOT DEFINED FAISS_HOME)
    SET(FAISS_HOME  /usr/local/  CACHE  STRING "")
ENDIF()
IF (NOT DEFINED ASCEND_TOOLKIT_PATH)
    SET(ASCEND_TOOLKIT_PATH  /usr/local/Ascend/ascend-toolkit/latest)
ENDIF()

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_COMPILER   gcc)
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_SKIP_BUILD_RPATH  TRUE)

IF(BUILD_ASCEND_SEARCH)
    ADD_COMPILE_OPTIONS(
        -fPIC  -fstack-protector-all  -Wall  -fno-strict-aliasing  -Wreturn-type
        -D_FORTIFY_SOURCE=2  -frename-registers  -fpeel-loops  -fopenmp  -O3
        -DHOSTCPU -DBUILD_IVFSP -DUSE_ACL_INTERFACE_V2 -Wfloat-equal -fno-common -Wextra)
ELSE()
    ADD_COMPILE_OPTIONS(
        -fPIC  -fstack-protector-all  -Wall  -fno-strict-aliasing  -Wreturn-type
        -D_FORTIFY_SOURCE=2  -frename-registers  -fpeel-loops  -fopenmp  -O3
        -DHOSTCPU  -DUSE_ACL_INTERFACE_V2 -Wfloat-equal -fno-common -Wextra)
ENDIF()

IF(BUILD_OCK)
    ADD_COMPILE_OPTIONS(-DBUILD_OCK)
ENDIF()

ADD_LINK_OPTIONS(
    -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack -s)

SET(TARGET_LIBRARY         ascendfaiss_host)
SET(TARGET_LIBRARY_RENAME  ascendfaiss)

IF(HITEST)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE hitestwrapper)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK hitestwrapper)
ENDIF()

IF(BUILD_ASCEND_SEARCH)
    set(ASCEND_SEARCH_HOME_LIB_DIR "${ASCEND_SEARCH_HOME}/lib")
    set(IVFSP_HOME_INC_DIR "${ASCEND_SEARCH_HOME}/IVFSP/include")
    set(IVFSP_CPP_FILES
        ${CMAKE_CURRENT_LIST_DIR}/../ascend/ivfsp/AscendIndexIVFSP.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../ascend/ivfsp/AscendIndexIVFSPImpl.cpp
    )
    set(MIX_SEARCH_INC_DIR
        "${ASCEND_SEARCH_HOME}/mix-index/include/Great_include"
        "${ASCEND_SEARCH_HOME}/mix-index/include/Vstar_include")
    set(MIX_SEARCH_SRC 
        ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/great/IndexGreatAdaptor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/vstar/IndexVStarAdapter.cpp)
ELSE()
    set(ASCEND_SEARCH_HOME_LIB_DIR "")
    set(IVFSP_HOME_INC_DIR "")
    set(IVFSP_CPP_FILES "")
    set(MIX_SEARCH_SRC 
        ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/great/IndexGreatStub.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/vstar/IndexVStarAdapterStub.cpp)
ENDIF()

IF(BUILD_OCK)
    SET(OCK_FILEDS
        ${CMAKE_CURRENT_LIST_DIR}/../common/hmm/HmmAdaptor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../common/hmm/AscendHMOImpl.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/TSInt8FlatHPPCos.cpp
    )
    set(VSA_HOME_INC_DIR "${OCK_HOME}/vsa/include")
    set(VSA_HOME_LIB_DIR "${OCK_HOME}/vsa/lib")
    set(HCPS_HOME_INC_DIR "${OCK_HOME}/hcps/include")
    set(HCPS_HOME_LIB_DIR "${OCK_HOME}/hcps/lib")
    set(HMM_HOME_INC_DIR "${OCK_HOME}/memory_bridge/include")
    set(HMM_HOME_LIB_DIR "${OCK_HOME}/memory_bridge/lib")
ELSE()
    SET(OCK_FILEDS ${CMAKE_CURRENT_LIST_DIR}/../common/hmm/HmmIntfStub.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/../common/hmm/TSInt8FlatHPPCosFactoryStub.cpp
    )
ENDIF()

# CMAKE_SOURCE_DIR: top level of the source tree
SET(ASCENDFAISS_HOST_SRC
    ${IVFSP_CPP_FILES}
    ${OCK_FILEDS}
    ${MIX_SEARCH_SRC}
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendCloner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendClonerOptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndex.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8Flat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVF.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFPQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFSQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexSQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendMultiIndexSearch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendNNInference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendClustering.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexFlatAT.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexFlatATInt8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSQT.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexFlatImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexInt8Impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexInt8FlatImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexIVFImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexIVFFlatImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexIVFPQImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexIVFSQImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendIndexSQImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/impl/AscendNNInferenceImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/impl/AscendIndexFlatATImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/impl/AscendIndexFlatATInt8Impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/impl/AscendIndexIVFSQFuzzyImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/impl/AscendIndexIVFSQCImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/impl/AscendIndexIVFSQTImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/IReduction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/NNReduction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/PcarReduction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/utils/fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/utils/MergeResUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/utils/Version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/StandardAscendResources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/ThreadSafeStandardAscendResources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/AuxIndexStructures.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/Index.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexInt8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexInt8Flat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexIVF.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexIVFFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexIVFPQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexIVFSQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/Int8L2Norm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl/IndexSQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/AscendMemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/AscendOpDesc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/AscendOperator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/AscendStackMemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/AscendUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/AscendRWLock.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/DeviceMemMng.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/DistComputeOpsManager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/ThreadSafeDistComputeOpsManager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/MemorySpace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/Random.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/TopkOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/ModelInference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils/ModelExecuter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/HeteroBlockGroupMgrImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/AscendClusteringImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/AscendIndexBinaryFlatImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/TSBase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/TSBinaryFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/TSInt8FlatCos.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/TSInt8FlatL2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/TSFlatIP.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/AscendIndexClusterImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/AscendIndexILFlatImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/AscendIndexBinaryFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/AscendIndexCluster.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/AscendIndexILFlat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/AscendIndexTS.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexFlatIPAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexFlatL2Aicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexInt8FlatL2Aicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexInt8FlatCosAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexIVFSQIPAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexIVFSQL2Aicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexSQIPAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/IndexSQL2Aicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexFlatAT.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexFlatATAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexFlatATInt8Aicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexFlatATSubAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexIVFSQCIPAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexIVFSQTIPAicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexInt8FlatApproxL2Aicpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../common/threadpool/AscendThreadPool.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/great/AscendIndexGreat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/vstar/AscendIndexVStar.cpp
)

INCLUDE_DIRECTORIES(ASCENDFAISS_HOST_INC_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../
    ${CMAKE_CURRENT_LIST_DIR}/../ascend
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/utils
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/include
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/utils
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include/impl
    ${CMAKE_CURRENT_LIST_DIR}/include/index
    ${FAISS_HOME}/include
    ${ASCEND_TOOLKIT_PATH}/include
    ${ASCEND_HOME}/driver/include/dvpp
    ${IVFSP_HOME_INC_DIR}
    ${VSA_HOME_INC_DIR}
    ${HCPS_HOME_INC_DIR}
    ${HMM_HOME_INC_DIR}
    ${MIX_SEARCH_INC_DIR}
)

LINK_DIRECTORIES(
    ${FAISS_HOME}/lib
    ${ASCEND_TOOLKIT_PATH}/lib64
    ${ASCEND_SEARCH_HOME_LIB_DIR}
    ${OCK_HOME_LIB_DIR}
    ${VSA_HOME_LIB_DIR}
    ${HCPS_HOME_LIB_DIR}
    ${HMM_HOME_LIB_DIR}
)

ADD_LIBRARY(${TARGET_LIBRARY}  SHARED  ${ASCENDFAISS_HOST_SRC})
GET_FILENAME_COMPONENT(ASCENDFAISS_DIR   ${CMAKE_CURRENT_LIST_DIR}/..  ABSOLUTE)
redefine_file_macro(${TARGET_LIBRARY}  ${ASCENDFAISS_DIR})
IF(BUILD_ASCEND_SEARCH)
    TARGET_LINK_LIBRARIES(${TARGET_LIBRARY}  faiss  ascendcl  c_sec  pthread ascendsearch acl_op_compiler)
ELSE()
    TARGET_LINK_LIBRARIES(${TARGET_LIBRARY}  faiss ascendcl  c_sec  pthread acl_op_compiler)
ENDIF()
IF(BUILD_OCK)
    TARGET_LINK_LIBRARIES(${TARGET_LIBRARY} ock_vsa_hpp ock_hcps_pier ock_hmm)
ENDIF()

SET_TARGET_PROPERTIES(${TARGET_LIBRARY}  PROPERTIES  OUTPUT_NAME  ${TARGET_LIBRARY_RENAME})
INSTALL(TARGETS  ${TARGET_LIBRARY}  DESTINATION  host/lib)
