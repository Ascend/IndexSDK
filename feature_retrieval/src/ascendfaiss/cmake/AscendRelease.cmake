# Released headers
FILE(GLOB  ASCEND_SRC_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendCloner.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendClonerOptions.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndex.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVF.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFPQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFRabitQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFSQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/ivfsp/AscendIndexIVFSP.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8Flat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexSQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendMultiIndexSearch.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendNNInference.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/include/index/AscendIndexBinaryFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/include/index/AscendIndexCluster.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/include/index/AscendIndexILFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/include/index/AscendIndexTS.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/include/AscendIndexGreat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/include/AscendIndexVStar.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/mixsearch/include/AscendIndexMixSearchParams.h
)

FILE(GLOB ASCEND_SRC_UTILS_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/utils/Version.h)

FILE(GLOB ASCEND_SRC_CUSTOM_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSQT.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/IReduction.h
)

INSTALL(FILES    ${ASCEND_SRC_HEADERS}         DESTINATION  include/faiss/ascend)
INSTALL(FILES    ${ASCEND_SRC_UTILS_HEADERS}   DESTINATION  include/faiss/ascend/utils)
INSTALL(FILES    ${ASCEND_SRC_CUSTOM_HEADERS}  DESTINATION  include/faiss/ascend/custom)

# Ascenddevice headers
FILE(GLOB DEVICE_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl_device/IndexIL.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl_device/IndexILFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../common/ErrorCode.h
)
INSTALL(FILES   ${DEVICE_HEADERS}  DESTINATION  device/include)