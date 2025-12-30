# Released headers
FILE(GLOB  ASCEND_SRC_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendCloner.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendClonerOptions.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndex.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVF.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFPQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexIVFSQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8Flat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8IVF.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexInt8IVFFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexPreTransform.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendIndexSQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendMultiIndexSearch.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendNNInference.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendVectorTransform.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/include/index/AscendIndexBinaryFlat.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/include/index/AscendIndexTSV2.h
)

FILE(GLOB ASCEND_SRC_UTILS_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/utils/Version.h)

FILE(GLOB ASCEND_SRC_CUSTOM_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSQC.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSQFuzzy.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSQT.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFast.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexTS.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/IReduction.h
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSPSQ.h
)

FILE(GLOB ASCEND_SRC_FV_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/fv/AscendIndexFVIVFPQ.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascend/fv/AscendIndexFVPQ.h)

INSTALL(FILES    ${ASCEND_SRC_HEADERS}         DESTINATION  include/ascendsearch/ascend)
INSTALL(FILES    ${ASCEND_SRC_UTILS_HEADERS}   DESTINATION  include/ascendsearch/ascend/utils)
INSTALL(FILES    ${ASCEND_SRC_CUSTOM_HEADERS}  DESTINATION  include/ascendsearch/ascend/custom)
INSTALL(FILES    ${ASCEND_SRC_FV_HEADERS}      DESTINATION  include/ascendsearch/ascend/fv)

# Ascenddevice headers
FILE(GLOB DEVICE_HEADERS
     ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl_device/IndexIL.h
     ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl_device/IndexILFlat.h
)
INSTALL(FILES   ${DEVICE_HEADERS}  DESTINATION  device/include)