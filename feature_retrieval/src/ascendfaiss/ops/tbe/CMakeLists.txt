SET(310_INI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/ai_core/ascend310)
SET(310P_INI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/ai_core/ascend310p)

# 310和310P共用的ini文件
SET(NEED_COPY_FILE_LIST
    ${310_INI_DIR}/cid_filter.ini
    ${310_INI_DIR}/corr_compute.ini
    ${310_INI_DIR}/distance_compute_flat.ini
    ${310_INI_DIR}/distance_compute_flat_min64.ini
    ${310_INI_DIR}/distance_filter.ini
    ${310_INI_DIR}/distance_flat_ip.ini
    ${310_INI_DIR}/distance_flat_ip_by_idx.ini
    ${310_INI_DIR}/distance_flat_ip_by_idx2.ini
    ${310_INI_DIR}/distance_flat_ip_by_idx_with_table.ini
    ${310_INI_DIR}/distance_flat_ip_maxs.ini
    ${310_INI_DIR}/distance_flat_ip_maxs_batch.ini
    ${310_INI_DIR}/distance_flat_ip_maxs_with_mask.ini
    ${310_INI_DIR}/distance_flat_ip_with_table.ini
    ${310_INI_DIR}/distance_flat_l2_mins.ini
    ${310_INI_DIR}/distance_flat_l2_mins_at.ini
    ${310_INI_DIR}/distance_flat_subcenters.ini
    ${310_INI_DIR}/distance_int8_cos_maxs.ini
    ${310_INI_DIR}/distance_int8_cos_maxs_filter.ini
    ${310_INI_DIR}/distance_int8_cos_maxs_with_mask.ini
    ${310_INI_DIR}/distance_int8_cos_maxs_with_mask_extra_score.ini
    ${310_INI_DIR}/distance_int8_l2_full_mins.ini
    ${310_INI_DIR}/distance_int8_l2_full_mins_with_mask.ini
    ${310_INI_DIR}/distance_int8_l2_mins.ini
    ${310_INI_DIR}/distance_int8_l2_mins_with_mask.ini
    ${310_INI_DIR}/distance_int8_l2_mins_wo_query_norm.ini
    ${310_INI_DIR}/distance_ivf_int8_cos.ini
    ${310_INI_DIR}/distance_ivf_int8_cos_l1.ini
    ${310_INI_DIR}/distance_ivf_int8_l2.ini
    ${310_INI_DIR}/distance_ivf_int8_l2_l1.ini
    ${310_INI_DIR}/distance_ivf_sq8_ip.ini
    ${310_INI_DIR}/distance_ivf_sq8_ip4.ini
    ${310_INI_DIR}/distance_ivf_sq8_ip8.ini
    ${310_INI_DIR}/distance_ivf_sq8_ip8_accum.ini
    ${310_INI_DIR}/distance_ivf_sq8_ipx.ini
    ${310_INI_DIR}/distance_ivf_sq8_l2.ini
    ${310_INI_DIR}/distance_marked_sq8_ip_maxs_dim64.ini
    ${310_INI_DIR}/distance_mask_generator.ini
    ${310_INI_DIR}/distance_val_mask_generator.ini
    ${310_INI_DIR}/distance_mask_generator_with_extra.ini
    ${310_INI_DIR}/distance_batch_mask_generator.ini
    ${310_INI_DIR}/distance_batch_val_mask_generator.ini
    ${310_INI_DIR}/distance_batch_mask_generator_with_extra.ini
    ${310_INI_DIR}/distance_batch_mask_generator_with_extra_and_base_mask.ini
    ${310_INI_DIR}/distance_masked_sq8_ip_maxs.ini
    ${310_INI_DIR}/distance_masked_sq8_l2_mins.ini
    ${310_INI_DIR}/distance_sq8_ip_maxs.ini
    ${310_INI_DIR}/distance_sq8_ip_maxs_dim64.ini
    ${310_INI_DIR}/distance_sq8_l2_mins.ini
    ${310_INI_DIR}/distance_table_build.ini
    ${310_INI_DIR}/int8_l2_norm.ini
    ${310_INI_DIR}/l2_norm.ini
    ${310_INI_DIR}/linear_transform.ini
    ${310_INI_DIR}/segment_max_ai.ini
)

FOREACH(FILE_ITEM ${NEED_COPY_FILE_LIST})
    CONFIGURE_FILE(${FILE_ITEM} ${310P_INI_DIR} COPYONLY)
ENDFOREACH()

MACRO(GEN_OPS_INFO_FILE op_type output_dir output_file_name_prefix)
    SET(INI_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/${op_type})
    EXECUTE_PROCESS(COMMAND  ls -1  ${INI_PATH}  OUTPUT_VARIABLE SUB_DIRS)
    STRING(REPLACE  "\n" ";"  SUB_DIRS  ${SUB_DIRS})
    FOREACH(SUB_DIR  ${SUB_DIRS})
        IF(IS_DIRECTORY  ${INI_PATH}/${SUB_DIR})
            EXECUTE_PROCESS(COMMAND  find  ${INI_PATH}/${SUB_DIR}  -name  "*.ini"  OUTPUT_VARIABLE  INI_FILES)
            IF(NOT "x${INI_FILES}" STREQUAL "x")
                STRING(REPLACE  "\n" "\t"  INI_FILES  ${INI_FILES})
            ENDIF()
            IF(NOT  "x${INI_FILES}"  STREQUAL  "x")
                SET(output_file_name  ${output_file_name_prefix}-${SUB_DIR}-ops-info.json)
                ADD_CUSTOM_COMMAND(OUTPUT ${output_file_name}
                        COMMAND  mkdir -p  ${output_dir}/${SUB_DIR}
                        COMMAND  python3  ${INI_2_JSON_PY}  ${INI_FILES}  ${output_dir}/${SUB_DIR}/${output_file_name})
                SET(OUTPUT_FILES ${OUTPUT_FILES} ${output_file_name})
            ELSE()
                MESSAGE(STATUS "ini file not exists.")
                EXECUTE_PROCESS(COMMAND mkdir -p ${output_dir})
            ENDIF()
        ENDIF()
    ENDFOREACH()
ENDMACRO(GEN_OPS_INFO_FILE)

GEN_OPS_INFO_FILE(ai_core ${AIC_OP_INFO_CFG_OUT_DIR} aic)
ADD_CUSTOM_TARGET(ai_core_ops_config_json  ALL DEPENDS  ${OUTPUT_FILES})