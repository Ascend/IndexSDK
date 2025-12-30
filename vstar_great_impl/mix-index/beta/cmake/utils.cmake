
FUNCTION(redefine_file_macro target_name unwanted_prefix)
    #获取当前目标的所有源文件
    GET_TARGET_PROPERTY(source_files "${target_name}" SOURCES)
    #遍历源文件
    FOREACH(sourcefile ${source_files})
        #获取当前源文件的编译参数
        GET_PROPERTY(defs SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS)
        #获取当前文件的绝对路径
        GET_FILENAME_COMPONENT(filepath "${sourcefile}" ABSOLUTE)
        #将绝对路径中的项目路径替换成空,得到源文件相对于项目路径的相对路径
        STRING(REPLACE ${unwanted_prefix}/ "" relpath ${filepath})
        #将我们要加的编译参数(__FILE__定义)添加到原来的编译参数里面
        LIST(APPEND defs "__FILE__=\"${relpath}\"")
        #重新设置源文件的编译参数
        SET_PROPERTY(
            SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS ${defs})
    ENDFOREACH()
ENDFUNCTION()

ADD_COMPILE_OPTIONS(-Wno-builtin-macro-redefined)