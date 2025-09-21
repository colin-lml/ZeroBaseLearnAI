# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CMakeProject1")


find_package(OpenCV REQUIRED PATHS D:/opencv_all/build)

## find_package(Torch REQUIRED PATHS D:/libtorch2.2.0/cpu/share/cmake/Torch)
## find_package(Torch REQUIRED PATHS D:/libtorch2.2.0/debug/share/cmake/Torch)
find_package(Torch REQUIRED PATHS D:/libtorch2.8.0/debug/share/cmake/Torch/)
## find_package(Torch REQUIRED PATHS D:/libtorch2.10.0/cou/share/cmake/Torch/)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# 将源代码添加到此项目的可执行文件。
add_executable (CMakeProject1 "CMakeProject1.cpp" "CMakeProject1.h")

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll"
                     "D:/opencv_all/build/x64/vc16/bin/*.dll")


add_custom_command(TARGET CMakeProject1
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CMakeProject1>)



if(OpenCV_FOUND)
    target_link_libraries(CMakeProject1 PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
else()
    target_link_libraries(CMakeProject1 PRIVATE ${TORCH_LIBRARIES} )
endif()



if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeProject1 PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
