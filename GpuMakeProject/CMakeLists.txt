# CMakeList.txt: GpuMakeProject 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("GpuMakeProject")

set(targetExeName "MakeProject")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(torchpath "D:/libtorch_gpu2.11.0/debug")
else()
    set(torchpath "D:/libtorch_gpu2.11.0/debug")
endif()

find_package(Torch REQUIRED PATHS "${torchpath}/share/cmake/Torch/")


include_directories(${TORCH_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}../Debug/)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}../bin2/)
endif()

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable (${targetExeName} ${SRC_FILES})


target_link_libraries(${targetExeName} "${TORCH_LIBRARIES}")

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll"
                     )

add_custom_command(TARGET ${targetExeName}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${targetExeName}>)





# 将源代码添加到此项目的可执行文件。
##add_executable (GpuMakeProject "GpuMakeProject.cpp" "GpuMakeProject.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${targetExeName} PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
