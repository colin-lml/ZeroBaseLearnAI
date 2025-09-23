# CMakeList.txt: LibtorchSimpleNeuralNetwork 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("LibtorchSimpleNeuralNetwork")


set(targetExeName "LibtorchSimpleNeuralNetwork")

set(opencvpath "D:/opencv/build/")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(torchpath "D:/libtorch2.8.0/debug")
else()
    set(torchpath "D:/libtorch2.8.0/cpu")
endif()

find_package(Torch REQUIRED PATHS "${torchpath}/share/cmake/Torch/")

find_package(OpenCV REQUIRED PATHS ${opencvpath})

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(${TORCH_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}../bin/)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}../bin2/)
endif()

add_executable (${targetExeName} "LibtorchSimpleNeuralNetwork.cpp" "LibtorchSimpleNeuralNetwork.h")


if(OpenCV_FOUND)
    target_link_libraries(${targetExeName} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
else()
    target_link_libraries(${targetExeName} "${TORCH_LIBRARIES}")
endif()


file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll"
                     "${opencvpath}/x64/vc16/bin/*.dll")

add_custom_command(TARGET ${targetExeName}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${targetExeName}>)





if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET LibtorchSimpleNeuralNetwork PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
