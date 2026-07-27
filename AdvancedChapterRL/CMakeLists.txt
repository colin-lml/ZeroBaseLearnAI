
cmake_minimum_required (VERSION 3.8)


if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("AdvancedChapterRL")
set(targetExeName "AdvancedChapterRL")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(torchpath "D:/libtorch_gpu2.11.0/debug")
else()
    set(torchpath "D:/libtorch_gpu2.11.0/debug")
endif()

find_package(Torch REQUIRED PATHS "${torchpath}/share/cmake/Torch/")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}../../AdvancedChapterRL/X64/)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}../bin2/)
endif()


file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable (${targetExeName} ${SRC_FILES})
target_link_libraries(${targetExeName} "${TORCH_LIBRARIES}")


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AdvancedChapterRL PROPERTY CXX_STANDARD 20)
endif()

