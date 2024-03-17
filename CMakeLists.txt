cmake_minimum_required(VERSION 3.17)

set(SAM_RESEARCH "$ENV{HOME}/SAMResearch"  CACHE PATH "SAM RESEARCH FOLDER")
include(${SAM_RESEARCH}/shared_header.cmake)
set(SR_LIBTORCH_CACHE "${SR_CACHE}/torch"  CACHE PATH "libtorch cache path")

rapids_cuda_init_architectures(DLBlueprint)
project(DLBlueprint LANGUAGES C CXX CUDA)
option(BUILD_EXAMPLES ON)
option(BUILD_TEST ON)

set(CMAKE_CXX_STANDARD 20)
set(Boost_USE_STATIC_LIBS        ON) # Turn off static linking
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)

find_package(fmt REQUIRED )
find_package(Boost REQUIRED )
find_package(yaml-cpp REQUIRED)
find_package(tl-ranges CONFIG REQUIRED)
include(${SR_CMAKE}/get_libtorch.cmake)

add_library(DLBlueprint STATIC compile.cpp variable.cpp blueprint.cpp)
target_include_directories(DLBlueprint PUBLIC .)
target_link_libraries(DLBlueprint PUBLIC torch yaml-cpp::yaml-cpp PRIVATE fmt::fmt)

#add_subdirectory(tensorboard_logger)

add_subdirectory(common)
add_subdirectory(factory)
add_subdirectory(models)
add_subdirectory(options)

if (${BUILD_TEST})
    add_subdirectory(test)
endif()
#
#if(${BUILD_EXAMPLES})
#    file(GLOB Examples examples/*.cpp)
#    foreach(SRC_FILE IN LISTS Examples)
#        string(REPLACE .cpp "" executable_name ${SRC_FILE})
#        string(REPLACE ${CMAKE_SOURCE_DIR}/examples/ "" executable_name ${executable_name})
#        add_executable(${executable_name} ${SRC_FILE} vision/impala_residual_block.h vision_debugger.h)
#        target_link_libraries(${executable_name} PRIVATE ${PROJECT_NAME} ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger)
#        target_include_directories( ${executable_name} PRIVATE . tensorboard_logger)
#    endforeach()
#endif()