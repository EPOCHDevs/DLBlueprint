add_library(DLBlueprint STATIC dl_blueprint/compile.cpp dl_blueprint/variable.cpp dl_blueprint/blueprint.cpp)
target_include_directories(DLBlueprint PUBLIC .)
target_link_libraries(DLBlueprint PUBLIC torch yaml-cpp::yaml-cpp PRIVATE fmt::fmt)

add_subdirectory(dl_blueprint)