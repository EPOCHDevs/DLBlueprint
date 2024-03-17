add_library(DLBlueprint STATIC compile.cpp variable.cpp blueprint.cpp)
target_include_directories(DLBlueprint PUBLIC .)
target_link_libraries(DLBlueprint PUBLIC torch yaml-cpp::yaml-cpp PRIVATE fmt::fmt)

add_subdirectory(common)
add_subdirectory(factory)
add_subdirectory(models)
add_subdirectory(options)