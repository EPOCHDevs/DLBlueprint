add_executable(compiler_test compiler_test.cpp)

find_package(Catch2 CONFIG REQUIRED)
target_link_libraries(compiler_test PRIVATE DLBlueprint Catch2::Catch2 Catch2::Catch2WithMain)

set(__TEST_FILES__ "${CMAKE_CURRENT_LIST_DIR}/files")
target_include_directories(compiler_test PRIVATE ..)
target_compile_definitions(compiler_test PRIVATE -D__TEST_FILES__="${__TEST_FILES__}")
