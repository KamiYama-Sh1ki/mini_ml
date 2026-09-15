cmake_minimum_required(VERSION 3.20)

project(mini_ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(mini_ml STATIC
    src/backends/cpu/serial.cpp
)

target_include_directories(mini_ml
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(test_dataset
    tests/test_dataset.cpp
)

target_include_directories(test_dataset
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME dataset
    COMMAND test_dataset
)

add_executable(test_random
    tests/test_random.cpp
)

target_include_directories(test_random
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME random
    COMMAND test_random
)

add_executable(test_sampling
    tests/test_sampling.cpp
)

target_include_directories(test_sampling
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME sampling
    COMMAND test_sampling
)

add_executable(test_backend
    tests/test_backend.cpp
)

target_link_libraries(test_backend
    PRIVATE
        mini_ml
)

add_test(
    NAME backend
    COMMAND test_backend
)

add_executable(lab1_polynomial
    lab1/main.cpp
)

target_link_libraries(lab1_polynomial
    PRIVATE
        mini_ml
)
