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

add_executable(test_conjugate_gradient
    tests/test_conjugate_gradient.cpp
)

target_include_directories(test_conjugate_gradient
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME conjugate_gradient
    COMMAND test_conjugate_gradient
)

add_executable(test_trainer
    tests/test_trainer.cpp
)

target_link_libraries(test_trainer
    PRIVATE
        mini_ml
)

add_test(
    NAME trainer
    COMMAND test_trainer
)

add_executable(lab1_polynomial
    lab/lab1/main.cpp
)

target_include_directories(lab1_polynomial
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(lab1_polynomial
    PRIVATE
        mini_ml
)

add_executable(lab2_gen
    lab/lab2/gen.cpp
)

add_executable(lab2_logistic
    lab/lab2/main.cpp
)

target_include_directories(lab2_logistic
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(lab2_logistic
    PRIVATE
        mini_ml
)

target_include_directories(lab2_gen
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(lab2_gen
    PRIVATE
        mini_ml
)

target_include_directories(lab1_polynomial
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(lab1_polynomial
    PRIVATE
        mini_ml
)
