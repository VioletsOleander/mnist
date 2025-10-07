cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mnist)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(mnist src/main.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}")
