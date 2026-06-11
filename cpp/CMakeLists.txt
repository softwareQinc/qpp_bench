cmake_minimum_required(VERSION 3.20)
project(standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(qpp REQUIRED)

add_executable(qft_bench src/qft_bench.cpp)
target_link_libraries(qft_bench PRIVATE qpp::qpp)
