cmake_minimum_required(VERSION 3.22)
project(wm)

set(CMAKE_CXX_STANDARD 17)

add_executable(wm src/wm/main.cpp src/tool/csv.h)
# include_directories(/usr/local/Cellar/eigen/3.4.0_1/include)