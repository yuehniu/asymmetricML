cmake_minimum_required(VERSION 3.10)
project(asymmetricML)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)

add_executable(asymmetricML main.cpp App/test.cpp App/test.h App/app.cpp App/app.h)

add_library(sgxdnn SHARED App/app.cpp App/app.h)
