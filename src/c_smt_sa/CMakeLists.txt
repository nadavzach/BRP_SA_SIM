cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SMT_SA)

add_compile_options(-O3)
file(GLOB SOURCES "*.cpp")
add_executable(smt_sa_tb ${SOURCES})
target_link_libraries(smt_sa_tb)

