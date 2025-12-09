cmake_minimum_required(VERSION 3.10)
project(mdp_tetris LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

add_executable(run_experiments "")

target_sources(run_experiments PRIVATE
    src/run_experiments.cpp
    src/tetris_mdp.cpp
)

target_include_directories(run_experiments PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(print_all_states_and_actions "")

target_sources(print_all_states_and_actions PRIVATE
    src/print_all_states_and_actions.cpp
    src/tetris_mdp.cpp
)

target_include_directories(print_all_states_and_actions PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)