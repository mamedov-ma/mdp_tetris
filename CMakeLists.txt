cmake_minimum_required(VERSION 3.10)
project(mdp_tetris LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

add_executable(print_all_states_and_actions_question1 "")
target_sources(print_all_states_and_actions_question1 PRIVATE
    src/print_all_states_and_actions_question1.cpp
    src/tetris_mdp.cpp
)
target_include_directories(print_all_states_and_actions_question1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(print_all_states_and_actions_question2 "")
target_sources(print_all_states_and_actions_question2 PRIVATE
    src/print_all_states_and_actions_question2.cpp
    src/tetris_mdp.cpp
)
target_include_directories(print_all_states_and_actions_question2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(run_experiments_question1 "")
target_sources(run_experiments_question1 PRIVATE
    src/run_experiments_question1.cpp
    src/tetris_mdp.cpp
)
target_include_directories(run_experiments_question1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(run_experiments_question2 "")
target_sources(run_experiments_question2 PRIVATE
    src/run_experiments_question2.cpp
    src/tetris_mdp.cpp
)
target_include_directories(run_experiments_question2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(run_experiments_question3 "")
target_sources(run_experiments_question3 PRIVATE
    src/run_experiments_question3.cpp
    src/tetris_mdp.cpp
)
target_include_directories(run_experiments_question3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
