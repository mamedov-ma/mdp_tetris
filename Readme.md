# Baby Tetris MDP

This project implements a Markov Decision Process (MDP) model of a simplified “Baby Tetris” game (4 columns, limited height) and solves it using value iteration. It also includes scripts to run experiments and print all reachable states and actions.

## Requirements

- Standard Linux environment
- POSIX shell (`/bin/sh` or `bash`)
- A C++ compiler with C++17 support (e.g. `g++`)
- No external libraries are required

## Build Instructions

From the project root directory, run:

```bash
chmod +x build.sh && ./build.sh
```

This script will:

* Create a `build/` directory (if it does not exist)
* Compile the project into two executables inside `build/`:


## Running the Programs

After a successful build, go to the `build/` directory:

```bash
cd build
```

### 1. Run experiments

```bash
./run_experiments_question1
./run_experiments_question2
```

This executable:

* Runs value iteration for several discount factors `γ`
* Prints the number of iterations, estimated value of the initial state, approximate gain, and execution time
* Simulates episodes to report average total score, average episode length, and average score per move for the optimal policy (and possibly baselines)

### 2. Print all states and actions

```bash
./print_all_states_and_actions_question1
./print_all_states_and_actions_question2
```

This executable:

* Enumerates all reachable states
* Prints each state, the current piece, and the list of available actions
* Shows the best action according to the computed optimal value function

---
