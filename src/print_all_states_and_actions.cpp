#include <iostream>
#include <vector>
#include <iomanip>

#include "tetris_mdp.hpp"

int main()
{
    MDP mdp;

    ValueIterationResult result = mdp.value_iteration();

    const auto &states = mdp.get_states();
    const auto &state_actions = mdp.get_state_actions();
    const auto &V = result.V;
    const auto &policy = result.policy;

    std::cout << "Total states: " << states.size() << "\n";
    for (int s_id = 0; s_id < static_cast<int>(states.size()); ++s_id) {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
            continue;

        std::cout << "State " << s_id << ": piece: " << piece_to_string(s.piece)
                  << ", board: " << board_to_string(s.board) << ", V=" << V[s_id] << "\n";

        std::cout << "  all actions: ";
        for (const auto &a : state_actions[s_id]) {
            std::cout << "(orient=" << a.orient_id << ", x=" << a.x << ") ";
        }
        std::cout << "\n";

        const Action &best = policy[s_id];
        if (best.orient_id == -1) {
            std::cout << "  best action: None\n";
        } else {
            std::cout << "  best action: (orient=" << best.orient_id << ", x=" << best.x << ")\n";
        }
    }

    return 0;
}
