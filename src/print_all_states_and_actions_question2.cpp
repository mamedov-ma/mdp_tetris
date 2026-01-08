#include <iostream>
#include <vector>
#include <iomanip>

#include "tetris_mdp.hpp"

int main()
{
    MDP mdp;

    auto result = mdp.minimax_value_iteration(GAMMA, 5000, 1e-8);

    const auto &states = mdp.get_states();
    const auto &state_actions = mdp.get_state_actions();
    const auto &V = result.V;
    const auto &pi = result.player_policy;
    const auto &sigma = result.adv_policy;

    std::cout << "Total states: " << states.size() << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";

    double v0 = mdp.initial_state_value_minimax(V);
    std::cout << "Initial minimax value (adversary chooses first piece): " << v0 << "\n";
    std::cout << "(1-gamma)*V0 approx gain: " << (1.0 - GAMMA) * v0 << "\n\n";

    for (int s_id = 0; s_id < static_cast<int>(states.size()); ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
        {
            continue;
        }

        std::cout << "State " << s_id
                  << ": piece=" << piece_to_string(s.piece)
                  << ", board=" << board_to_string(s.board)
                  << ", V=" << V[s_id]
                  << ", adv_choice(next_piece)=" << piece_to_string(sigma[s_id])
                  << "\n";

        std::cout << "  all actions: ";
        for (const auto &a : state_actions[s_id])
        {
            std::cout << "(orient=" << a.orient_id << ", x=" << a.x << ") ";
        }
        std::cout << "\n";

        const Action &best = pi[s_id];
        if (best.orient_id == -1)
        {
            std::cout << "  best action: None\n";
        }
        else
        {
            std::cout << "  best action: (orient=" << best.orient_id << ", x=" << best.x << ")\n";
        }
    }

    return 0;
}
