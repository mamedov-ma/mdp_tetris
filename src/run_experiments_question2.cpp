#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

#include "tetris_mdp.hpp"

int main()
{
    MDP mdp;
    std::vector<double> gammas = {0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.98, 0.99, 0.995};
    std::cout << std::fixed << std::setprecision(3);

    for (double gamma : gammas)
    {
        using clock = std::chrono::high_resolution_clock;

        auto start = clock::now();
        MinimaxResult res = mdp.minimax_value_iteration(gamma, 50000);
        auto end = clock::now();
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double V0 = mdp.initial_state_value_minimax(res.V);
        double gain = (1.0 - gamma) * V0;
        auto stats = mdp.evaluate_minimax_strategies(res.player_policy, res.adv_policy, 1000);
        std::cout << "gamma=" << gamma << ", iterations=" << res.iterations << ", V0=" << V0 << ", gain=" << gain
                  << ", execution time ms: " << elapsed_ns
                  << ", avg total score per game: " << stats.avg_total_reward
                  << ", avg episode length: " << stats.avg_length
                  << ", avg score per move: " << stats.avg_reward_per_step << "\n";
    }

    auto result = mdp.minimax_value_iteration(0.99, 50000);
    MDP mdp2;
    auto res_opt = mdp2.value_iteration(0.99, 50000);

    {
        auto stats = mdp.evaluate_minimax_strategies(res_opt.policy, result.adv_policy, 1000);
        std::cout << "Simulated (deterministic) run with (pi*, sigma*):\n";
        std::cout << "avg total reward per game: " << stats.avg_total_reward << "\n";
        std::cout << "avg episode length:        " << stats.avg_length << "\n";
        std::cout << "avg reward per step:       " << stats.avg_reward_per_step << "\n";
    }

    {
        auto stats = mdp.evaluate_minimax_strategies(result.player_policy, result.adv_policy, 1000);
        std::cout << "Simulated (deterministic) run with (pi*, sigma*):\n";
        std::cout << "avg total reward per game: " << stats.avg_total_reward << "\n";
        std::cout << "avg episode length:        " << stats.avg_length << "\n";
        std::cout << "avg reward per step:       " << stats.avg_reward_per_step << "\n";
    }

    {
        std::mt19937 rng(12345);
        EvalStats stats = mdp2.evaluate_policy(result.player_policy, 1000, rng);
        std::cout << "Simulated (deterministic) run with (pi*, sigma*):\n";
        std::cout << "avg total reward per game: " << stats.avg_total_reward << "\n";
        std::cout << "avg episode length:        " << stats.avg_length << "\n";
        std::cout << "avg reward per step:       " << stats.avg_reward_per_step << "\n";
    }

    int diff = 0;
    for (size_t i = 0; i < result.player_policy.size(); ++i)
    {
        if (!(result.player_policy[i] == res_opt.policy[i]))
            diff++;
    }
    std::cout << "Policy differences (Q2 vs Q1): " << diff << " states\n";

    auto traj_q1 = mdp.rollout_adversarial_states(res_opt.policy, result.adv_policy, 1000);
    auto traj_q2 = mdp.rollout_adversarial_states(result.player_policy, result.adv_policy, 1000);

    int diff_on_traj = 0;
    for (size_t t = 0; t + 1 < traj_q1.size() && t + 1 < traj_q2.size(); ++t)
    {
        int s1 = traj_q1[t];
        int s2 = traj_q2[t];
        if (s1 == s2)
        {
            if (!(res_opt.policy[s1] == result.player_policy[s1]))
                diff_on_traj++;
        }
    }
    std::cout << "Diff decisions on visited states: " << diff_on_traj << "\n";

    return 0;
}
