#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

#include "tetris_mdp.hpp"

int main()
{
    MDP mdp;

    std::mt19937 rng(12345);

    std::vector<double> gammas = {0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.98, 0.99, 0.995};
    std::cout << std::fixed << std::setprecision(3);

    for (double gamma : gammas)
    {
        using clock = std::chrono::high_resolution_clock;

        auto start = clock::now();
        ValueIterationResult res = mdp.value_iteration(gamma, 50000);
        auto end = clock::now();
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double V0 = mdp.initial_state_value(res.V);
        double gain = (1.0 - gamma) * V0;
        EvalStats stats_opt = mdp.evaluate_policy(res.policy, 1000, rng);
        std::cout << "gamma=" << gamma << ", iterations=" << res.iterations << ", V0=" << V0 << ", gain=" << gain
                  << ", execution time ms: " << elapsed_ns
                  << ", avg total score per game: " << stats_opt.avg_total_reward
                  << ", avg episode length: " << stats_opt.avg_length
                  << ", avg score per move: " << stats_opt.avg_reward_per_step << "\n";
    }

    double gamma_opt = 0.99;
    ValueIterationResult res_opt = mdp.value_iteration(gamma_opt);
    EvalStats stats_opt = mdp.evaluate_policy(res_opt.policy, 1000, rng);

    std::vector<Action> rand_pol = mdp.random_policy(rng);
    EvalStats stats_rand = mdp.evaluate_policy(rand_pol, 1000, rng);

    std::cout << "\nEvaluation with gamma=" << gamma_opt << " (1000 episodes):\n";

    std::cout << "Optimal policy:\n";
    std::cout << "  avg total score per game = " << stats_opt.avg_total_reward << "\n";
    std::cout << "  avg episode length       = " << stats_opt.avg_length << " moves\n";
    std::cout << "  avg score per move       = " << stats_opt.avg_reward_per_step << "\n";

    std::cout << "Random policy:\n";
    std::cout << "  avg total score per game = " << stats_rand.avg_total_reward << "\n";
    std::cout << "  avg episode length       = " << stats_rand.avg_length << " moves\n";
    std::cout << "  avg score per move       = " << stats_rand.avg_reward_per_step << "\n";

    return 0;
}
