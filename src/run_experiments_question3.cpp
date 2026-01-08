#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "tetris_mdp.hpp"
#include "exp3.hpp"

static double clamp01(double x)
{
    if (x < 0.0)
        return 0.0;
    if (x > 1.0)
        return 1.0;
    return x;
}

static double normalize_episode_reward(double total_reward, int max_steps)
{
    double denom = 6.0 * (double)max_steps;
    if (denom <= 0.0)
        denom = 1.0;
    return clamp01(total_reward / denom);
}

int main()
{
    std::cout << std::fixed << std::setprecision(3);

    MDP mdp;

    const double discount = 0.99;
    auto q1 = mdp.value_iteration(discount, 50000);
    auto q2 = mdp.minimax_value_iteration(discount, 50000);

    std::mt19937 rng(12345);

    std::vector<std::vector<Action>> experts;
    experts.push_back(q2.player_policy);
    experts.push_back(q1.policy);
    experts.push_back(mdp.random_policy(rng));

    const int K = (int)experts.size();

    auto q2_selection = [&mdp, &q2](const std::vector<int> &board, int /*step*/, std::mt19937 & /*rng*/) -> PieceType
    {
        return mdp.adv_piece_for_board(board, q2.adv_policy);
    };

    auto q1_selection = [](const std::vector<int> & /*board*/, int /*step*/, std::mt19937 &rng) -> PieceType
    {
        std::bernoulli_distribution d(0.5);
        return d(rng) ? PieceType::Line : PieceType::Angle;
    };

    auto switching = [](const std::vector<int> & /*board*/, int step, std::mt19937 &rng) -> PieceType
    {
        // example: first 400 moves p(line)=0.9, afterwards p(line)=0.1
        double p_line = (step < 400) ? 0.9 : 0.1;
        std::bernoulli_distribution d(p_line);
        return d(rng) ? PieceType::Line : PieceType::Angle;
    };

    // Pick which piece selection to run here:
    // const MDP::PieceSelector env = q1_selection;
    const MDP::PieceSelector env = q2_selection;
    // const MDP::PieceSelector env = switching;

    // EXP3 parameters (bandit-level)
    const double gamma_explore = 0.10; // exploration mixing
    const double eta = 0.07;           // learning rate (works well for small K)
    EXP3 exp3(K, eta, gamma_explore);

    const int max_steps = 1000; // cap for a "full game" (as in your Q2 experiments)
    const int N = 2000;         // number of bandit rounds (episodes)

    double total_alg_reward = 0.0;

    for (int n = 0; n < N; ++n)
    {
        int arm = exp3.sample(rng);
        auto ep = mdp.simulate_episode_custom(experts[arm], env, rng, max_steps);

        total_alg_reward += ep.total_reward;

        // EXP3 expects reward in [0,1]
        double r01 = normalize_episode_reward(ep.total_reward, max_steps);
        exp3.update(arm, r01);

        if ((n + 1) % 200 == 0)
        {
            auto p = exp3.probs();
            std::cout << "Episode " << (n + 1)
                      << ": avg_reward=" << (total_alg_reward / (n + 1))
                      << ", probs=[";
            for (int k = 0; k < K; ++k)
            {
                std::cout << p[k] << (k + 1 < K ? ", " : "]\n");
            }
        }
    }

    double alg_avg = total_alg_reward / N;

    // Evaluate each expert alone on the same environment (Monte Carlo estimate)
    // This gives a practical proxy for regret vs the best expert in hindsight.
    const int eval_eps = 1000;
    std::vector<double> expert_avgs(K, 0.0);
    for (int k = 0; k < K; ++k)
    {
        std::mt19937 rng_k(777);
        EvalStats st = mdp.evaluate_policy_custom(experts[k], env, eval_eps, rng_k, max_steps);
        expert_avgs[k] = st.avg_total_reward;
    }

    int best_k = 0;
    for (int k = 1; k < K; ++k)
    {
        if (expert_avgs[k] > expert_avgs[best_k])
            best_k = k;
    }

    std::cout << "\n=== Q3 EXP3 results (episode = bandit round) ===\n";
    std::cout << "Piece selection:  Q2 \n";
    std::cout << "N episodes: " << N << ", max_steps: " << max_steps << "\n";
    std::cout << "Algorithm (EXP3) avg total reward per episode: " << alg_avg << "\n";

    for (int k = 0; k < K; ++k)
    {
        std::cout << "Expert " << k
                  << " avg reward: " << expert_avgs[k]
                  << (k == 0 ? " (Q2 policy)" : (k == 1 ? " (Q1 policy)" : " (random policy)"))
                  << "\n";
    }

    std::cout << "Best expert (estimated): k=" << best_k
              << ", avg=" << expert_avgs[best_k] << "\n";

    // A simple regret proxy (per-episode gap * N). This is NOT the formal bandit regret
    // because counterfactual rewards are unobserved; we estimate expert means via Monte Carlo.
    double regret_proxy = (expert_avgs[best_k] - alg_avg) * N;
    std::cout << "Regret proxy (N*(best_avg - alg_avg)): " << regret_proxy << "\n";

    return 0;
}
