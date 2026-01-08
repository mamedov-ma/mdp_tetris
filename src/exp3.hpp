#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// EXP3 for adversarial bandits.
// - K arms.
// - Rewards assumed in [0,1].
// - Uses exploration mixing parameter "gamma".
//
// Algorithm (standard):
//   p_i = (1-gamma) * w_i/sum(w) + gamma/K
//   sample I ~ p
//   r_hat_I = r_I / p_I
//   w_I <- w_I * exp(eta * r_hat_I)

class EXP3
{
public:
    EXP3(int K, double eta, double gamma)
        : K_(K), eta_(eta), gamma_(gamma), w_(K, 1.0)
    {
        if (K_ <= 0)
            throw std::invalid_argument("K must be positive");
        if (!(eta_ > 0.0))
            throw std::invalid_argument("eta must be > 0");
        if (!(gamma_ > 0.0 && gamma_ <= 1.0))
            throw std::invalid_argument("gamma must be in (0,1]");
    }

    int K() const { return K_; }

    std::vector<double> probs() const
    {
        double sum_w = std::accumulate(w_.begin(), w_.end(), 0.0);
        if (sum_w <= 0.0)
            sum_w = 1.0;
        std::vector<double> p(K_, 0.0);
        for (int i = 0; i < K_; ++i)
        {
            p[i] = (1.0 - gamma_) * (w_[i] / sum_w) + gamma_ * (1.0 / K_);
        }
        return p;
    }

    int sample(std::mt19937 &rng) const
    {
        auto p = probs();
        std::discrete_distribution<int> dist(p.begin(), p.end());
        return dist(rng);
    }

    void update(int chosen_arm, double reward01)
    {
        if (chosen_arm < 0 || chosen_arm >= K_)
            throw std::out_of_range("arm index out of range");
        if (reward01 < 0.0)
            reward01 = 0.0;
        if (reward01 > 1.0)
            reward01 = 1.0;

        auto p = probs();
        double p_i = p[chosen_arm];
        if (p_i <= 0.0)
            return;

        double r_hat = reward01 / p_i;
        w_[chosen_arm] *= std::exp(eta_ * r_hat);

        double max_w = 0.0;
        for (double wi : w_)
            max_w = std::max(max_w, wi);
        if (max_w > 1e100)
        {
            for (double &wi : w_)
                wi /= max_w;
        }
    }

private:
    int K_;
    double eta_;
    double gamma_;
    std::vector<double> w_;
};
