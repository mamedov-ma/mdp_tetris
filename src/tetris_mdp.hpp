#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <random>

// Parameters for mdp
inline constexpr int W = 4;            // width (columns)
inline constexpr int H = 4;            // height (rows)
inline constexpr double GAMMA = 0.99;  // Discount coeff
inline constexpr int FULL_ROW_MASK = (1 << W) - 1;

enum class PieceType { Line, Angle, Terminal };

std::string piece_to_string(PieceType p);

struct Cell {
    int dx;
    int dy;
};

struct State {
    std::vector<int> board;
    PieceType piece;

    bool operator==(const State &other) const
    {
        return piece == other.piece && board == other.board;
    }
};

struct StateHash {
    std::size_t operator()(const State &s) const;
};

struct Action {
    int orient_id;
    int x;

    bool operator==(const Action &other) const
    {
        return orient_id == other.orient_id && x == other.x;
    }
};

struct SAKey {
    int state_id;
    Action action;

    bool operator==(const SAKey &other) const
    {
        return state_id == other.state_id && action.orient_id == other.action.orient_id && action.x == other.action.x;
    }
};

struct SAKeyHash {
    std::size_t operator()(const SAKey &k) const;
};

struct Transition {
    std::unordered_map<int, double> probs;  // s' -> p
    double reward;
};

std::string board_to_string(const std::vector<int> &board);

struct ValueIterationResult {
    std::vector<double> V;
    std::vector<Action> policy;
    int iterations;
};

struct EvalStats {
    double avg_total_reward;
    double avg_length;
    double avg_reward_per_step;
};

class MDP {
public:
    static constexpr int W = 4;
    static constexpr int H = 4;
    static constexpr int FULL_ROW_MASK = (1 << W) - 1;

    MDP();

    ValueIterationResult value_iteration(double gamma = 0.99, int max_iter = 500, double tol = 1e-6) const;

    const std::vector<State> &get_states() const
    {
        return states;
    }
    const std::vector<std::vector<Action>> &get_state_actions() const
    {
        return state_actions;
    }

    double initial_state_value(const std::vector<double> &V) const;
    EvalStats evaluate_policy(const std::vector<Action> &policy, int episodes, std::mt19937 &rng,
                              int max_steps = 1000) const;
    std::vector<Action> random_policy(std::mt19937 &rng) const;

private:
    std::vector<State> states;
    std::unordered_map<State, int, StateHash> state_index;
    int terminal_id = -1;
    std::unordered_map<SAKey, Transition, SAKeyHash> transitions;
    std::vector<std::vector<Action>> state_actions;
    std::vector<PieceType> piece_types = {PieceType::Line, PieceType::Angle};
    std::unordered_map<PieceType, std::vector<std::vector<Cell>>> pieces;

    void init_pieces();
    std::pair<std::vector<int>, int> clear_full_lines(const std::vector<int> &board) const;

    struct DropResult {
        std::vector<int> new_board;
        double reward;
        bool game_over;
    };

    DropResult drop_piece(const std::vector<int> &board, const std::vector<Cell> &shape, int x) const;
    std::pair<int, bool> add_state(const std::vector<int> &board, PieceType piece);
    int sample_next_state(int s_id, const Action &a, std::mt19937 &rng, double &reward) const;
};
