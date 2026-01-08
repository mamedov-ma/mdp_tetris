#include "tetris_mdp.hpp"

#include <queue>
#include <cmath>
#include <limits>
#include <functional>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>

std::size_t StateHash::operator()(const State &s) const
{
    std::size_t h = 0;
    std::hash<int> hi;
    for (int row : s.board)
    {
        h ^= hi(row) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    h ^= hi(static_cast<int>(s.piece)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

std::size_t SAKeyHash::operator()(const SAKey &k) const
{
    std::size_t h = 0;
    std::hash<int> hi;
    h ^= hi(k.state_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hi(k.action.orient_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hi(k.action.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

std::string piece_to_string(PieceType p)
{
    switch (p)
    {
    case PieceType::Line:
        return "line";
    case PieceType::Angle:
        return "angle";
    case PieceType::Terminal:
        return "terminal";
    }
    return "unknown";
}

std::string board_to_string(const std::vector<int> &board)
{
    std::string s = "[";
    for (std::size_t i = 0; i < board.size(); ++i)
    {
        s += std::to_string(board[i]);
        if (i + 1 < board.size())
            s += ", ";
    }
    s += "]";
    return s;
}

void MDP::init_pieces()
{
    // line
    pieces[PieceType::Line] = {// Horizontal: three in a row on same row
                               {{0, 0}, {1, 0}, {2, 0}},
                               // Vertical: three stacked
                               {{0, 0}, {0, 1}, {0, 2}}};
    // angle (L-shapes in 2x2 box, bottom row y=0)
    pieces[PieceType::Angle] = {
        {{0, 0}, {1, 0}, {0, 1}}, // L0
        {{0, 0}, {0, 1}, {1, 1}}, // L1
        {{0, 1}, {1, 1}, {1, 0}}, // L2
        {{0, 0}, {1, 0}, {1, 1}}  // L3
    };
}

int MDP::sample_next_state(int s_id, const Action &a, std::mt19937 &rng, double &reward) const
{
    SAKey key{s_id, a};
    const Transition &tr = transitions.at(key);
    reward = tr.reward;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    double cum = 0.0;
    int last_ns = -1;

    for (const auto &kv : tr.probs)
    {
        int ns_id = kv.first;
        double p = kv.second;
        cum += p;
        last_ns = ns_id;
        if (u <= cum)
        {
            return ns_id;
        }
    }
    // in case of issues with probability sums
    return last_ns;
}

double MDP::initial_state_value(const std::vector<double> &V) const
{
    std::vector<int> empty_board(H, 0);
    int line_id = -1;
    int angle_id = -1;

    for (int i = 0; i < static_cast<int>(states.size()); ++i)
    {
        const State &s = states[i];
        if (s.board == empty_board)
        {
            if (s.piece == PieceType::Line)
                line_id = i;
            else if (s.piece == PieceType::Angle)
                angle_id = i;
        }
    }

    if (line_id < 0 || angle_id < 0)
    {
        throw std::runtime_error("Initial states (empty board) not found");
    }

    return 0.5 * (V[line_id] + V[angle_id]);
}

EvalStats MDP::evaluate_policy(const std::vector<Action> &policy, int episodes, std::mt19937 &rng, int max_steps) const
{
    // Find the ids of the empty starting states once
    std::vector<int> empty_board(H, 0);
    int line_id = -1;
    int angle_id = -1;

    for (int i = 0; i < static_cast<int>(states.size()); ++i)
    {
        const State &s = states[i];
        if (s.board == empty_board)
        {
            if (s.piece == PieceType::Line)
                line_id = i;
            else if (s.piece == PieceType::Angle)
                angle_id = i;
        }
    }

    if (line_id < 0 || angle_id < 0)
    {
        throw std::runtime_error("Initial states (empty board) not found in evaluate_policy");
    }

    std::uniform_int_distribution<int> piece_dist(0, 1);
    double total_reward = 0.0;
    long long total_steps = 0;

    for (int ep = 0; ep < episodes; ++ep)
    {
        int s_id = (piece_dist(rng) == 0) ? line_id : angle_id;
        double R = 0.0;
        int steps = 0;

        for (int t = 0; t < max_steps; ++t)
        {
            const State &s = states[s_id];
            if (s.piece == PieceType::Terminal)
                break;

            const Action &a = policy[s_id];
            if (a.orient_id < 0) // no action
                break;

            double r = 0.0;
            int ns_id = sample_next_state(s_id, a, rng, r);
            R += r;
            ++steps;
            s_id = ns_id;
        }

        total_reward += R;
        total_steps += steps;
    }

    EvalStats stats;
    stats.avg_total_reward = total_reward / episodes;
    stats.avg_length = (total_steps > 0) ? (double)total_steps / episodes : 0.0;
    stats.avg_reward_per_step = (total_steps > 0) ? total_reward / total_steps : 0.0;
    return stats;
}

std::vector<Action> MDP::random_policy(std::mt19937 &rng) const
{
    std::vector<Action> pol(states.size(), Action{-1, -1});
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int s_id = 0; s_id < static_cast<int>(states.size()); ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
            continue;
        const auto &actions = state_actions[s_id];
        if (actions.empty())
            continue;

        std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(actions.size()) - 1);
        pol[s_id] = actions[idx_dist(rng)];
    }
    return pol;
}

std::pair<std::vector<int>, int> MDP::clear_full_lines(const std::vector<int> &board) const
{
    std::vector<int> rows = board;
    std::vector<int> new_rows;
    int k = 0;

    for (int row : rows)
    {
        if (row == FULL_ROW_MASK)
        {
            ++k;
        }
        else
        {
            new_rows.push_back(row);
        }
    }

    if (k == 0)
    {
        return {board, 0};
    }

    for (int i = 0; i < k; ++i)
    {
        new_rows.push_back(0);
    }

    return {new_rows, k};
}

MDP::DropResult MDP::drop_piece(const std::vector<int> &board, const std::vector<Cell> &shape, int x) const
{
    struct PosCells
    {
        int y;
        std::vector<std::pair<int, int>> cells;
    };
    std::vector<PosCells> valid_positions;

    for (int y = 0; y < H; ++y)
    {
        bool inside = true;
        std::vector<std::pair<int, int>> cells;

        for (const auto &c : shape)
        {
            int cx = x + c.dx;
            int cy = y + c.dy;
            if (!(0 <= cx && cx < W && 0 <= cy && cy < H))
            {
                inside = false;
                break;
            }
            if (board[cy] & (1 << cx))
            {
                inside = false;
                break;
            }
            cells.emplace_back(cx, cy);
        }

        if (inside)
        {
            valid_positions.push_back(PosCells{y, cells});
        }
    }

    if (valid_positions.empty())
    {
        return {board, 0.0, true};
    }

    std::vector<int> ys;
    ys.reserve(valid_positions.size());
    std::unordered_map<int, std::vector<std::pair<int, int>>> y_to_cells;
    for (const auto &pc : valid_positions)
    {
        ys.push_back(pc.y);
        y_to_cells[pc.y] = pc.cells;
    }

    std::unordered_map<int, bool> ys_set;
    for (int y : ys)
        ys_set[y] = true;

    std::vector<int> candidates;
    for (int y : ys)
    {
        if (y == 0 || ys_set.find(y - 1) == ys_set.end())
        {
            candidates.push_back(y);
        }
    }

    int landing_y = candidates[0];
    for (int y : candidates)
    {
        if (y < landing_y)
            landing_y = y;
    }

    const auto &landing_cells = y_to_cells[landing_y];

    std::vector<int> new_rows = board;
    for (auto [cx, cy] : landing_cells)
    {
        new_rows[cy] |= (1 << cx);
    }

    auto cleared_res = clear_full_lines(new_rows);
    new_rows = cleared_res.first;
    int cleared = cleared_res.second;

    int reward_table[4] = {0, 1, 3, 6};
    double reward = 0.0;
    if (cleared >= 0 && cleared < 4)
    {
        reward = reward_table[cleared];
    }

    return {new_rows, reward, false};
}

std::pair<int, bool> MDP::add_state(const std::vector<int> &board, PieceType piece)
{
    State s{board, piece};
    auto it = state_index.find(s);
    if (it != state_index.end())
    {
        return {it->second, false};
    }
    int idx = static_cast<int>(states.size());
    states.push_back(s);
    state_index[s] = idx;
    state_actions.emplace_back(); // empty action list
    return {idx, true};
}

MDP::MDP()
{
    init_pieces();
    std::vector<int> empty_board(H, 0);

    // 1. Terminal state
    {
        std::vector<int> dummy;
        auto res = add_state(dummy, PieceType::Terminal);
        terminal_id = res.first;
    }

    // 2. Starting states: empty board + each piece
    std::queue<int> q;
    for (PieceType p : piece_types)
    {
        auto res = add_state(empty_board, p);
        int s_id = res.first;
        bool is_new = res.second;
        if (is_new)
            q.push(s_id);
    }

    // 3. BFS over reachable states
    while (!q.empty())
    {
        int s_id = q.front();
        q.pop();

        State s = states[s_id];

        if (s.piece == PieceType::Terminal)
            continue;

        const auto &piece_orients = pieces[s.piece];

        for (int orient_id = 0; orient_id < static_cast<int>(piece_orients.size()); ++orient_id)
        {
            const auto &shape = piece_orients[orient_id];

            int max_dx = 0;
            for (const auto &c : shape)
                if (c.dx > max_dx)
                    max_dx = c.dx;

            for (int x = 0; x < W - max_dx; ++x)
            {
                Action a{orient_id, x};
                SAKey key{s_id, a};
                if (transitions.find(key) != transitions.end())
                    continue;

                auto drop_res = drop_piece(s.board, shape, x);
                const std::vector<int> &new_board = drop_res.new_board;
                double reward = drop_res.reward;
                bool game_over = drop_res.game_over;

                Transition t;
                t.reward = reward;

                if (game_over)
                {
                    t.probs[terminal_id] = 1.0;
                }
                else
                {
                    double prob = 1.0 / piece_types.size();
                    for (PieceType next_piece : piece_types)
                    {
                        auto res2 = add_state(new_board, next_piece);
                        int ns_id = res2.first;
                        bool is_new2 = res2.second;
                        t.probs[ns_id] += prob;
                        if (is_new2)
                            q.push(ns_id);
                    }
                }

                transitions[key] = t;
                state_actions[s_id].push_back(a);
            }
        }
    }
}

ValueIterationResult MDP::value_iteration(double gamma, int max_iter, double tol) const
{
    int n_states = static_cast<int>(states.size());
    std::vector<double> V(n_states, 0.0);
    std::vector<Action> policy(n_states, Action{-1, -1});

    int iterations = 0;

    for (int it = 0; it < max_iter; ++it)
    {
        iterations = it + 1;
        double delta = 0.0;
        std::vector<double> V_new = V;

        for (int s_id = 0; s_id < n_states; ++s_id)
        {
            const State &s = states[s_id];
            if (s.piece == PieceType::Terminal)
            {
                V_new[s_id] = 0.0;
                continue;
            }
            const auto &actions = state_actions[s_id];
            if (actions.empty())
            {
                V_new[s_id] = 0.0;
                continue;
            }

            double best_q = -std::numeric_limits<double>::infinity();
            for (const auto &a : actions)
            {
                SAKey key{s_id, a};
                const Transition &tr = transitions.at(key);
                double r = tr.reward;
                double q = 0.0;
                for (const auto &kv : tr.probs)
                {
                    int ns_id = kv.first;
                    double p = kv.second;
                    q += p * (r + gamma * V[ns_id]);
                }
                if (q > best_q)
                {
                    best_q = q;
                }
            }
            V_new[s_id] = best_q;
            double diff = std::fabs(V_new[s_id] - V[s_id]);
            if (diff > delta)
                delta = diff;
        }

        V.swap(V_new);
        if (delta < tol)
            break;
    }

    // Extract the greedy policy
    for (int s_id = 0; s_id < n_states; ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
        {
            policy[s_id] = Action{-1, -1};
            continue;
        }
        const auto &actions = state_actions[s_id];
        if (actions.empty())
        {
            policy[s_id] = Action{-1, -1};
            continue;
        }

        double best_q = -std::numeric_limits<double>::infinity();
        Action best_a{-1, -1};

        for (const auto &a : actions)
        {
            SAKey key{s_id, a};
            const Transition &tr = transitions.at(key);
            double r = tr.reward;
            double q = 0.0;
            for (const auto &kv : tr.probs)
            {
                int ns_id = kv.first;
                double p = kv.second;
                q += p * (r + gamma * V[ns_id]);
            }
            if (q > best_q)
            {
                best_q = q;
                best_a = a;
            }
        }
        policy[s_id] = best_a;
    }

    ValueIterationResult res;
    res.V = std::move(V);
    res.policy = std::move(policy);
    res.iterations = iterations;
    return res;
}

MinimaxResult MDP::minimax_value_iteration(double gamma, int max_iter, double tol) const
{
    int n_states = static_cast<int>(states.size());
    std::vector<double> V(n_states, 0.0);
    std::vector<Action> player_policy(n_states, Action{-1, -1});

    int iterations = 0;

    for (int it = 0; it < max_iter; ++it)
    {
        iterations = it + 1;
        double delta = 0.0;
        std::vector<double> V_new = V;

        for (int s_id = 0; s_id < n_states; ++s_id)
        {
            const State &s = states[s_id];
            if (s.piece == PieceType::Terminal)
            {
                V_new[s_id] = 0.0;
                continue;
            }

            const auto &actions = state_actions[s_id];
            if (actions.empty())
            {
                V_new[s_id] = 0.0;
                continue;
            }

            double best_q = -std::numeric_limits<double>::infinity();

            for (const auto &a : actions)
            {
                SAKey key{s_id, a};
                const Transition &tr = transitions.at(key);

                // adversary chooses the next piece => take min over successors
                double next_val = std::numeric_limits<double>::infinity();
                for (const auto &kv : tr.probs)
                {
                    int ns_id = kv.first;
                    next_val = std::min(next_val, V[ns_id]);
                }

                double q = tr.reward + gamma * next_val;
                if (q > best_q)
                {
                    best_q = q;
                }
            }

            V_new[s_id] = best_q;

            double diff = std::fabs(V_new[s_id] - V[s_id]);
            if (diff > delta)
                delta = diff;
        }

        V.swap(V_new);
        if (delta < tol)
            break;
    }

    // Extract optimal player policy (greedy w.r.t minimax backup)
    for (int s_id = 0; s_id < n_states; ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
        {
            player_policy[s_id] = Action{-1, -1};
            continue;
        }

        const auto &actions = state_actions[s_id];
        if (actions.empty())
        {
            player_policy[s_id] = Action{-1, -1};
            continue;
        }

        double best_q = -std::numeric_limits<double>::infinity();
        Action best_a{-1, -1};

        for (const auto &a : actions)
        {
            SAKey key{s_id, a};
            const Transition &tr = transitions.at(key);

            double next_val = std::numeric_limits<double>::infinity();
            for (const auto &kv : tr.probs)
            {
                int ns_id = kv.first;
                next_val = std::min(next_val, V[ns_id]);
            }

            double q = tr.reward + gamma * next_val;
            if (q > best_q)
            {
                best_q = q;
                best_a = a;
            }
        }

        player_policy[s_id] = best_a;
    }

    // Extract optimal adversary policy σ*(board): choose piece that minimizes V(board, piece)
    std::vector<PieceType> adv_policy(n_states, PieceType::Terminal);

    std::unordered_map<std::vector<int>, std::pair<int, int>, BoardHash> board_to_ids;
    board_to_ids.reserve(states.size());

    for (int s_id = 0; s_id < n_states; ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
            continue;

        auto &pr = board_to_ids[s.board];
        if (pr.first == 0 && pr.second == 0)
        {
            // default pair is {0,0}; but state_id can be 0.
            // we'll fix it by using a separate sentinel check below.
        }
    }

    board_to_ids.clear();
    for (int s_id = 0; s_id < n_states; ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
            continue;

        auto it = board_to_ids.find(s.board);
        if (it == board_to_ids.end())
        {
            board_to_ids.emplace(s.board, std::make_pair(-1, -1));
            it = board_to_ids.find(s.board);
        }

        if (s.piece == PieceType::Line)
        {
            it->second.first = s_id;
        }
        else if (s.piece == PieceType::Angle)
        {
            it->second.second = s_id;
        }
    }

    for (const auto &kv : board_to_ids)
    {
        int line_id = kv.second.first;
        int angle_id = kv.second.second;
        if (line_id < 0 || angle_id < 0)
        {
            continue;
        }

        PieceType choice = (V[line_id] <= V[angle_id]) ? PieceType::Line : PieceType::Angle;
        adv_policy[line_id] = choice;
        adv_policy[angle_id] = choice;
    }

    MinimaxResult res;
    res.V = std::move(V);
    res.player_policy = std::move(player_policy);
    res.adv_policy = std::move(adv_policy);
    res.iterations = iterations;
    return res;
}

double MDP::initial_state_value_minimax(const std::vector<double> &V) const
{
    std::vector<int> empty_board(H, 0);
    int line_id = -1;
    int angle_id = -1;

    for (int i = 0; i < static_cast<int>(states.size()); ++i)
    {
        const State &s = states[i];
        if (s.board == empty_board)
        {
            if (s.piece == PieceType::Line)
                line_id = i;
            else if (s.piece == PieceType::Angle)
                angle_id = i;
        }
    }

    if (line_id < 0 || angle_id < 0)
    {
        throw std::runtime_error("Initial states (empty board) not found");
    }

    return std::min(V[line_id], V[angle_id]);
}

int MDP::next_state_adversarial(int s_id, const Action &a, const std::vector<PieceType> &adv_policy, double &reward) const
{
    SAKey key{s_id, a};
    const Transition &tr = transitions.at(key);
    reward = tr.reward;

    if (tr.probs.size() == 1)
    {
        auto it = tr.probs.begin();
        int only_ns = it->first;
        return only_ns;
    }

    // Otherwise we should have exactly two successors: same board, different piece.
    int ns1 = -1;
    int ns2 = -1;
    for (const auto &kv : tr.probs)
    {
        if (ns1 < 0)
            ns1 = kv.first;
        else
            ns2 = kv.first;
    }

    if (ns1 < 0)
        return terminal_id;

    PieceType chosen_piece = adv_policy[ns1]; // same for ns2
    if (ns2 < 0)
    {
        return ns1;
    }

    if (states[ns1].piece == chosen_piece)
        return ns1;
    return ns2;
}

EvalStats MDP::evaluate_minimax_strategies(const std::vector<Action> &player_policy, const std::vector<PieceType> &adv_policy, int episodes, int max_steps) const
{
    std::vector<int> empty_board(H, 0);
    int line_id = -1;
    int angle_id = -1;

    for (int i = 0; i < static_cast<int>(states.size()); ++i)
    {
        const State &s = states[i];
        if (s.board == empty_board)
        {
            if (s.piece == PieceType::Line)
                line_id = i;
            else if (s.piece == PieceType::Angle)
                angle_id = i;
        }
    }

    if (line_id < 0 || angle_id < 0)
    {
        throw std::runtime_error("Initial states (empty board) not found in evaluate_minimax_strategies");
    }

    // initial piece chosen by adversary
    PieceType init_piece = adv_policy[line_id];
    int start_id = (states[line_id].piece == init_piece) ? line_id : angle_id;

    double total_reward = 0.0;
    long long total_steps = 0;

    for (int ep = 0; ep < episodes; ++ep)
    {
        int s_id = start_id;
        double R = 0.0;
        int steps = 0;

        for (int t = 0; t < max_steps; ++t)
        {
            const State &s = states[s_id];
            if (s.piece == PieceType::Terminal)
                break;

            const Action &a = player_policy[s_id];
            if (a.orient_id < 0)
                break;

            double r = 0.0;
            int ns_id = next_state_adversarial(s_id, a, adv_policy, r);

            R += r;
            ++steps;
            s_id = ns_id;
        }

        total_reward += R;
        total_steps += steps;
    }

    EvalStats stats;
    stats.avg_total_reward = total_reward / episodes;
    stats.avg_length = (total_steps > 0) ? (double)total_steps / episodes : 0.0;
    stats.avg_reward_per_step = (total_steps > 0) ? total_reward / total_steps : 0.0;
    return stats;
}

std::vector<int> MDP::rollout_adversarial_states(const std::vector<Action> &player_policy, const std::vector<PieceType> &adv_policy, int max_steps) const
{
    std::vector<int> empty_board(H, 0);
    int line_id = -1, angle_id = -1;
    for (int i = 0; i < (int)states.size(); ++i)
    {
        const State &s = states[i];
        if (s.board == empty_board)
        {
            if (s.piece == PieceType::Line)
                line_id = i;
            else if (s.piece == PieceType::Angle)
                angle_id = i;
        }
    }
    if (line_id < 0 || angle_id < 0)
        throw std::runtime_error("Empty start not found");

    PieceType init_piece = adv_policy[line_id];
    int s_id = (states[line_id].piece == init_piece) ? line_id : angle_id;

    std::vector<int> visited;
    visited.reserve(max_steps + 1);
    visited.push_back(s_id);

    for (int t = 0; t < max_steps; ++t)
    {
        if (states[s_id].piece == PieceType::Terminal)
            break;
        const Action &a = player_policy[s_id];
        if (a.orient_id < 0)
            break;

        double r = 0.0;
        int ns_id = next_state_adversarial(s_id, a, adv_policy, r);
        s_id = ns_id;
        visited.push_back(s_id);
    }
    return visited;
}

std::pair<int, int> MDP::successors_line_angle(int s_id, const Action &a, double &reward) const
{
    SAKey key{s_id, a};
    const Transition &tr = transitions.at(key);
    reward = tr.reward;

    // Terminal transition
    if (tr.probs.size() == 1)
    {
        auto it = tr.probs.begin();
        int only_ns = it->first;
        return {only_ns, only_ns};
    }

    int line_id = -1;
    int angle_id = -1;
    for (const auto &kv : tr.probs)
    {
        int ns_id = kv.first;
        const State &ns = states[ns_id];
        if (ns.piece == PieceType::Line)
            line_id = ns_id;
        else if (ns.piece == PieceType::Angle)
            angle_id = ns_id;
        else if (ns.piece == PieceType::Terminal)
        {
            // Should not happen here, but handle gracefully.
            line_id = angle_id = ns_id;
        }
    }

    if (line_id < 0 || angle_id < 0)
    {
        // Fallback: return any ordering we have
        int ns1 = -1, ns2 = -1;
        for (const auto &kv : tr.probs)
        {
            if (ns1 < 0)
                ns1 = kv.first;
            else
                ns2 = kv.first;
        }
        if (ns2 < 0)
            return {ns1, ns1};
        return {ns1, ns2};
    }
    return {line_id, angle_id};
}

ValueIterationResult MDP::value_iteration_with_piece_model(const std::function<double(const std::vector<int> &board)> &p_line, double gamma, int max_iter, double tol) const
{
    int n_states = static_cast<int>(states.size());
    std::vector<double> V(n_states, 0.0);
    std::vector<Action> policy(n_states, Action{-1, -1});

    int iterations = 0;

    for (int it = 0; it < max_iter; ++it)
    {
        iterations = it + 1;
        double delta = 0.0;
        std::vector<double> V_new = V;

        for (int s_id = 0; s_id < n_states; ++s_id)
        {
            const State &s = states[s_id];
            if (s.piece == PieceType::Terminal)
            {
                V_new[s_id] = 0.0;
                continue;
            }

            const auto &actions = state_actions[s_id];
            if (actions.empty())
            {
                V_new[s_id] = 0.0;
                continue;
            }

            double best_q = -std::numeric_limits<double>::infinity();

            for (const auto &a : actions)
            {
                double r = 0.0;
                auto [line_id, angle_id] = successors_line_angle(s_id, a, r);

                // If terminal, successors are equal.
                if (line_id == angle_id)
                {
                    best_q = std::max(best_q, r + gamma * V[line_id]);
                    continue;
                }

                // Both successors share the same post-drop board.
                const std::vector<int> &b_next = states[line_id].board;
                double pl = std::clamp(p_line(b_next), 0.0, 1.0);
                double q = r + gamma * (pl * V[line_id] + (1.0 - pl) * V[angle_id]);
                if (q > best_q)
                    best_q = q;
            }

            V_new[s_id] = best_q;
            double diff = std::fabs(V_new[s_id] - V[s_id]);
            if (diff > delta)
                delta = diff;
        }

        V.swap(V_new);
        if (delta < tol)
            break;
    }

    // Greedy policy extraction under the learned model
    for (int s_id = 0; s_id < n_states; ++s_id)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
        {
            policy[s_id] = Action{-1, -1};
            continue;
        }
        const auto &actions = state_actions[s_id];
        if (actions.empty())
        {
            policy[s_id] = Action{-1, -1};
            continue;
        }

        double best_q = -std::numeric_limits<double>::infinity();
        Action best_a{-1, -1};

        for (const auto &a : actions)
        {
            double r = 0.0;
            auto [line_id, angle_id] = successors_line_angle(s_id, a, r);
            double q;
            if (line_id == angle_id)
            {
                q = r + gamma * V[line_id];
            }
            else
            {
                const std::vector<int> &b_next = states[line_id].board;
                double pl = std::clamp(p_line(b_next), 0.0, 1.0);
                q = r + gamma * (pl * V[line_id] + (1.0 - pl) * V[angle_id]);
            }
            if (q > best_q)
            {
                best_q = q;
                best_a = a;
            }
        }

        policy[s_id] = best_a;
    }

    ValueIterationResult res;
    res.V = std::move(V);
    res.policy = std::move(policy);
    res.iterations = iterations;
    return res;
}

int MDP::next_state_by_piece_choice(int s_id, const Action &a, PieceType chosen_piece, double &reward) const
{
    SAKey key{s_id, a};
    const Transition &tr = transitions.at(key);
    reward = tr.reward;

    // terminal transition
    if (tr.probs.size() == 1)
    {
        return tr.probs.begin()->first;
    }

    // Otherwise we expect two successors: same board, different piece.
    int ns_line = -1;
    int ns_angle = -1;
    for (const auto &kv : tr.probs)
    {
        int ns = kv.first;
        if (states[ns].piece == PieceType::Line)
            ns_line = ns;
        else if (states[ns].piece == PieceType::Angle)
            ns_angle = ns;
    }

    if (chosen_piece == PieceType::Line && ns_line >= 0)
        return ns_line;
    if (chosen_piece == PieceType::Angle && ns_angle >= 0)
        return ns_angle;

    // if something is inconsistent, go terminal.
    return terminal_id;
}

MDP::EpisodeResult MDP::simulate_episode_custom(const std::vector<Action> &policy, const PieceSelector &selector, std::mt19937 &rng, int max_steps) const
{
    if (max_steps <= 0)
        throw std::invalid_argument("max_steps must be positive");

    // find empty-board state ids
    std::vector<int> empty_board(H, 0);
    int line_id = -1;
    int angle_id = -1;
    for (int i = 0; i < (int)states.size(); ++i)
    {
        const State &s = states[i];
        if (s.board == empty_board)
        {
            if (s.piece == PieceType::Line)
                line_id = i;
            else if (s.piece == PieceType::Angle)
                angle_id = i;
        }
    }
    if (line_id < 0 || angle_id < 0)
        throw std::runtime_error("Initial states (empty board) not found in simulate_episode_custom");

    // adversary chooses the initial piece based on the empty board
    PieceType init_piece = selector(empty_board, /*step=*/0, rng);
    if (init_piece != PieceType::Line && init_piece != PieceType::Angle)
        init_piece = PieceType::Line;

    int s_id = (init_piece == PieceType::Line) ? line_id : angle_id;

    double total_reward = 0.0;
    int steps = 0;

    for (int t = 0; t < max_steps; ++t)
    {
        const State &s = states[s_id];
        if (s.piece == PieceType::Terminal)
            break;

        const Action &a = policy[s_id];
        if (a.orient_id < 0)
            break;

        double r = 0.0;
        // Apply player's action -> deterministic next board embedded in successors
        // Then choose next piece based on the resulting board
        // We need the resulting board, so we peek at one successor state.

        SAKey key{s_id, a};
        const Transition &tr = transitions.at(key);
        r = tr.reward;

        if (tr.probs.size() == 1)
        {
            // terminal
            total_reward += r;
            ++steps;
            s_id = tr.probs.begin()->first;
            break;
        }

        // get the next board from any successor
        int any_ns = tr.probs.begin()->first;
        const std::vector<int> &next_board = states[any_ns].board;

        PieceType next_piece = selector(next_board, /*step=*/t + 1, rng);
        if (next_piece != PieceType::Line && next_piece != PieceType::Angle)
            next_piece = PieceType::Line;

        int ns_id = next_state_by_piece_choice(s_id, a, next_piece, r);

        total_reward += r;
        ++steps;
        s_id = ns_id;
    }

    EpisodeResult out;
    out.total_reward = total_reward;
    out.steps = steps;
    return out;
}

EvalStats MDP::evaluate_policy_custom(const std::vector<Action> &policy, const PieceSelector &selector, int episodes, std::mt19937 &rng, int max_steps) const
{
    if (episodes <= 0)
        throw std::invalid_argument("episodes must be positive");

    double total_reward = 0.0;
    long long total_steps = 0;

    for (int ep = 0; ep < episodes; ++ep)
    {
        EpisodeResult res = simulate_episode_custom(policy, selector, rng, max_steps);
        total_reward += res.total_reward;
        total_steps += res.steps;
    }

    EvalStats stats;
    stats.avg_total_reward = total_reward / episodes;
    stats.avg_length = (double)total_steps / episodes;
    stats.avg_reward_per_step = (total_steps > 0) ? (total_reward / (double)total_steps) : 0.0;
    return stats;
}

PieceType MDP::adv_piece_for_board(const std::vector<int> &board, const std::vector<PieceType> &adv_policy) const
{
    // Locate the two state ids corresponding to this board (line and angle).
    int line_id = -1;
    int angle_id = -1;

    for (int i = 0; i < (int)states.size(); ++i)
    {
        const State &s = states[i];
        if (s.piece == PieceType::Terminal)
            continue;
        if (s.board != board)
            continue;
        if (s.piece == PieceType::Line)
            line_id = i;
        else if (s.piece == PieceType::Angle)
            angle_id = i;
    }

    if (line_id < 0 || angle_id < 0)
        return PieceType::Line;
    if ((int)adv_policy.size() <= line_id)
        return PieceType::Line;

    PieceType choice = adv_policy[line_id]; // same choice is stored for both ids in Q2 code
    if (choice != PieceType::Line && choice != PieceType::Angle)
        choice = PieceType::Line;
    return choice;
}
