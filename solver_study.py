from collections import defaultdict, deque
import math

# -------------------------
# 1. Parameters
# -------------------------

W = 4  # width (columns)
H = 4  # height (rows)
GAMMA = 0.99

# Board representation:
# board is a tuple of H integers; each int is a bitmask of length W.
# row 0 is the bottom row, row H-1 is the top row.

EMPTY_BOARD = tuple([0] * H)
FULL_ROW_MASK = (1 << W) - 1  # 0b1111 for W=4

PIECE_TYPES = ("line", "angle")

# -------------------------
# 2. Pieces (orientations)
# -------------------------

PIECES = {
    "line": [
        # Horizontal: three in a row on same row
        [(0, 0), (1, 0), (2, 0)],
        # Vertical: three stacked
        [(0, 0), (0, 1), (0, 2)],
    ],
    "angle": [
        # L-shapes in a 2x2 box, bottom row is y=0
        [(0, 0), (1, 0), (0, 1)],  # L0
        [(0, 0), (0, 1), (1, 1)],  # L1
        [(0, 1), (1, 1), (1, 0)],  # L2
        [(0, 0), (1, 0), (1, 1)],  # L3
    ],
}

# -------------------------
# 3. Board helpers
# -------------------------

def clear_full_lines(board):
    """Remove fully filled rows and return (new_board, num_cleared)."""
    rows = list(board)
    full_indices = [i for i, row in enumerate(rows) if row == FULL_ROW_MASK]
    k = len(full_indices)
    if k == 0:
        return board, 0

    # Keep only non-full rows
    new_rows = [row for i, row in enumerate(rows) if i not in full_indices]
    # Add empty rows on top
    new_rows += [0] * k
    return tuple(new_rows), k


def drop_piece(board, shape, x):
    valid_positions = []  # (y, cells)

    for y in range(H):
        cells = []
        inside = True
        for dx, dy in shape:
            cx = x + dx
            cy = y + dy
            if not (0 <= cx < W and 0 <= cy < H):
                inside = False
                break
            if board[cy] & (1 << cx):
                inside = False
                break
            cells.append((cx, cy))
        if inside:
            valid_positions.append((y, cells))

    if not valid_positions:
        return board, 0, True

    ys = [y for y, _ in valid_positions]
    ys_set = set(ys)
    y_to_cells = {y: cells for y, cells in valid_positions}

    candidates = [y for y in ys if (y == 0 or (y - 1) not in ys_set)]
    landing_y = min(candidates)
    landing_cells = y_to_cells[landing_y]

    new_rows = list(board)
    for cx, cy in landing_cells:
        new_rows[cy] |= (1 << cx)
    new_board = tuple(new_rows)

    new_board, cleared = clear_full_lines(new_board)
    reward_table = {0: 0, 1: 1, 2: 3, 3: 6}
    reward = reward_table[cleared]

    return new_board, reward, False



# -------------------------
# 4. Build MDP (states, transitions)
# -------------------------

state_index = {}
states = []  # list of (board, piece)


def add_state(board, piece):
    """Add a new state if not exists. Return (index, is_new)."""
    key = (board, piece)
    if key in state_index:
        return state_index[key], False
    idx = len(states)
    state_index[key] = idx
    states.append(key)
    return idx, True


# Terminal state
terminal_id, _ = add_state(None, "terminal")

# Transitions: (s, a) -> {"probs": {s': p}, "reward": r}
transitions = {}
state_actions = defaultdict(list)

# BFS queue over reachable states (excluding terminal)
queue = deque()

# Start from empty board with each possible current piece
for p in PIECE_TYPES:
    s_id, is_new = add_state(EMPTY_BOARD, p)
    if is_new:
        queue.append(s_id)

while queue:
    s_id = queue.popleft()
    board, piece = states[s_id]

    if piece == "terminal":
        continue

    # Enumerate actions: all orientations and horizontal positions
    piece_orients = PIECES[piece]
    for orient_id, shape in enumerate(piece_orients):
        # Max x so that piece's max dx stays inside board
        max_dx = max(dx for dx, dy in shape)
        for x in range(W - max_dx):
            a = (orient_id, x)
            key = (s_id, a)
            if key in transitions:
                continue  # already processed this action

            new_board, reward, game_over = drop_piece(board, shape, x)
            probs = defaultdict(float)

            if game_over:
                # Game ends
                probs[terminal_id] = 1.0
            else:
                # Next piece is random among PIECE_TYPES
                for next_piece in PIECE_TYPES:
                    ns_id, is_new = add_state(new_board, next_piece)
                    probs[ns_id] += 1.0 / len(PIECE_TYPES)
                    if is_new:
                        queue.append(ns_id)

            transitions[key] = {"probs": dict(probs), "reward": reward}
            state_actions[s_id].append(a)

# -------------------------
# 5. Value Iteration
# -------------------------

def value_iteration(gamma=GAMMA, max_iter=50000, tol=1e-6):
    n_states = len(states)
    V = [0.0] * n_states

    iters = 0
    for it in range(max_iter):
        iters = it + 1
        delta = 0.0
        V_new = V[:]

        for s_id, (board, piece) in enumerate(states):
            if piece == "terminal":
                V_new[s_id] = 0.0
                continue

            actions = state_actions.get(s_id, [])
            if not actions:
                V_new[s_id] = 0.0
                continue

            best_q = -math.inf
            for a in actions:
                info = transitions[(s_id, a)]
                r = info["reward"]
                q = 0.0
                for ns_id, p in info["probs"].items():
                    q += p * (r + gamma * V[ns_id])
                if q > best_q:
                    best_q = q
            V_new[s_id] = best_q
            delta = max(delta, abs(V_new[s_id] - V[s_id]))

        V = V_new
        if delta < tol:
            break

    # Extract greedy policy
    policy = {}
    for s_id, (board, piece) in enumerate(states):
        if piece == "terminal":
            policy[s_id] = None
            continue
        actions = state_actions.get(s_id, [])
        if not actions:
            policy[s_id] = None
            continue
        best_a = None
        best_q = -math.inf
        for a in actions:
            info = transitions[(s_id, a)]
            r = info["reward"]
            q = 0.0
            for ns_id, p in info["probs"].items():
                q += p * (r + gamma * V[ns_id])
            if q > best_q:
                best_q = q
                best_a = a
        policy[s_id] = best_a

    return V, policy, iters

def initial_state_value(V):
    s_line = state_index[(EMPTY_BOARD, "line")]
    s_angle = state_index[(EMPTY_BOARD, "angle")]
    return 0.5 * (V[s_line] + V[s_angle])

import random

def sample_next_state(info):
    r = info["reward"]
    rnd = random.random()
    cum = 0.0
    for ns_id, p in info["probs"].items():
        cum += p
        if rnd <= cum:
            return ns_id, r
    # На всякий случай, если из-за численной ошибки rnd > сумма
    ns_id = list(info["probs"].keys())[-1]
    return ns_id, r

def simulate_episode(policy, max_steps=1000):
    # начальное состояние: пустая доска, случайная фигура
    s_line = state_index[(EMPTY_BOARD, "line")]
    s_angle = state_index[(EMPTY_BOARD, "angle")]
    s_id = random.choice([s_line, s_angle])

    total_reward = 0
    steps = 0

    for _ in range(max_steps):
        board, piece = states[s_id]
        if piece == "terminal":
            break

        a = policy.get(s_id)
        if a is None:
            break

        info = transitions[(s_id, a)]
        s_id, r = sample_next_state(info)
        total_reward += r
        steps += 1

    return total_reward, steps

def evaluate_policy(policy, episodes=1000):
    total_rewards = 0.0
    total_steps = 0
    for _ in range(episodes):
        r, steps = simulate_episode(policy)
        total_rewards += r
        total_steps += steps
    avg_reward = total_rewards / episodes
    avg_length = total_steps / episodes
    avg_reward_per_step = total_rewards / total_steps if total_steps > 0 else 0.0
    return avg_reward, avg_length, avg_reward_per_step

def random_policy():
    pol = {}
    for s_id, (board, piece) in enumerate(states):
        if piece == "terminal":
            pol[s_id] = None
            continue
        actions = state_actions.get(s_id, [])
        pol[s_id] = random.choice(actions) if actions else None
    return pol


if __name__ == "__main__":
    # V, policy = value_iteration()

    # print("Total states:", len(states))
    # for s_id, (board, piece) in enumerate(states):
    #     if piece == "terminal":
    #         continue
    #     print("State", s_id, "piece:", piece, "board:", board)
    #     print("  all actions:", state_actions[s_id])
    #     print("  best action:", policy[s_id])

    gammas = [0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.98, 0.99, 0.995]

    for gamma in gammas:
        V, policy, iters = value_iteration(gamma=gamma)
        v0 = initial_state_value(V)
        gain_est = (1.0 - gamma) * v0
        print(f"gamma={gamma:.3f}, iterations={iters}, V0={v0:.4f}, gain≈{gain_est:.4f}")
        avgR, avgL, avgR_step = evaluate_policy(policy, episodes=1000)
        print(f"gamma={gamma}")
        print(f"  avg total score per game: {avgR:.3f}")
        print(f"  avg episode length: {avgL:.3f} moves")
        print(f"  avg score per move: {avgR_step:.3f}")

    gamma = 0.99
    V, opt_policy, iters = value_iteration(gamma=gamma)

    avgR_opt, avgL_opt, avgRstep_opt = evaluate_policy(opt_policy)
    rand_pol = random_policy()
    avgR_rand, avgL_rand, avgRstep_rand = evaluate_policy(rand_pol)

    print("Optimal policy:")
    print(f"  avg total score: {avgR_opt:.3f}, avg len: {avgL_opt:.3f}, avg score/step: {avgRstep_opt:.3f}")
    print("Random policy:")
    print(f"  avg total score: {avgR_rand:.3f}, avg len: {avgL_rand:.3f}, avg score/step: {avgRstep_rand:.3f}")
