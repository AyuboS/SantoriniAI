import math
import time
import pandas as pd
import matplotlib.pyplot as plt

WIN_SCORE = 999999
LOSE_SCORE = -999999


class GameState:
    def __init__(self, board, workers, turn):
        self.board = board
        self.workers = workers
        self.turn = turn


def create_opening_position():
    return GameState([[0] * 5 for _ in range(5)], {'X': [(1, 2), (2, 2)], 'O': [(3, 1), (3, 3)]}, "X")


def create_early_game_position():
    return GameState([[0, 0, 1, 2, 0], [1, 2, 3, 1, 0], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                     {'X': [(1, 2), (2, 2)], 'O': [(3, 1), (3, 3)]}, "X")


def create_mid_game_position():
    return GameState([[0, 1, 1, 2, 0], [1, 2, 3, 2, 0], [0, 2, 2, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                     {'X': [(2, 1), (3, 2)], 'O': [(1, 3), (4, 4)]}, "X")


def create_late_game_position():
    return GameState([[0, 1, 2, 3, 1], [1, 3, 3, 2, 1], [2, 2, 2, 1, 0], [1, 3, 1, 0, 0], [0, 1, 0, 0, 0]],
                     {'X': [(1, 2), (2, 3)], 'O': [(3, 1), (4, 4)]}, "X")


def generate_all_legal_moves(state):
    moves = []
    for worker in state.workers[state.turn]:
        x, y = worker
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 5:
                    moves.append(((x, y), (nx, ny)))
    return moves


def apply_action(state, action):
    old_pos, new_pos = action
    new_workers = {p: list(workers) for p, workers in state.workers.items()}
    new_workers[state.turn].remove(old_pos)
    new_workers[state.turn].append(new_pos)

    return GameState([row[:] for row in state.board], new_workers, 'O' if state.turn == 'X' else 'X')


def is_terminal(state):
    return False


def evaluate(state):
    return sum(state.board[x][y] for x, y in state.workers['X']) - sum(state.board[x][y] for x, y in state.workers['O'])


def minimax_with_metrics(state, depth, maximizing, metrics):
    metrics['nodes_generated'] += 1
    if depth == 0 or is_terminal(state):
        metrics['nodes_evaluated'] += 1
        return evaluate(state)

    if maximizing:
        value = -math.inf
        for action in generate_all_legal_moves(state):
            new_state = apply_action(state, action)
            value = max(value, minimax_with_metrics(new_state, depth - 1, False, metrics))
        return value
    else:
        value = math.inf
        for action in generate_all_legal_moves(state):
            new_state = apply_action(state, action)
            value = min(value, minimax_with_metrics(new_state, depth - 1, True, metrics))
        return value


def alpha_beta_with_metrics(state, depth, alpha, beta, maximizing, metrics):
    metrics['nodes_generated'] += 1
    if depth == 0 or is_terminal(state):
        metrics['nodes_evaluated'] += 1
        return evaluate(state)

    if maximizing:
        value = -math.inf
        for action in generate_all_legal_moves(state):
            new_state = apply_action(state, action)
            value = max(value, alpha_beta_with_metrics(new_state, depth - 1, alpha, beta, False, metrics))
            alpha = max(alpha, value)
            if alpha >= beta:
                metrics['nodes_pruned'] += 1
                break
        return value
    else:
        value = math.inf
        for action in generate_all_legal_moves(state):
            new_state = apply_action(state, action)
            value = min(value, alpha_beta_with_metrics(new_state, depth - 1, alpha, beta, True, metrics))
            beta = min(beta, value)
            if alpha >= beta:
                metrics['nodes_pruned'] += 1
                break
        return value


def run_complexity_analysis():
    test_positions = [create_opening_position(), create_early_game_position(), create_mid_game_position(),
                      create_late_game_position()]
    depths = [1, 2, 3, 4]
    results = []

    for position_name, position in enumerate(test_positions):
        position_name = ["Opening", "Early", "Mid", "Late"][position_name]
        actual_branching = len(generate_all_legal_moves(position))

        for depth in depths:
            minimax_metrics = {'nodes_generated': 0, 'nodes_evaluated': 0}
            start_time = time.time()
            minimax_with_metrics(position, depth, True, minimax_metrics)
            minimax_time = time.time() - start_time

            alphabeta_metrics = {'nodes_generated': 0, 'nodes_evaluated': 0, 'nodes_pruned': 0}
            start_time = time.time()
            alpha_beta_with_metrics(position, depth, float('-inf'), float('inf'), True, alphabeta_metrics)
            alphabeta_time = time.time() - start_time

            results.append({
                'position': position_name,
                'depth': depth,
                'algorithm': 'Minimax',
                'nodes_generated': minimax_metrics['nodes_generated'],
                'time_ms': minimax_time * 1000
            })
            results.append({
                'position': position_name,
                'depth': depth,
                'algorithm': 'Alpha-Beta',
                'nodes_generated': alphabeta_metrics['nodes_generated'],
                'time_ms': alphabeta_time * 1000
            })

    df = pd.DataFrame(results)
    print(df)


run_complexity_analysis()
