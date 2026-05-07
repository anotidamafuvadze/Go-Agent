from typing import Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import pickle

from tqdm import tqdm
from go_search_problem import GoProblem, GoState, HeuristicGoProblem

torch.set_default_tensor_type(torch.FloatTensor)
BLACK = 0
WHITE = 1

FEATURE_SIZE = 21 

def get_features(game_state: GoState):
    """
    Map a game state to a list of features.

    Returns a list of 21 features describing the board state.

    Some useful functions from game_state include:
        game_state.size: size of the board
        get_pieces_coordinates(player_index): get coordinates of all pieces of a player (0 or 1)
        get_pieces_array(player_index): get a 2D array of pieces of a player (0 or 1)
        get_board(): get a 2D array of the board with 4 channels

    Input:
        game_state: GoState to encode into a fixed size list of features
    Output:
        features: list of 21 features
    """

    # Handle dict input from dataset
    if isinstance(game_state, dict):
        board = game_state['board']
        board_size = game_state['size']
        current_player = game_state['player_to_move']
        opponent_player = 1 - current_player
        terminal_val = 0
        current_player_pieces = board[current_player]
        opponent_player_pieces = board[opponent_player]
        empty_spaces = board[2]
        current_player_coordinates = game_state['black_pieces'] if current_player == 0 else game_state['white_pieces']
        opponent_player_coordinates = game_state['white_pieces'] if current_player == 0 else game_state['black_pieces']

    # Handle GoState input during actual gameplay
    else:
        board_size = game_state.size

        # Return early for terminal states
        if game_state.is_terminal():
            terminal_val = game_state.terminal_value()[0]
            return [terminal_val] + [0] * (FEATURE_SIZE - 1)

        current_player = game_state.player_to_move()
        opponent_player = 1 - current_player
        terminal_val = 0
        current_player_pieces = game_state.get_pieces_array(current_player)
        opponent_player_pieces = game_state.get_pieces_array(opponent_player)
        empty_spaces = game_state.get_empty_spaces()
        current_player_coordinates = game_state.get_pieces_coordinates(current_player)
        opponent_player_coordinates = game_state.get_pieces_coordinates(opponent_player)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # --- Stone counts ---
    current_stones = int(current_player_pieces.sum())
    opponent_stones = int(opponent_player_pieces.sum())
    stone_difference = current_stones - opponent_stones

    # --- Liberty counts ---
    current_player_liberties = 0
    opponent_player_liberties = 0

    for coord in current_player_coordinates:
        row, col = coord
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if empty_spaces[nr, nc]:
                    current_player_liberties += 1

    for coord in opponent_player_coordinates:
        row, col = coord
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if empty_spaces[nr, nc]:
                    opponent_player_liberties += 1

    liberty_difference = current_player_liberties - opponent_player_liberties

    # --- Board coverage ---
    current_player_coverage = current_stones / (board_size * board_size)
    opponent_player_coverage = opponent_stones / (board_size * board_size)
    total_board_coverage = current_player_coverage + opponent_player_coverage

    # --- Center control ---
    # Stones closer to center score higher (center = board_size // 2)
    center = board_size // 2
    current_center_control = 0
    opponent_center_control = 0

    for coord in current_player_coordinates:
        row, col = coord
        dist = abs(row - center) + abs(col - center)
        current_center_control += max(0, center - dist)

    for coord in opponent_player_coordinates:
        row, col = coord
        dist = abs(row - center) + abs(col - center)
        opponent_center_control += max(0, center - dist)

    center_control_diff = current_center_control - opponent_center_control

    # --- Stones at risk ---
    # A stone is "at risk" if it is adjacent to an opponent stone
    current_at_risk = 0
    opponent_at_risk = 0

    for coord in current_player_coordinates:
        row, col = coord
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if opponent_player_pieces[nr, nc]:
                    current_at_risk += 1
                    break

    for coord in opponent_player_coordinates:
        row, col = coord
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if current_player_pieces[nr, nc]:
                    opponent_at_risk += 1
                    break

    at_risk_diff = current_at_risk - opponent_at_risk

    # --- Edge stones ---
    # Stones on the edge of the board are generally weaker (fewer liberties)
    current_edge = sum(
        1 for r, c in current_player_coordinates
        if r == 0 or r == board_size - 1 or c == 0 or c == board_size - 1
    )
    opponent_edge = sum(
        1 for r, c in opponent_player_coordinates
        if r == 0 or r == board_size - 1 or c == 0 or c == board_size - 1
    )
    edge_diff = current_edge - opponent_edge

    # --- Game phase ---
    # Fraction of empty spaces remaining (0 = full board, 1 = empty board)
    empty_count = float(empty_spaces.sum()) / (board_size * board_size)

    features = [
        terminal_val,                   # 0: is game over and who won
        current_player,                 # 1: whose turn it is (0=black, 1=white)
        stone_difference,               # 2: current - opponent stone count
        current_stones,                 # 3: raw stone count for current player
        opponent_stones,                # 4: raw stone count for opponent
        current_player_liberties,       # 5: total liberties for current player
        opponent_player_liberties,      # 6: total liberties for opponent
        liberty_difference,             # 7: liberty difference
        current_player_coverage,        # 8: board fraction covered by current player
        opponent_player_coverage,       # 9: board fraction covered by opponent
        total_board_coverage,           # 10: total board occupation fraction
        current_center_control,         # 11: center proximity score for current player
        opponent_center_control,        # 12: center proximity score for opponent
        center_control_diff,            # 13: center control difference
        current_at_risk,                # 14: current player stones adjacent to opponent
        opponent_at_risk,               # 15: opponent stones adjacent to current player
        at_risk_diff,                   # 16: at-risk difference
        current_edge,                   # 17: current player edge stones
        opponent_edge,                  # 18: opponent edge stones
        edge_diff,                      # 19: edge stone difference
        empty_count,                    # 20: game phase indicator
    ]

    return features


class GoProblemSimpleHeuristic(HeuristicGoProblem):
    def __init__(self, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)

    def heuristic(self, state, player_index):
        """
        Simple heuristic comparing stone counts.
        Returns value from BLACK's perspective: positive = good for BLACK, negative = good for WHITE.
        """
        black_stones = len(state.get_pieces_coordinates(BLACK))
        white_stones = len(state.get_pieces_coordinates(WHITE))
        return black_stones - white_stones

    def __str__(self) -> str:
        return "Simple Heuristic"


class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model

    def encoding(self, state):
        """
        Get encoding of state (convert state to features).

        Input:
            state: GoState to encode into a fixed size list of features
        Output:
            features: list of features
        """
        return get_features(state)

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state using the learned model.

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        features = self.encoding(state)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        value = self.model(features_tensor)
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"