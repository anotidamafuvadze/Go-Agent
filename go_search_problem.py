from abc import ABC, abstractmethod
from random import random
from typing import Sequence, Set, Type, Tuple, List, Optional, Union, Any
import numpy as np
from adversarial_search_problem import AdversarialSearchProblem, GameState
import copy
import go_utils

Action = int
Coordinate = Tuple[int, int]

DEFAULT_SIZE = 5

class GoState(GameState):
    """
    A state of the game of Go.
    
    This class wraps either a pyspiel state or a PyGo state to provide a single
    interface for Go game states. It includes methods for accessing board state,
    legal moves, and game status.
    
    """
    def __init__(self, pyspiel_state: Any, player_to_move: int = 0) -> None:
        """
        Initialize GoState with pyspiel or PyGo as backend Go engine.
        
        The initial state is created with a call to create_go_game() in go_utils.py.
        Every other state will be generated from applying actions to the initial state.
        This essentially functions as a wrapper class to convert pyspiel/PyGo game states
        to the AdversarialSearchProblem interface used by the search algorithms.

        Args:
            pyspiel_state: pyspiel state or PyGo state of the game
            player_to_move: player to move (0=BLACK, 1=WHITE)
        """
        self.internal_state = pyspiel_state
        # Calculate board size from observation tensor length
        # Tensor has 4 channels (black, white, empty, turn) so total length = 4 * size^2
        self.size = int(np.sqrt(len(pyspiel_state.observation_tensor()) / 4))


    def player_to_move(self) -> int:
        """
        Get the current player to move.
        
        Returns:
            Player to move: 0 for BLACK, 1 for WHITE
        """
        return self.internal_state.current_player()

    def get_board(self) -> np.ndarray:
        """
        Return the current board as a 4-channel numpy array.
        
        The board has shape (4, size, size) with the following channels:
        - Channel 0: BLACK pieces (1 where black stones are, 0 elsewhere)
        - Channel 1: WHITE pieces (1 where white stones are, 0 elsewhere) 
        - Channel 2: EMPTY spaces (1 where board is empty, 0 elsewhere)
        - Channel 3: Current player (0 for BLACK's turn, 1 for WHITE's turn)

        This is the standard observation tensor format used by pyspiel
        
        Returns:
            Board representation as numpy array of shape (4, size, size)
        """
        return np.array(self.internal_state.observation_tensor(0)).reshape(-1, self.size, self.size)

    def terminal_value(self) -> float:
        """
        Return the terminal value of the game
        
        Returns:
            List of terminal values: [black_value, white_value]
            - 1 if BLACK wins
            - -1 if WHITE wins
        """
        return self.internal_state.returns()

    def clone(self) -> 'GoState':
        """
        Create a deep copy of the current game state.
        
        This is useful for search algorithms that need to explore different
        move sequences without affecting the original game state. Each search
        branch gets its own copy to work with.
        
        Returns:
            A new GoState instance that is a complete copy of this state
        """
        return GoState(self.internal_state.clone(), self.internal_state.current_player())

    def is_terminal_state(self) -> bool:
        """
        Check if the game is in a terminal state.
        
        A Go game terminates when:
        - Both players pass consecutively
        - No legal moves remain
        - Game reaches maximum move limit

        Returns:
            True if the game is terminal, False otherwise
        """
        return self.internal_state.is_terminal()
    
    def is_terminal(self) -> bool:
        """Alias for is_terminal_state for compatibility."""
        return self.is_terminal_state()

    def legal_actions(self) -> Sequence[Action]:
        """
        Return all legal actions available in the current state.
        
        Actions are represented as integers where:
        - 0 to (size^2 - 1): Board positions
        - size^2: Pass action
        
        For action conversion to coordinates, use action_index_to_coord().
        
        Returns:
            Sequence of legal action indices
        """
        return self.internal_state.legal_actions()

    def apply_action(self, action: Action) -> None:
        """
        Apply an action to update the internal game state.
        
        This method modifies the current state in-place. For search algorithms,
        it's better to use GoProblem.transition() which creates a new state.
        
        Args:
            action: Action index (0 to size^2-1 for moves, size^2 for pass)
        """
        self.internal_state.apply_action(action)

    def get_pieces_coordinates(self, player_index: int) -> np.ndarray:
        """
        Get board coordinates of all stones for the specified player.
        
        Args:
            player_index: 0 for BLACK, 1 for WHITE
            
        Returns:
            Array of shape (n, 2) containing (row, col) coordinates of all stones
            for the specified player
        """
        player_board = np.array(self.internal_state.observation_tensor(
            0)).reshape((-1, self.size, self.size))[player_index]
        return np.argwhere(player_board == 1)

    def get_pieces_array(self, player_index: int) -> np.ndarray:
        """
        Get a binary mask array showing stone positions for the specified player.
        
        Args:
            player_index: 0 for BLACK, 1 for WHITE
            
        Returns:
            2D numpy array of shape (size, size) with 1s where the player has
            stones and 0s elsewhere
        """
        
        player_board = np.array(self.internal_state.observation_tensor(
            0)).reshape((-1, self.size, self.size))[player_index]
        return player_board

    def get_empty_spaces(self) -> np.ndarray:
        """
        Get a binary mask array showing empty positions on the board.
        
        Returns:
            2D numpy array of shape (size, size) with 1s where the board is
            empty and 0s where there are stones
        """
        # Extract the empty channel from the observation tensor and reshape to 2D
        return np.array(self.internal_state.observation_tensor()).reshape(-1, self.size, self.size)[2]
    
    def observation_tensor(self) -> List[float]:
        """
        Return the observation tensor representation of the current state.
        
        Returns:
            Flattened observation tensor as list of floats
        """
        return self.internal_state.observation_tensor()
    
    def action_index_to_coord(self, action: Action) -> Coordinate:
        """
        Convert an action index to board coordinates.
        
        Args:
            action: Action index (0 to size^2-1)
            
        Returns:
            Tuple (row, col) representing the board position
            
        Note:
            For the pass action (action = size^2), this will return coordinates
            outside the board bounds. Check if action is a pass before calling.
        """
        return (action // self.size, action % self.size)  # (row, col)

    def __repr__(self) -> str:
        return str(self.internal_state)


class GoProblem(AdversarialSearchProblem[GoState, Action]):
    """
    Go game search problem implementing the AdversarialSearchProblem interface.
    """

    def __init__(self, size: int = DEFAULT_SIZE, state: Optional[Any] = None, 
                 player_to_move: int = 0) -> None:
        """
        Create a new Go search problem.
        
        Args:
            size: Board size (typically 5, 9, or 19)
            state: Existing game state, or None to create a new game
            player_to_move: Initial player (0=BLACK, 1=WHITE) - currently unused
        """
        if state is None:
            game_state = go_utils.create_go_game(size)
        else:
            game_state = state
        self.start_state = GoState(game_state, player_to_move)

    def get_available_actions(self, state: GoState) -> List[Action]:
        """
        Get all legal actions available in the given state.
        
        Actions are encoded as integers:
        - 0 to (size^2 - 1): Board positions in row-major order
        - size^2: Pass action
        
        To convert action indices to coordinates, use:
        (row, col) = (action // size, action % size)
        
        Args:
            state: Current game state
            
        Returns:
            List of legal action indices
        """
        actions = list(state.legal_actions())
        np.random.shuffle(actions) # Shuffle actions 
        
        return actions

    def transition(self, state: GoState, action: Action) -> GoState:
        """
        Apply an action to a state and return the resulting new state.
        
        This method creates a copy of the current state and applies the action
        to it, leaving the original state unchanged.
        
        Args:
            state: Current game state
            action: Action to apply (0 to size^2 for moves, size^2 for pass)
            
        Returns:
            New game state after applying the action
        """
        new_state = state.clone()
        new_state.apply_action(action)
        return new_state

    def is_terminal_state(self, state: GoState) -> bool:
        """
        Check if the given state is terminal (game over).
        
        A Go game is terminal when:
        - Both players have passed consecutively
        - No legal moves remain (board full)
        
        Args:
            state: Game state to check
            
        Returns:
            True if the game is over, False otherwise
        """
        return state.is_terminal_state()

    def get_result(self, state: GoState) -> float:
        """
        Get the game result from the perspective of the first player (BLACK).
        
        This method should only be called on terminal states. The result
        represents the outcome for BLACK:
        - +1.0: BLACK wins
        - -1.0: WHITE wins
        
        Args:
            state: Terminal game state
            
        Returns:
            Game result from BLACK's perspective
        """
        return state.terminal_value()[0]

    def action_index_to_string(self, action: Action) -> str:
        """
        Convert an action index to a human-readable string representation.
        
        Args:
            action: Action index to convert
            
        Returns:
            String representation like "(2, 3)" for board positions or "pass"
        """
        if action == self.start_state.size ** 2:
            return "pass"
        row = action // self.start_state.size
        col = action % self.start_state.size
        return f"({row}, {col})"
    
class HeuristicGoProblem(GoProblem, ABC):
    """
    Abstract base class for Go search problems with a heuristic function.
    
    This class extends GoProblem by adding an abstract heuristic method,
    which is crucial for search algorithms like Minimax and Alpha-Beta
    that need to evaluate non-terminal states.
    """
    @abstractmethod
    def heuristic(self, state: GoState, player_index: int) -> float:
        """
        Evaluate the given game state using a heuristic function.
        
        This function estimates the value of a non-terminal state for a given player.
        A higher value indicates a better state for the player.
        
        Args:
            state: The game state to evaluate
            player_index: The index of the player for whom to evaluate the state
                          (0 for BLACK, 1 for WHITE)
                          
        Returns:
            A numerical score representing the estimated value of the state.
        """
        pass