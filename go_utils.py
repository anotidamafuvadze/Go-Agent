import numpy as np
from typing import Any, List, Dict, Optional, Union, Tuple
import copy
import sys

try:
    import pyspiel
    PYSPIEL_AVAILABLE = True
except ImportError:
    print("pyspiel not found, using PyGo instead")
    PYSPIEL_AVAILABLE = False
    try:
        from pygo.game import Game
        from pygo.utils import Stone
        PYGO_AVAILABLE = True
    except ImportError:
        print("Warning: Neither pyspiel nor PyGo is available")
        PYGO_AVAILABLE = False



def create_go_game(size: int) -> Any:
    """
    Create a Go game instance with the specified board size.
    
    This function attempts to use pyspiel first, falling back to PyGo
    if pyspiel is not available. 
    
    The komi (compensation points for White)
    is set based on standard values for different board sizes.
    
    Args:
        size: Board size (typically 5, 9, or 19)
        
    Returns:
        Game state object (pyspiel state or PyGoInternalState)
        
    Raises:
        ImportError: If neither pyspiel nor PyGo is available
    """
    # Set komi (compensation points for White) based on board size
    if size == 5:
        komi = 0.5  # Small board, minimal komi
    elif size == 9:
        komi = 5.5  # Standard 9x9 komi
    else:
        komi = 7.5  # Standard 19x19 komi (also used for other sizes)
    
    if PYSPIEL_AVAILABLE:
        try:
            game = pyspiel.load_game("go", {"board_size": size, "komi": komi})
            state = game.new_initial_state()
            return state
        except Exception as e:
            print(f"Failed to create pyspiel game: {e}")
    
    if PYGO_AVAILABLE:
        return PyGoInternalState(size, komi)
    
    raise ImportError("Neither pyspiel nor PyGo is available for Go game creation")


class PyGoInternalState:
    """
    Implementation of internal Go game state matching pyspiel's interface.
    
    This class provides a fallback Go implementation when pyspiel is not available,
    using the PyGo library. It mimics the pyspiel interface to ensure compatibility
    with the rest of the Go engine.
    
    Attributes:
        config: Configuration dictionary for the PyGo game
        game: PyGo Game instance
        size: Board size
        consecutive_passes: Count of consecutive passes
        _current_player: Current player (0=BLACK, 1=WHITE)
    """
    def __init__(self, size: int = 5, komi: float = 7.5) -> None:
        """
        Initialize a new Go game state using PyGo.
        
        Args:
            size: Board size (typically 5, 9, or 19)
            komi: Compensation points for White player
        """
        if not PYGO_AVAILABLE:
            raise ImportError("PyGo is required but not available")
            
        self.config = {
            'board_size': size,
            'black_stone': '●',
            'white_stone': '○', 
            'enable_self_destruct': False,
            'komi': komi
        }
        self.game = Game(self.config)
        self.size = size
        self.consecutive_passes = 0
        self._current_player = 0  # 0 for BLACK, 1 for WHITE
        
    def current_player(self) -> int:
        return self._current_player

    def observation_tensor(self, player: int = 0) -> List[int]:
        """
        Get the observation tensor for the current state.
        
        The tensor has 4 channels (BLACK, WHITE, EMPTY, TURN) flattened into
        a 1D list, matching the pyspiel interface.
        
        Args:
            player: Player perspective (currently unused)
            
        Returns:
            Flattened observation tensor as a list
        """
        board = np.array(self.game.board)
        tensor = np.zeros((4, self.size, self.size), dtype=np.int32)

        black_channel = (board == Stone.BLACK)
        tensor[0] = black_channel
        
        white_channel = (board == Stone.WHITE)
        tensor[1] = white_channel
        
        empty_channel = (board == Stone.EMPTY)
        tensor[2] = empty_channel
        
        tensor[3] += self._current_player

        return tensor.flatten()

    def returns(self) -> List[float]:
        """
        Get the game result from each player's perspective.
        
        Returns:
            List of returns: [black_return, white_return]
            - [1, -1] if BLACK wins
            - [-1, 1] if WHITE wins  
            - [0, 0] if game is not terminal
        """
        if not self.is_terminal():
            return [0, 0]

        scores = self.game.get_scores()
        black_score = scores[Stone.BLACK]
        white_score = scores[Stone.WHITE]
        
        if black_score > white_score:
            return [1, -1]
        else:
            return [-1, 1]

    def clone(self) -> 'PyGoInternalState':
        """
        Create a deep copy of this game state.
        
        Returns:
            New PyGoInternalState instance that is a copy of this state
        """
        return copy.deepcopy(self)

    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state.
        
        The game ends when:
        - Two consecutive passes have occurred
        - The board is completely full (rare)
        
        Returns:
            True if the game is over, False otherwise
        """
        board = np.array(self.game.board)
        empty_count = np.sum(board == Stone.EMPTY)
        return self.consecutive_passes >= 2 or empty_count == 0

    def legal_actions(self) -> List[int]:
        """
        Get all legal actions in the current state.
        
        Actions are encoded as integers:
        - 0 to (size^2 - 1): Board positions in row-major order
        - size^2: Pass action
        
        Returns:
            List of legal action indices
        """
        if self.is_terminal():
            return []

        legal = []
        board = np.array(self.game.board)
        empty_positions = np.where(board == Stone.EMPTY)
        stone = Stone.BLACK if self._current_player == 0 else Stone.WHITE
        
        # Try each empty position to see if it's a legal move
        for y, x in zip(empty_positions[0], empty_positions[1]):
            action = y * self.size + x
            try:
                # Test if this move is legal by trying it on a copy
                test_state = self.clone()
                if stone == Stone.BLACK:
                    test_state.game.place_black(y, x)
                else:
                    test_state.game.place_white(y, x)
                legal.append(action)
            except Exception:
                # Move is illegal (e.g., suicide, ko rule violation)
                continue
        
        # Pass is always legal (unless game is already terminal)
        pass_move = self.size * self.size
        legal.append(pass_move)
        
        return legal

    def apply_action(self, action: int) -> None:
        """
        Apply an action to modify this game state.
        
        Args:
            action: Action to apply
                   - 0 to (size^2 - 1): Place stone at board position
                   - size^2: Pass
        """
        if self.is_terminal():
            return
            
        pass_move = self.size * self.size
        
        if action == pass_move:
            # Player passes
            self.consecutive_passes += 1
            self.game.pass_turn()
        else:
            # Player places a stone
            y = action // self.size
            x = action % self.size
            
            if self._current_player == 0:  # BLACK's turn
                self.game.place_black(y, x)
            else:  # WHITE's turn
                self.game.place_white(y, x)
            self.consecutive_passes = 0  # Reset pass counter
        
        # Switch to the other player
        self._current_player = 1 - self._current_player