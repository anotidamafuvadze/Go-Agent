import time
import argparse
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass, field
import numpy as np
import pygame
import tqdm

from go_search_problem import GoProblem, GoState
from go_gui import GoGUI
from agents import *
from agents import create_value_agent_from_model

# Initialize pygame once
pygame.init()
CLOCK = pygame.time.Clock()

# Player constants
BLACK = MAXIMIZER = 0
WHITE = MINIMIZER = 1


@dataclass
class GameStats:
    """Statistics for a single game."""
    winner: int
    player1_time_remaining: float
    player2_time_remaining: float
    player1_move_times: List[float] = field(default_factory=list)
    player2_move_times: List[float] = field(default_factory=list)
    
    @property
    def player1_avg_time(self) -> float:
        return np.mean(self.player1_move_times) if self.player1_move_times else 0.0
    
    @property
    def player2_avg_time(self) -> float:
        return np.mean(self.player2_move_times) if self.player2_move_times else 0.0
    
    @property
    def player1_max_time(self) -> float:
        return np.max(self.player1_move_times) if self.player1_move_times else 0.0
    
    @property
    def player2_max_time(self) -> float:
        return np.max(self.player2_move_times) if self.player2_move_times else 0.0


@dataclass
class TournamentStats:
    """Statistics for a tournament of multiple games."""
    player1_wins: int = 0
    player2_wins: int = 0
    player1_wins_as_black: int = 0
    player2_wins_as_black: int = 0
    player1_total_time: float = 0.0
    player2_total_time: float = 0.0
    player1_min_time_remaining: float = float('inf')
    player2_min_time_remaining: float = float('inf')
    player1_max_move_time: float = 0.0
    player2_max_move_time: float = 0.0
    games_played: int = 0
    
    def add_game(self, stats: GameStats, player1_is_black: bool) -> None:
        """Add results from a single game to tournament stats."""
        self.games_played += 1
        
        # Track wins
        if stats.winner == 1:  # Player 1 wins
            self.player1_wins += 1
            if player1_is_black:
                self.player1_wins_as_black += 1
        else:  # Player 2 wins
            self.player2_wins += 1
            if not player1_is_black:
                self.player2_wins_as_black += 1
        
        # Track timing stats
        self.player1_total_time += stats.player1_avg_time
        self.player2_total_time += stats.player2_avg_time
        self.player1_min_time_remaining = min(self.player1_min_time_remaining, stats.player1_time_remaining)
        self.player2_min_time_remaining = min(self.player2_min_time_remaining, stats.player2_time_remaining)
        self.player1_max_move_time = max(self.player1_max_move_time, stats.player1_max_time)
        self.player2_max_move_time = max(self.player2_max_move_time, stats.player2_max_time)
    
    def print_summary(self, agent1: Any, agent2: Any) -> None:
        """Print a summary of tournament results."""
        print(f"Tournament Results")
        print(f"Games played: {self.games_played}")
        print(f"{agent1} wins: {self.player1_wins} ({self.player1_wins/self.games_played:.1%})")
        print(f"{agent2} wins: {self.player2_wins} ({self.player2_wins/self.games_played:.1%})")
        print(f"{agent1} wins as BLACK: {self.player1_wins_as_black}")
        print(f"{agent2} wins as BLACK: {self.player2_wins_as_black}")
        print(f"{agent1} avg move time: {self.player1_total_time/self.games_played:.3f}s")
        print(f"{agent2} avg move time: {self.player2_total_time/self.games_played:.3f}s")
        print(f"{agent1} min time remaining: {self.player1_min_time_remaining:.1f}s")
        print(f"{agent2} min time remaining: {self.player2_min_time_remaining:.1f}s")


class GameRunner:
    """Main class for running Go games with various modes and configurations."""
    
    def __init__(self, board_size: int = 5, time_limit: float = 15.0, 
                 time_increment: float = 1.0, hard_time_cutoff: bool = True):
        """
        Initialize the game runner.
        
        Args:
            board_size: Size of the Go board
            time_limit: Initial time limit per player in seconds
            time_increment: Time added per move in seconds
            hard_time_cutoff: If True, end game when player runs out of time
        """
        self.board_size = board_size
        self.time_limit = time_limit
        self.time_increment = time_increment
        self.hard_time_cutoff = hard_time_cutoff
    
    def play_single_game(self, agent1: Any, agent2: Any) -> GameStats:
        """
        Play a single game between two agents.
        
        Args:
            agent1: First agent (plays BLACK)
            agent2: Second agent (plays WHITE)
            
        Returns:
            GameStats object with game results and statistics
        """
        game = GoProblem(size=self.board_size)
        state = game.start_state
        
        # Initialize timing
        player_times = [self.time_limit, self.time_limit]
        move_times = [[], []]
        
        # Game loop
        current_player = 0
        agents = [agent1, agent2]
        
        while not game.is_terminal_state(state):
            # print(state)
            start_time = time.time()
            
            # Get move from current agent
            action = agents[current_player].get_move(state.clone(), player_times[current_player])
            move_duration = time.time() - start_time
            
            # Update timing
            player_times[current_player] -= move_duration
            move_times[current_player].append(move_duration)
            
            # Check time limit
            if player_times[current_player] <= 0 and self.hard_time_cutoff:
                print(f"Player {current_player + 1} ran out of time and forfeits the game!")
                if current_player == 0:
                    winner = -1
                else:
                    winner = 1
                return GameStats(
                    winner=winner,
                    player1_time_remaining=player_times[0],
                    player2_time_remaining=player_times[1],
                    player1_move_times=move_times[0],
                    player2_move_times=move_times[1]
                )
            
            # Add time increment and apply move
            player_times[current_player] += self.time_increment
            state = game.transition(state, action)
            current_player = 1 - current_player
        
        # Game finished normally
        winner = game.get_result(state)
        return GameStats(
            winner=winner,
            player1_time_remaining=player_times[0],
            player2_time_remaining=player_times[1],
            player1_move_times=move_times[0],
            player2_move_times=move_times[1]
        )
    
    def play_tournament(self, agent1: Any, agent2: Any, num_games: int = 10, 
                       verbose: bool = True) -> TournamentStats:
        """
        Play a tournament between two agents with color alternation.
        
        Args:
            agent1: First agent
            agent2: Second agent
            num_games: Total number of games to play (must be even)
            verbose: If True, show progress and print results
            
        Returns:
            TournamentStats with complete tournament results
        """
        if num_games % 2 != 0:
            num_games += 1  # Ensure even number for fair color distribution
        
        stats = TournamentStats()
        
        # Progress bar if verbose
        iterator = range(num_games // 2)
        if verbose:
            iterator = tqdm.tqdm(iterator, desc="Playing tournament")
        
        for _ in iterator:
            # Game 1: agent1 as BLACK, agent2 as WHITE
            agent1.reset()
            agent2.reset()
            game_stats = self.play_single_game(agent1, agent2)
            stats.add_game(game_stats, player1_is_black=True)
            
            # Game 2: agent2 as BLACK, agent1 as WHITE
            game_stats = self.play_single_game(agent2, agent1)
            # Flip winner perspective since agents are swapped
            game_stats.winner = -game_stats.winner
            # Swap timing stats to maintain agent1/agent2 perspective
            game_stats.player1_time_remaining, game_stats.player2_time_remaining = \
                game_stats.player2_time_remaining, game_stats.player1_time_remaining
            game_stats.player1_move_times, game_stats.player2_move_times = \
                game_stats.player2_move_times, game_stats.player1_move_times
            stats.add_game(game_stats, player1_is_black=False)
        
        if verbose:
            stats.print_summary(agent1, agent2)
        
        return stats
    
    def play_vs_human_text(self, agent: Any) -> None:
        """Play against a human using text interface."""
        print(f"Starting game on {self.board_size}x{self.board_size} board...")
        game = GoProblem(size=self.board_size)
        state = game.start_state
        
        while not game.is_terminal_state(state):
            # Agent move
            agent_action = agent.get_move(state.clone(), self.time_limit)
            print(f"Agent plays: {game.action_index_to_string(agent_action)}")
            state = game.transition(state, agent_action)
            
            if game.is_terminal_state(state):
                break
            
            # Display board and get human move
            self._print_board(state)
            human_action = self._get_human_move_text(state, game)
            print(f"Human plays: {game.action_index_to_string(human_action)}")
            state = game.transition(state, human_action)
        
        # Game over
        result = game.get_result(state)
        print("Game over!")
        if result == 1:
            print("Agent wins!")
        elif result == -1:
            print("Human wins!")
        else:
            print("Draw!")
    
    def play_vs_human_gui(self, agent: Any) -> None:
        """Play against a human using GUI interface."""
        print(f"Starting GUI game on {self.board_size}x{self.board_size} board...")
        game = GoProblem(size=self.board_size)
        state = game.start_state
        gui = GoGUI(game)
        
        while not game.is_terminal_state(state):
            # Agent move
            agent_action = agent.get_move(state.clone(), self.time_limit)
            state = game.transition(state, agent_action)
            gui.update_state(state)
            gui.render()
            
            if game.is_terminal_state(state):
                break
            
            # Human move
            human_action = self._get_human_move_gui(state, gui)
            state = game.transition(state, human_action)
            gui.update_state(state)
            gui.render()
            CLOCK.tick(60)
        
        # Game over
        result = game.get_result(state)
        if result == 1:
            print("Agent wins!")
        else:
            print("Human wins!")
    
    def _print_board(self, state: GoState) -> None:
        """Print board state in text format."""
        board = state.get_board()
        size = self.board_size
        
        # Print column headers
        print("  ", end="")
        for i in range(size):
            print(f"{i} ", end="")
        print()
        
        # Print rows
        for i in range(size):
            print(f"{i} ", end="")
            for j in range(size):
                if board[BLACK][i][j] == 1:
                    print("B ", end="")
                elif board[WHITE][i][j] == 1:
                    print("W ", end="")
                else:
                    print(". ", end="")
            print()
        print()
    
    def _get_human_move_text(self, state: GoState, game: GoProblem) -> int:
        """Get human move via text input."""
        legal_actions = state.legal_actions()
        
        while True:
            try:
                if input("Pass? (y/n): ").lower().startswith('y'):
                    action = self.board_size ** 2  # Pass action
                else:
                    row = int(input("Enter row (0-indexed): "))
                    col = int(input("Enter col (0-indexed): "))
                    action = row * self.board_size + col
                
                if action in legal_actions:
                    return action
                else:
                    print("Illegal move! Try again.")
            except (ValueError, IndexError):
                print("Invalid input! Try again.")
    
    def _get_human_move_gui(self, state: GoState, gui: GoGUI) -> int:
        """Get human move via GUI interaction."""
        legal_actions = state.legal_actions()
        
        while True:
            action = gui.get_user_input_action()
            gui.render()
            CLOCK.tick(60)
            
            if action is not None and action in legal_actions:
                return action


def create_agent(agent_type: str, **kwargs) -> Any:
    """
    Factory function to create agents.
    
    Args:
        agent_type: Type of agent ('alphabeta', 'minimax', 'random', etc.)
        **kwargs: Additional parameters (e.g., depth=3)
        
    Returns:
        Agent instance
    """
    agent_map = {
        'alphabeta': lambda: AlphaBetaAgent(depth_cutoff=kwargs.get('depth', 2)),
        'minimax': lambda: MinimaxAgent(depth_cutoff=kwargs.get('depth', 2)),
        'random': lambda: RandomAgent(),
        'greedy': lambda: GreedyAgent(),
        'ids': lambda: IterativeDeepeningAgent(),
        'learned': lambda: create_value_agent_from_model(),
        'mcts': lambda: MCTSAgent(),
        'value_mcts': lambda: get_final_agent_5x5(),
        'value_mcts_9x9': lambda: get_final_agent_9x9(),
        # Add more agents as needed for part 3
    }
    
    if agent_type.lower() not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_map[agent_type.lower()]()


def main():
    parser = argparse.ArgumentParser(description='Go Game Runner')
    
    parser.add_argument('--mode', choices=['gui', 'text', 'vs', 'tournament'], 
                       default='gui', help='Game mode')
    
    parser.add_argument('--agent1', default='alphabeta', help='Agent 1 type')
    parser.add_argument('--agent1-depth', type=int, default=2, help='Agent 1 depth')
    parser.add_argument('--agent2', default='random', help='Agent 2 type')
    parser.add_argument('--agent2-depth', type=int, default=2, help='Agent 2 depth')
    
    # Game settings
    parser.add_argument('--size', type=int, default=5, help='Board size')
    parser.add_argument('--time-limit', type=float, default=15.0, help='Time limit per player')
    parser.add_argument('--time-increment', type=float, default=1.0, help='Time increment per move')
    parser.add_argument('--soft-time', action='store_true', help='Allow time overruns')
    
    # Tournament settings
    parser.add_argument('--games', '--num-games', type=int, default=10, help='Number of tournament games')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create game runner
    runner = GameRunner(
        board_size=args.size,
        time_limit=args.time_limit,
        time_increment=args.time_increment,
        hard_time_cutoff=not args.soft_time
    )
    
    # Create agents
    agent1 = create_agent(args.agent1, depth=args.agent1_depth)
    
    # Run appropriate mode
    if args.mode == 'gui':
        runner.play_vs_human_gui(agent1)
    elif args.mode == 'text':
        runner.play_vs_human_text(agent1)
    else:
        agent2 = create_agent(args.agent2, depth=args.agent2_depth)
        if args.mode == 'vs':
            stats = runner.play_single_game(agent1, agent2)
            print(f"Winner: {stats.winner}")
            print(f"Player 1 time remaining: {stats.player1_time_remaining:.1f}s")
            print(f"Player 2 time remaining: {stats.player2_time_remaining:.1f}s")
        elif args.mode == 'tournament':
            runner.play_tournament(agent1, agent2, args.games, verbose=not args.quiet)


if __name__ == "__main__":
    main()