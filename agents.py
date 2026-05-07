from pyexpat import model
import random
import math
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import pickle

from go_search_problem import GoProblem, GoState, Action, HeuristicGoProblem
from heuristic_go_problems import GoProblemLearnedHeuristic, GoProblemSimpleHeuristic, get_features
from models import PolicyNetwork, ValueNetwork, load_model

MAXIMIZER = 0
MIMIZER = 1


def create_value_agent_from_model():
    """
    Create agent object from saved model. 
    This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.

    In the game_runner file, there is a factory function that will call this function to create an agent.
    You can run games with your agent against other agents by running game_runner.py with the appropriate command line arguments.
    """
    model_path = "value_model.pt"
    feature_size = 11
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    learned_agent = GreedyAgent(heuristic_search_problem)

    return learned_agent


class GameAgent(ABC):
    """Abstract base class for all Go game agents."""
    
    @abstractmethod
    def get_move(self, state: GoState, time_limit: float) -> Action:
        """Get the best move for the given state within the time limit.
        
        Args:
            state: Current game state
            time_limit: Maximum time in seconds to spend on this move
            
        Returns:
            Action index representing the chosen move
        """
        pass

    def reset(self):
        """Reset any internal state of the agent if necessary.
            Called after a new game is started.
        """
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem: HeuristicGoProblem = GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)

#############################################
# 
#
# Part 1: Basic Adversarial Search Algorithms
#
#
#############################################


class MinimaxAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        MinimaxAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        action, stats = self.minimax(game_state, self.depth)

        if action is None:
            action = random.choice(self.search_problem.get_available_actions(game_state))

        return action
    
    def minimax(self, game_state: GoState, cutoff_depth: float = float('inf')) -> Tuple[Optional[Action], Dict[str, int]]:
        """
        Implements the minimax algorithm for adversarial search.

        Args:
            game_state: current game state
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).

        Returns:
            Tuple containing:
                - Best action to take from the current state
                - Dictionary with search statistics including 'states_expanded'
        """
        best_action = None
        stats = {
            'states_expanded': 0
        }

        player = game_state.player_to_move()

        if player == 0:
            best_action, _, stats = self.max_value(game_state, cutoff_depth, stats)
        else:
            best_action, _, stats = self.min_value(game_state, cutoff_depth, stats)
        
        return best_action, stats
        

    def max_value(self, game_state: GoState, depth: float = float('inf'), stats: Optional[Dict[str, int]] = None) -> Tuple[Optional[Action], float, Dict[str, int]]:
        
        if stats is None:
            stats = {'states_expanded': 0}

        if self.search_problem.is_terminal_state(game_state):
            terminal_value = self.search_problem.get_result(game_state) * math.inf
            return None, terminal_value, stats
        
        if (depth == 0):
            return None, self.search_problem.heuristic(game_state, game_state.player_to_move()), stats
        
        best_action = None
        max_value = float('-inf')
        children = self.search_problem.get_available_actions(game_state)
        stats['states_expanded'] += 1

        for action in children:
            child_state = self.search_problem.transition(game_state, action)
            _, value, stats = self.min_value(child_state, depth - 1, stats)
            if value > max_value:
                max_value = value
                best_action = action

        return best_action, max_value, stats


    def min_value(self, game_state: GoState, depth: float = float('inf'), stats: Optional[Dict[str, int]] = None) -> Tuple[Optional[Action], float, Dict[str, int]]:
        
        if stats is None:
            stats = {'states_expanded': 0}
 
        if self.search_problem.is_terminal_state(game_state):
            terminal_value = self.search_problem.get_result(game_state) * math.inf
            return None, terminal_value, stats
        
        if (depth == 0):
            return None, self.search_problem.heuristic(game_state, game_state.player_to_move()), stats
        
        best_action = None
        min_value = float('inf')
        children = self.search_problem.get_available_actions(game_state)
        stats['states_expanded'] += 1

        for action in children:
            child_state = self.search_problem.transition(game_state, action)
            _, value, stats = self.max_value(child_state, depth - 1, stats)
            if value < min_value:
                min_value = value
                best_action = action

        return best_action, min_value, stats


class AlphaBetaAgent(GameAgent):
  
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        action, stats = self.alpha_beta(game_state, self.depth)

        if action is None:
            action = random.choice(self.search_problem.get_available_actions(game_state))

        return action
    
    def alpha_beta(self, game_state: GoState, cutoff_depth: float = float('inf')) -> Tuple[Optional[Action], Dict[str, int]]:
        best_action = None
        stats = {
            'states_expanded': 0
        }

        player = game_state.player_to_move()

        if player == 0:
            best_action, _, stats = self.max_value_pruning(game_state, cutoff_depth, stats, alpha=float('-inf'), beta=float('inf'))
        else:
            best_action, _, stats = self.min_value_pruning(game_state, cutoff_depth, stats, alpha=float('-inf'), beta=float('inf'))
        
        return best_action, stats
        

    def max_value_pruning(self, game_state: GoState, depth: float = float('inf'), stats: Optional[Dict[str, int]] = None, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[Optional[Action], float, Dict[str, int]]:

        if stats is None:
            stats = {'states_expanded': 0}

        if self.search_problem.is_terminal_state(game_state):
            terminal_value = self.search_problem.get_result(game_state) * math.inf
            return None, terminal_value, stats
        
        if (depth == 0):
            return None, self.search_problem.heuristic(game_state, game_state.player_to_move()), stats
        
        best_action = None
        max_value = float('-inf')
        children = self.search_problem.get_available_actions(game_state)
        stats['states_expanded'] += 1

        for action in children:
            child_state = self.search_problem.transition(game_state, action)
            _, value, stats = self.min_value_pruning(child_state, depth - 1, stats, alpha, beta)
            if value > max_value:
                max_value = value
                best_action = action

            if max_value >= alpha:
                alpha = max_value
            
            if beta <= alpha:
                break

        return best_action, max_value, stats


    def min_value_pruning(self, game_state: GoState, depth: float = float('inf'), stats: Optional[Dict[str, int]] = None, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[Optional[Action], float, Dict[str, int]]:

        if stats is None:
            stats = {'states_expanded': 0}

        if self.search_problem.is_terminal_state(game_state):
            terminal_value = self.search_problem.get_result(game_state) * math.inf
            return None, terminal_value, stats
        
        if (depth == 0):
            return None, self.search_problem.heuristic(game_state, game_state.player_to_move()), stats
        
        best_action = None
        min_value = float('inf')
        children = self.search_problem.get_available_actions(game_state)
        stats['states_expanded'] += 1

        for action in children:
            child_state = self.search_problem.transition(game_state, action)
            _, value, stats = self.max_value_pruning(child_state, depth - 1, stats, alpha, beta)
            if value < min_value:
                min_value = value
                best_action = action

            if min_value <= beta:
                beta = min_value

            if beta <= alpha:
                break

        return best_action, min_value, stats

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)


################################################
#
# Part 2: Advanced Adversarial Search Algorithms
#
################################################

class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        return self.iterative_deepening(game_state, time_limit)

    def iterative_deepening(self, game_state, time_limit=None):
        best_action = None
        depth = 1
        start_time = time.time()
        per_move_limit = min(time_limit * 0.1, 1.0) - 0.05

        while self.time_remaining(start_time, per_move_limit):
            action = self.alpha_beta_time_limited(game_state, cutoff_depth=depth, start_time=start_time, time_limit=per_move_limit)
            if action is not None:
                best_action = action
            depth += 1

        return best_action
    
    def time_remaining(self, start_time, time_limit):
        return (time.time() - start_time) < time_limit

    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)
    
    def alpha_beta_time_limited(self, game_state: GoState, cutoff_depth: float = float('inf'), start_time: float = float('inf'), time_limit: float = float('inf')) -> Optional[Action]:

        if not self.time_remaining(start_time, time_limit):
            return None
        
        best_action = None

        player = game_state.player_to_move()

        if player == 0:
            best_action, _ = self.max_value_pruning_time_limited(game_state, cutoff_depth, alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=time_limit)
        else:
            best_action, _ = self.min_value_pruning_time_limited(game_state, cutoff_depth, alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=time_limit)
        
        return best_action

    def max_value_pruning_time_limited(self, game_state: GoState, depth: float = float('inf'), alpha: float = float('-inf'), beta: float = float('inf'), start_time: float = float('inf'), time_limit: float = float('inf')) -> Tuple[Optional[Action], float]:

        if not self.time_remaining(start_time, time_limit):
            return None, 0

        if self.search_problem.is_terminal_state(game_state):
            terminal_value = self.search_problem.get_result(game_state) * math.inf
            return None, terminal_value
        
        if (depth == 0):
            return None, self.search_problem.heuristic(game_state, game_state.player_to_move())
        
        best_action = None
        max_value = float('-inf')
        children = self.search_problem.get_available_actions(game_state)

        for action in children:
            child_state = self.search_problem.transition(game_state, action)
            _, value = self.min_value_pruning_time_limited(child_state, depth - 1, alpha, beta, start_time, time_limit)
            if value > max_value:
                max_value = value
                best_action = action

            if max_value >= alpha:
                alpha = max_value
            
            if beta <= alpha:
                break

        return best_action, max_value


    def min_value_pruning_time_limited(self, game_state: GoState, depth: float = float('inf'), alpha: float = float('-inf'), beta: float = float('inf'), start_time: float = float('inf'), time_limit: float = float('inf')) -> Tuple[Optional[Action], float]:

        if not self.time_remaining(start_time, time_limit):
            return None, 0

        if self.search_problem.is_terminal_state(game_state):
            terminal_value = self.search_problem.get_result(game_state) * math.inf
            return None, terminal_value,
        
        if (depth == 0):
            return None, self.search_problem.heuristic(game_state, game_state.player_to_move()),
        
        best_action = None
        min_value = float('inf')
        children = self.search_problem.get_available_actions(game_state)

        for action in children:
            child_state = self.search_problem.transition(game_state, action)
            _, value = self.max_value_pruning_time_limited(child_state, depth - 1, alpha, beta, start_time, time_limit)
            if value < min_value:
                min_value = value
                best_action = action

            if min_value <= beta:
                beta = min_value

            if beta <= alpha:
                break

        return best_action, min_value


class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (accumulated result from simulations, always from black's perspective)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        return hash(self.state)


class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2), debug=False):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c
        self.debug = debug

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm.
        """
        root = MCTSNode(game_state)

        actions = self.search_problem.get_available_actions(game_state)
        if len(actions) == 1:
            if self.debug:
                print(f"[MCTS] Only one legal action: {actions[0]}")
            return actions[0]

        start_time = time.time()
        per_move_limit = max(0.05, time_limit * 0.90 - 0.05)

        iterations = 0
        result_counts = {-1: 0, 0: 0, 1: 0}

        if self.debug:
            print("\n[MCTS] ---------------- NEW MOVE ----------------")
            print(f"[MCTS] Player to move: {game_state.player_to_move()}")
            print(f"[MCTS] Legal actions: {len(actions)}")
            print(f"[MCTS] Given time_limit: {time_limit:.3f}")
            print(f"[MCTS] Using per_move_limit: {per_move_limit:.3f}")

        while self.time_remaining(start_time, per_move_limit):

            node = self.selection(root)

            if not self.search_problem.is_terminal_state(node.state):
                node = self.expansion(node)

            result = self.simulation(node.state)

            if result in result_counts:
                result_counts[result] += 1
            else:
                result_counts[result] = result_counts.get(result, 0) + 1

            self.backpropagation(node, result)
            iterations += 1

        elapsed = time.time() - start_time

        if self.debug:
            print(f"[MCTS] Search finished after {elapsed:.3f}s")
            print(f"[MCTS] Iterations: {iterations}")
            print(f"[MCTS] Root visits: {root.visits}")
            print(f"[MCTS] Root children: {len(root.children)}")
            print(f"[MCTS] Simulation results: {result_counts}")

        if not root.children:
            fallback = random.choice(actions)
            if self.debug:
                print(f"[MCTS] No root children. Returning random fallback: {fallback}")
            return fallback

        root_player = game_state.player_to_move()

        def final_score(child):
            avg_black_value = child.value / child.visits
            if root_player == 0:
                return avg_black_value
            else:
                return -avg_black_value

        best_child = max(root.children, key=final_score)

        if self.debug:
            print("[MCTS] Root child stats:")
            for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
                avg_black_value = child.value / child.visits if child.visits > 0 else 0
                score_for_player = final_score(child)
                print(
                    f"  action={child.action}, "
                    f"visits={child.visits}, "
                    f"value={child.value:.2f}, "
                    f"avg_black_value={avg_black_value:.3f}, "
                    f"score_for_player={score_for_player:.3f}"
                )
            print(f"[MCTS] Chosen action: {best_child.action}")
            print("[MCTS] ------------------------------------------\n")

        return best_child.action

    def selection(self, node):
        """
        Selection step of MCTS algorithm. Select child with highest UCT value until leaf node is reached.
        UCT exploitation term is flipped based on the current player so both players play optimally.
        """
        while len(node.children) > 0:
            actions = self.search_problem.get_available_actions(node.state)

            # If there are still unvisited children, return node for expansion
            if len(node.children) < len(actions):
                return node

            unvisited = [child for child in node.children if child.visits == 0]
            if unvisited:
                return random.choice(unvisited)

            current_player = node.state.player_to_move()
            parent_visits = max(1, node.visits)

            def uct(child):
                avg_black_value = child.value / child.visits
                # Flip exploitation for white since value is stored from black's perspective
                if current_player == 0:
                    exploitation = avg_black_value
                else:
                    exploitation = -avg_black_value

                exploration = self.c * np.sqrt(np.log(parent_visits) / child.visits)
                return exploitation + exploration

            uct_values = [uct(child) for child in node.children]
            max_uct = max(uct_values)
            best_children = [
                child for child, uct_value in zip(node.children, uct_values)
                if uct_value == max_uct
            ]
            node = random.choice(best_children)

        return node

    def expansion(self, node):
        """
        Expansion step of MCTS algorithm. Add one child for the first untried action.
        """
        actions = self.search_problem.get_available_actions(node.state)
        tried_actions = {child.action for child in node.children}

        for action in actions:
            if action not in tried_actions:
                new_state = self.search_problem.transition(node.state, action)
                new_node = MCTSNode(new_state, parent=node, action=action)
                node.children.append(new_node)

                if self.debug and node.parent is None:
                    print(f"[Expansion] Added root child action={action}")

                return new_node

        if self.debug:
            print("[Expansion] No untried actions left. Returning original node.")

        return node

    def simulation(self, state):
        """
        Simulation step of MCTS algorithm. Random rollout until terminal state.
        Returns +1 if black wins, -1 if white wins, 0 if draw.
        """
        current_state = state
        rollout_length = 0

        while not self.search_problem.is_terminal_state(current_state):
            actions = self.search_problem.get_available_actions(current_state)

            if not actions:
                if self.debug:
                    print("[Simulation] No available actions before terminal state.")
                break

            action = random.choice(actions)
            current_state = self.search_problem.transition(current_state, action)
            rollout_length += 1

        if self.search_problem.is_terminal_state(current_state):
            result = self.search_problem.get_result(current_state)
            if self.debug and random.random() < 0.01:
                print(f"[Simulation] Rollout ended. Length={rollout_length}, result={result}")
            return result

        if self.debug:
            print("[Simulation] Returning 0 because rollout did not reach terminal.")
        return 0

    def backpropagation(self, node, result):
        """
        Backpropagation step. Propagates result (always from black's perspective) up to root.
        """
        path_length = 0
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent
            path_length += 1

        if self.debug and random.random() < 0.01:
            print(f"[Backpropagation] Updated path length={path_length}, result={result}")

    def time_remaining(self, start_time, time_limit):
        return (time.time() - start_time) < time_limit

    def __str__(self):
        return "MCTS"

    def create_figure(self):
        start_state = GoState(5)
        mcts_agent = MCTSAgent()
        root_01 = mcts_agent.get_move(start_state, time_limit=0.1)
        root_1 = mcts_agent.get_move(start_state, time_limit=1)

        rates_01 = self.get_win_rate(root_01)
        rates_1 = self.get_win_rate(root_1)

        figure, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].bar(range(len(rates_01)), rates_01)
        ax[0].set_title('MCTS Win Rates with 0.1s Time Limit')
        ax[0].set_xlabel('Child Index')
        ax[0].set_ylabel('Win Rate')

        ax[1].bar(range(len(rates_1)), rates_1)
        ax[1].set_title('MCTS Win Rates with 1s Time Limit')
        ax[1].set_xlabel('Child Index')
        ax[1].set_ylabel('Win Rate')
        figure.show()

    def get_visit_counts(self, node):
        actions = [child.action for child in node.children]
        visits  = [child.visits for child in node.children]
        return actions, visits


###################################################
#
# Part 3: Final Agent
#
# Approach: MCTS enhanced with a learned value network.
# Instead of always doing random rollouts, we sometimes
# evaluate leaf nodes using a neural network trained on
# game outcome data.
#
###################################################

class ValueMCTSAgent(MCTSAgent):
    """
    MCTS agent that uses a trained value network to evaluate leaf nodes.

    Inherits selection, expansion, simulation, and backpropagation from MCTSAgent.
    Values are stored from black's perspective, matching the MCTSAgent logic.
    """

    def __init__(self, value_model, c=np.sqrt(2), debug=False):
        super().__init__(c, debug=debug)
        self.value_model = value_model
        self.value_model.eval()

    def __str__(self):
        return "ValueMCTS (MCTS + NN)"

    # ------------------------------------------------------------------
    # Core MCTS loop
    # ------------------------------------------------------------------

    def get_move(self, game_state, time_limit):
        """
        Run MCTS for the allowed time budget and return the best action.
        Every 5th iteration uses the value network instead of a random rollout.
        """
        root = MCTSNode(game_state)

        actions = self.search_problem.get_available_actions(game_state)
        if len(actions) == 1:
            return actions[0]

        start_time = time.time()
        per_move_limit = max(0.05, time_limit * 0.90 - 0.05)

        iteration = 0
        value_evals = 0
        rollout_evals = 0

        if self.debug:
            print("\n[ValueMCTS] ---------------- NEW MOVE ----------------")
            print(f"[ValueMCTS] Player to move: {game_state.player_to_move()}")
            print(f"[ValueMCTS] Legal actions: {len(actions)}")
            print(f"[ValueMCTS] Given time_limit: {time_limit:.3f}")
            print(f"[ValueMCTS] Using per_move_limit: {per_move_limit:.3f}")

        while self.time_remaining(start_time, per_move_limit):
            node = self.selection(root)

            if not self.search_problem.is_terminal_state(node.state):
                node = self.expansion(node)

            # Use the value network occasionally, but keep most evaluations as rollouts
            if iteration % 5 == 0:
                result = self.evaluate_state(node.state)
                value_evals += 1
            else:
                result = self.simulation(node.state)
                rollout_evals += 1

            self.backpropagation(node, result)
            iteration += 1

        if not root.children:
            return random.choice(actions)

        root_player = game_state.player_to_move()

        def final_score(child):
            avg_black_value = child.value / child.visits

            if root_player == 0:
                return avg_black_value
            else:
                return -avg_black_value

        best_child = max(root.children, key=final_score)

        if self.debug:
            elapsed = time.time() - start_time
            print(f"[ValueMCTS] Search finished after {elapsed:.3f}s")
            print(f"[ValueMCTS] Iterations: {iteration}")
            print(f"[ValueMCTS] Value evals: {value_evals}")
            print(f"[ValueMCTS] Rollout evals: {rollout_evals}")
            print(f"[ValueMCTS] Root visits: {root.visits}")
            print(f"[ValueMCTS] Root children: {len(root.children)}")

            print("[ValueMCTS] Root child stats:")
            for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
                avg_black_value = child.value / child.visits if child.visits > 0 else 0
                score_for_player = final_score(child)

                print(
                    f"  action={child.action}, "
                    f"visits={child.visits}, "
                    f"value={child.value:.2f}, "
                    f"avg_black_value={avg_black_value:.3f}, "
                    f"score_for_player={score_for_player:.3f}"
                )

            print(f"[ValueMCTS] Chosen action: {best_child.action}")
            print("[ValueMCTS] ------------------------------------------\n")

        return best_child.action

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_state(self, state):
        """
        Use the value network to estimate the outcome of a state.
        Return the value from black's perspective.
        """
        if self.search_problem.is_terminal_state(state):
            return self.search_problem.get_result(state)

        features = get_features(state)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            current_player_value = self.value_model(features_tensor).item()

        current_player_value = max(-1.0, min(1.0, current_player_value))

        # Convert to black's perspective
        if state.player_to_move() == 0:
            black_value = current_player_value
        else:
            black_value = -current_player_value

        if self.debug and random.random() < 0.01:
            print(
                f"[ValueNet] player={state.player_to_move()}, "
                f"current_player_value={current_player_value:.3f}, "
                f"black_value={black_value:.3f}"
            )

        return black_value

    # ------------------------------------------------------------------
    # Policy network helpers (disabled)
    # ------------------------------------------------------------------

    # def rank_actions_with_policy(self, state, legal_actions, top_k=3):
    #     """
    #     Rank legal actions by policy network probability.
    #     Only the top_k moves are placed first; the rest are shuffled.
    #
    #     Disabled: the policy network converged to near-uniform probabilities
    #     and did not learn a useful prior from the available features.
    #     """
    #     features = get_features(state)
    #     x = torch.tensor(features, dtype=torch.float32)
    #
    #     with torch.no_grad():
    #         logits = self.policy_model(x)
    #         probs = torch.softmax(logits, dim=-1)
    #
    #     ranked = sorted(
    #         legal_actions,
    #         key=lambda action: probs[action].item(),
    #         reverse=True
    #     )
    #
    #     top_moves = ranked[:top_k]
    #     rest = ranked[top_k:]
    #     random.shuffle(rest)
    #     return top_moves + rest


# ------------------------------------------------------------------
# Model and dataset utilities
# ------------------------------------------------------------------

def load_model(path: str, model):
    """Load saved weights into a model with matching architecture."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_model(path: str, model):
    """Save model weights."""
    torch.save(
        {"model_state_dict": model.state_dict()},
        path
    )


def load_dataset(path: str):
    """Load a pickled training dataset from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_player_to_move(state):
    """
    Helper for handling both GoState objects and dictionary states.
    """
    if isinstance(state, dict):
        return state["player_to_move"]

    return state.player_to_move()


def convert_result_to_current_player_value(state, result):
    """
    Convert final game result into the current player's perspective.

    result is from black's perspective:
        +1 = black wins
        -1 = white wins
    """
    player = get_player_to_move(state)

    if player == 0:
        return result
    else:
        return -result


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_value_network(dataset, num_epochs, learning_rate):
    """
    Train a value network to predict game outcomes from board features.
    Labels are from the current player's perspective.
    """
    import torch.optim as optim

    random.shuffle(dataset)

    input_size = len(get_features(dataset[0][0]))
    model = ValueNetwork(input_size)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 32

    for epoch in range(num_epochs):
        random.shuffle(dataset)
        total_loss = 0.0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            features = []
            labels = []

            for state, action, result in batch:
                features.append(get_features(state))

                # The model learns value from the current player's perspective
                label = convert_result_to_current_player_value(state, result)
                labels.append(label)

            features_tensor = torch.tensor(features, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

            predictions = model(features_tensor)
            loss = loss_function(predictions, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Value Network - Epoch {epoch + 1}/{num_epochs}, loss: {total_loss:.4f}")

    return model


# Disabled: the policy network did not learn a strong enough policy to improve win rate
# def train_policy_network(dataset, num_epochs, learning_rate):
#     """
#     Train a policy network to predict which move was played from a given state.
#     Trains only on moves made by the winning player.
#     """
#     import torch.optim as optim
#
#     filtered_dataset = []
#
#     for state, action, result in dataset:
#         try:
#             player = get_player_to_move(state)
#
#             if (result == 1 and player == 0) or (result == -1 and player == 1):
#                 filtered_dataset.append((state, action))
#         except Exception:
#             filtered_dataset.append((state, action))
#
#     print(f"Policy training: {len(filtered_dataset)} examples (from {len(dataset)} total)")
#
#     if len(filtered_dataset) == 0:
#         filtered_dataset = [(state, action) for state, action, result in dataset]
#
#     random.shuffle(filtered_dataset)
#
#     input_size = len(get_features(filtered_dataset[0][0]))
#     output_size = 26  # 5x5 board: 25 positions + 1 pass action
#     model = PolicyNetwork(input_size, output_size)
#
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     batch_size = 32
#
#     for epoch in range(num_epochs):
#         random.shuffle(filtered_dataset)
#         total_loss = 0.0
#
#         for i in range(0, len(filtered_dataset), batch_size):
#             batch = filtered_dataset[i:i + batch_size]
#
#             features = [get_features(state) for state, action in batch]
#             labels = [action for state, action in batch]
#
#             features_tensor = torch.tensor(features, dtype=torch.float32)
#             labels_tensor = torch.tensor(labels, dtype=torch.long)
#
#             predictions = model(features_tensor)
#             loss = loss_function(predictions, labels_tensor)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         print(f"Policy Network - Epoch {epoch + 1}/{num_epochs}, loss: {total_loss:.4f}")
#
#     return model


def collect_self_play_data(num_games=10, board_size=5, time_limit=1.0):
    """
    Generate training data by running self-play games.
    Each (state, action, result) tuple records a move and the final outcome.
    """
    dataset = []
    problem = GoProblem(size=board_size)
    agent = MCTSAgent()

    for game_num in range(num_games):
        state = problem.start_state
        trajectory = []

        while not problem.is_terminal_state(state):
            action = agent.get_move(state, time_limit=time_limit)
            trajectory.append((state, action))
            state = problem.transition(state, action)

        result = problem.get_result(state)

        for state, action in trajectory:
            dataset.append((state, action, result))

        print(f"Self-play game {game_num + 1}/{num_games} complete, result: {result}")

    return dataset


# ------------------------------------------------------------------
# Agent factory (called by game runner)
# ------------------------------------------------------------------

def get_final_agent_5x5():
    """Load trained model and return the final submission agent for 5x5."""
    feature_size = 21
    value_model = ValueNetwork(feature_size)
    value_model = load_model("value_net_5x5.pth", value_model)
    return ValueMCTSAgent(value_model)


def get_final_agent_9x9():
    """Load trained model and return the final submission agent for 9x9."""
    feature_size = 21
    value_model = ValueNetwork(feature_size)
    value_model = load_model("value_net_9x9.pth", value_model)
    return ValueMCTSAgent(value_model)


# ------------------------------------------------------------------
# Training script (run directly to retrain and save models)
# ------------------------------------------------------------------

if __name__ == "__main__":
    base_dataset = load_dataset("dataset_5x5_pygo.pkl")

    # 5x5 value network training
    value_model_5x5 = train_value_network(
        base_dataset,
        num_epochs=20,
        learning_rate=0.001,
    )

    save_model("value_net_5x5.pth", value_model_5x5)

    # 9x9 training
    dataset_9x9 = collect_self_play_data(
        num_games=50,
        board_size=9,
        time_limit=1.0,
    )

    value_model_9x9 = train_value_network(
        dataset_9x9,
        num_epochs=20,
        learning_rate=0.001,
    )

    save_model("value_net_9x9.pth", value_model_9x9)

    print("Models saved.")


# ------------------------------------------------------------------
# Policy network diagnostic — check if it learned a useful prior
# ------------------------------------------------------------------
# problem = GoProblem()
# state = problem.start_state
# legal = problem.get_available_actions(state)
# features = get_features(state)
# x = torch.tensor(features, dtype=torch.float32)
# with torch.no_grad():
#     probs = torch.softmax(policy_model(x), dim=-1)
# ranked = sorted(legal, key=lambda a: probs[a].item(), reverse=True)
# print("Policy ranking:", ranked)
# print("Probs:", [round(probs[a].item(), 3) for a in ranked])
# # A well-trained policy should have max prob > 0.15.
# # Near-uniform probabilities mean the network did not learn a useful prior.