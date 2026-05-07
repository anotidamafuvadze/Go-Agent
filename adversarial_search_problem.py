from abc import ABC, abstractmethod
from typing import Generic, Set, Tuple, TypeVar

###############################################################################
# This file is taken from HW 3, Adversarial Search.
# 
# An AdversarialSearchProblem is a representation of a game that is convenient
# for running adversarial search algorithms.
#
# A game can be put into this form by extending the AdversarialSearchProblem
# class. See tttproblem.py for an example of this.
#
# Every subclass of AdversarialSearchProblem has its game states represented
# as instances of a subclass of GameState. The only requirement that of a
# subclass of GameState is that it must implement that player_to_move(.) method,
# which returns the index (0-indexed) of the next player to move.
###############################################################################


class GameState(ABC):
    @abstractmethod
    def player_to_move(self) -> int:
        """
        Output- Returns the index of the player who will move next.
        """
        pass


State = TypeVar("State", bound=GameState)

# Action represents the type of actions that an instance of AdversarialSearchProblem uses to
# cause a transition. It's generic because different games have different actions: TTT requires
# placing a piece on a *2D* grid, while Connect 4 just involves selecting a column.
Action = TypeVar("Action")


class AdversarialSearchProblem(ABC, Generic[State, Action]):
    def get_start_state(self):
        """
        Output- Returns the state from which to start.
        """
        return self._start_state

    def set_start_state(self, state: State):
        """
        Changes the start state to the given state.
        Note to student: You should not need to use this.
        This is only for running games.

        Input:
                state- a GameState
        """
        self._start_state = state

    @abstractmethod
    def get_available_actions(self, state: State) -> Set[Action]:
        """
        Input:
                state- a GameState
        Output:
                Returns the set of actions available to the player-to-move
                from the given state
        """
        pass

    @abstractmethod
    def transition(self, state: State, action: Action) -> State:
        """
        Input:
                state- a Gamestate
                action- the action to take
        Ouput:
                Returns the state that results from taking the given action
                from the given state. (Assume deterministic transitions.)
        """
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)
        pass

    @abstractmethod
    def is_terminal_state(self, state: State) -> bool:
        """
        Input:
                state: a GameState
        Output:
                Returns a boolean indicating whether or not the given
                state is terminal.
        """
        pass

    # Used to be called evaluate_state
    @abstractmethod
    def get_result(self, state: State) -> float:
        """
        Called to determine the value of a *terminal* state.
        The range of values returned here should be synchronized with heuristic.

        Input:
                state: a TERMINAL GameState
        Output:
                Returns the value of a terminal state.
        """
        assert self.is_terminal_state(state)
        pass


###############################################################################
# GameUI is an abstraction that allows you to interact directly with
# an AdversarialSearchProblem (through gamerunner.py). See tttproblem or
# connect4problem for examples.
#
# Utilizing GameUI is NOT necessary for this assignment, although you can use
# it with any ASPs you may decide to create.
###############################################################################


class GameUI(ABC):
    def update_state(self, state: GameState):
        """
        Updates the state currently being rendered.
        """
        self._state = state

    @abstractmethod
    def render(self):
        """
        Renders the GameUI instance's render (presumably this will be called continuously).
        """
        pass

    @abstractmethod
    def get_user_input_action(self):
        """
        Output- Returns an action obtained through the GameUI input itself.
        (It is expected that GameUI validates that the action is valid).
        """
        pass
