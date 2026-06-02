"""Monte Carlo Tree Search implementations for Gomoku."""

from __future__ import annotations

import copy
from operator import itemgetter
from typing import Callable, Iterable

import numpy as np

from wuzhiqi.game import Board, Move


PolicyValueFn = Callable[[Board], tuple[Iterable[tuple[Move, float]], float]]


def softmax(x: np.ndarray) -> np.ndarray:
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """A node in the PUCT search tree."""

    def __init__(self, parent: "TreeNode | None", prior_p: float) -> None:
        self._parent = parent
        self._children: dict[Move, TreeNode] = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = float(prior_p)

    def expand(self, action_priors: Iterable[tuple[Move, float]]) -> None:
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, float(prob))

    def select(self, c_puct: float) -> tuple[Move, "TreeNode"]:
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value: float) -> None:
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value: float) -> None:
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct: float) -> float:
        if self._parent is None:
            return self._Q
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self) -> bool:
        return self._children == {}

    def is_root(self) -> bool:
        return self._parent is None


class MCTS:
    """PUCT MCTS guided by a policy-value function."""

    def __init__(self, policy_value_fn: PolicyValueFn, c_puct: float = 5.0, n_playout: int = 400):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state: Board) -> None:
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        elif winner == -1:
            leaf_value = 0.0
        else:
            leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state: Board, temp: float = 1e-3) -> tuple[tuple[Move, ...], np.ndarray]:
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        if not act_visits:
            return tuple(), np.array([], dtype=np.float32)
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits, dtype=np.float64) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move: Move) -> None:
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self) -> str:
        return "MCTS"


class MCTSPlayer:
    """AlphaZero-style MCTS player."""

    def __init__(
        self,
        policy_value_function: PolicyValueFn,
        c_puct: float = 5.0,
        n_playout: int = 400,
        is_selfplay: int = 0,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
    ) -> None:
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self._dirichlet_alpha = dirichlet_alpha
        self._exploration_fraction = exploration_fraction
        self.player = 0

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, board: Board, temp: float = 1e-3, return_prob: int = 0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height, dtype=np.float32)
        if not sensible_moves:
            raise ValueError("the board is full")

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            noise = np.random.dirichlet(self._dirichlet_alpha * np.ones(len(probs)))
            mixed_probs = (1 - self._exploration_fraction) * probs + self._exploration_fraction * noise
            move = int(np.random.choice(acts, p=mixed_probs))
            self.mcts.update_with_move(move)
        else:
            move = int(np.random.choice(acts, p=probs))
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        return move

    def __str__(self) -> str:
        return f"MCTS {self.player}"


def rollout_policy_fn(board: Board):
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def uniform_policy_value_fn(board: Board):
    action_probs = np.ones(len(board.availables), dtype=np.float32) / len(board.availables)
    return zip(board.availables, action_probs), 0.0


class PureMCTS(MCTS):
    """MCTS with random rollouts instead of a neural value."""

    def _playout(self, state: Board) -> None:
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)
        action_probs, _ = self._policy(state)
        end, _ = state.game_end()
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state: Board, limit: int = 1000) -> float:
        player = state.get_current_player()
        winner = -1
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        if winner == -1:
            return 0.0
        return 1.0 if winner == player else -1.0

    def get_move(self, state: Board) -> Move:
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]


class PureMCTSPlayer:
    """Baseline MCTS player with random rollouts."""

    def __init__(self, c_puct: float = 5.0, n_playout: int = 1000) -> None:
        self.mcts = PureMCTS(uniform_policy_value_fn, c_puct, n_playout)
        self.player = 0

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, board: Board) -> Move:
        if not board.availables:
            raise ValueError("the board is full")
        move = self.mcts.get_move(board)
        self.mcts.update_with_move(-1)
        return move

    def __str__(self) -> str:
        return f"MCTS {self.player}"
