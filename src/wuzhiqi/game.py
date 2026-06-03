"""Gomoku board rules and game drivers.

This module is adapted from the AlphaZero_Gomoku template and kept
dependency-light apart from NumPy state encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


PlayerId = int
Move = int


class Player(Protocol):
    player: PlayerId

    def set_player_ind(self, player: PlayerId) -> None: ...

    def get_action(self, board: "Board") -> Move: ...


@dataclass
class Board:
    """Board state for free-style Gomoku.

    Moves use row-major indexing: ``move = row * width + col``. Players are
    encoded as ``1`` and ``2`` to match the original template.
    """

    width: int = 6
    height: int = 6
    n_in_row: int = 4

    def __post_init__(self) -> None:
        self.players = [1, 2]
        self.states: dict[Move, PlayerId] = {}
        self.availables: list[Move] = []
        self.current_player: PlayerId = self.players[0]
        self.last_move: Move = -1
        self.init_board()

    def init_board(self, start_player: int = 0) -> None:
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise ValueError(
                f"board width and height cannot be less than {self.n_in_row}"
            )
        if start_player not in (0, 1):
            raise ValueError("start_player must be 0 or 1")
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move: Move) -> list[int]:
        return [move // self.width, move % self.width]

    def location_to_move(self, location: Iterable[int]) -> Move:
        loc = list(location)
        if len(loc) != 2:
            return -1
        row, col = loc
        move = row * self.width + col
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self) -> np.ndarray:
        """Return a 4-plane tensor from the current player's perspective."""

        square_state = np.zeros((4, self.width, self.height), dtype=np.float32)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            current_moves = moves[players == self.current_player]
            opponent_moves = moves[players != self.current_player]
            square_state[0][current_moves // self.width, current_moves % self.height] = 1.0
            square_state[1][opponent_moves // self.width, opponent_moves % self.height] = 1.0
            if self.last_move >= 0:
                square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        return square_state[:, ::-1, :]

    def do_move(self, move: Move) -> None:
        if move not in self.availables:
            raise ValueError(f"illegal move: {move}")
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self) -> tuple[bool, PlayerId]:
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = set(range(width * height)) - set(self.availables)
        if len(moved) < n * 2 - 1:
            return False, -1

        for move in moved:
            row = move // width
            col = move % width
            player = states[move]
            if col in range(width - n + 1) and all(
                states.get(i, -1) == player for i in range(move, move + n)
            ):
                return True, player
            if row in range(height - n + 1) and all(
                states.get(i, -1) == player for i in range(move, move + n * width, width)
            ):
                return True, player
            if col in range(width - n + 1) and row in range(height - n + 1) and all(
                states.get(i, -1) == player
                for i in range(move, move + n * (width + 1), width + 1)
            ):
                return True, player
            if col in range(n - 1, width) and row in range(height - n + 1) and all(
                states.get(i, -1) == player
                for i in range(move, move + n * (width - 1), width - 1)
            ):
                return True, player
        return False, -1

    def game_end(self) -> tuple[bool, PlayerId]:
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        if not self.availables:
            return True, -1
        return False, -1

    def get_current_player(self) -> PlayerId:
        return self.current_player


class Game:
    """Game runner for human/AI matches and self-play collection."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self.last_self_play_metadata: dict[str, object] = {}

    def graphic(self, board: Board, player1: PlayerId, player2: PlayerId) -> None:
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for col in range(board.width):
            print(f"{col:8}", end="")
        print("\r\n")
        for row in range(board.height - 1, -1, -1):
            print(f"{row:4d}", end="")
            for col in range(board.width):
                loc = row * board.width + col
                player = board.states.get(loc, -1)
                if player == player1:
                    print("X".center(8), end="")
                elif player == player2:
                    print("O".center(8), end="")
                else:
                    print("_".center(8), end="")
            print("\r\n\r\n")

    def start_play(
        self,
        player1: Player,
        player2: Player,
        start_player: int = 0,
        is_shown: int = 1,
    ) -> PlayerId:
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    print(f"Game end. Winner is {players[winner]}" if winner != -1 else "Game end. Tie")
                return winner

    def start_self_play(self, player: object, is_shown: int = 0, temp: float = 1e-3, start_player: int = 0):
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        first_mover = self.board.players[start_player]
        second_mover = self.board.players[1 - start_player]
        states: list[np.ndarray] = []
        mcts_probs: list[np.ndarray] = []
        current_players: list[PlayerId] = []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players), dtype=np.float32)
                if winner != -1:
                    players_arr = np.array(current_players)
                    winners_z[players_arr == winner] = 1.0
                    winners_z[players_arr != winner] = -1.0
                player.reset_player()
                self.last_self_play_metadata = {
                    "start_player": start_player,
                    "first_mover": first_mover,
                    "second_mover": second_mover,
                    "winner": winner,
                    "first_mover_won": winner == first_mover if winner != -1 else None,
                    "second_mover_won": winner == second_mover if winner != -1 else None,
                    "episode_len": len(current_players),
                }
                if is_shown:
                    print(f"Game end. Winner is player: {winner}" if winner != -1 else "Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)


class Gomoku:
    """Small compatibility wrapper used by the original scaffold tests."""

    def __init__(self, board_size: int = 6, n_in_row: int = 4) -> None:
        self.board_size = board_size
        self.n_in_row = n_in_row
        self._board = Board(width=board_size, height=board_size, n_in_row=n_in_row)
        self.reset()

    def reset(self) -> list[list[int]]:
        self._board.init_board()
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        return self.board

    def legal_moves(self) -> list[int]:
        return list(self._board.availables)

    def step(self, action: Move) -> tuple[list[list[int]], int, bool, dict[str, object]]:
        player = self._board.current_player
        self._board.do_move(action)
        row, col = divmod(action, self.board_size)
        self.board[row][col] = player
        end, winner = self._board.game_end()
        self.current_player = self._board.current_player
        reward = 1 if winner == player else 0
        return self.board, reward, end, {"winner": winner}
