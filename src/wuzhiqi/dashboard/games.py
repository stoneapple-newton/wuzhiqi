"""Interactive game sessions used by the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wuzhiqi.game import Board
from wuzhiqi.heuristic import HeuristicPlayer
from wuzhiqi.mcts import MCTSPlayer
from wuzhiqi.network import PolicyValueNet
from wuzhiqi.utils import load_config


@dataclass
class HumanVsModelSession:
    config_path: Path
    model_path: Path | None = None
    playouts: int | None = None
    human_first: bool = True
    use_heuristic: bool = False
    board: Board = field(init=False)
    ai: object = field(init=False)
    human_player_id: int = field(init=False)
    ai_player_id: int = field(init=False)
    winner: int = -1
    ended: bool = False

    def __post_init__(self) -> None:
        config = load_config(self.config_path)
        width = int(config.get("board_width", config.get("board_size", 6)))
        height = int(config.get("board_height", config.get("board_size", 6)))
        n_in_row = int(config.get("n_in_row", 4))
        self.board = Board(width=width, height=height, n_in_row=n_in_row)
        start_player = 0 if self.human_first else 1
        self.board.init_board(start_player=start_player)
        self.human_player_id = self.board.players[0]
        self.ai_player_id = self.board.players[1]
        if not self.human_first:
            self.human_player_id, self.ai_player_id = self.ai_player_id, self.human_player_id
        if self.use_heuristic:
            self.ai = HeuristicPlayer()
        else:
            if self.model_path is None:
                raise ValueError("model_path is required")
            mcts_cfg = config.get("mcts", {})
            policy = PolicyValueNet(width, height, model_file=self.model_path)
            self.ai = MCTSPlayer(
                policy.policy_value_fn,
                c_puct=float(mcts_cfg.get("cpuct", 5.0)),
                n_playout=int(self.playouts or mcts_cfg.get("simulations", 400)),
            )
        self.ai.set_player_ind(self.ai_player_id)
        if not self.human_first:
            self._ai_move()

    def human_move(self, row: int, col: int) -> dict[str, object]:
        if self.ended:
            return self.snapshot()
        if self.board.get_current_player() != self.human_player_id:
            raise ValueError("it is not the human player's turn")
        move = self.board.location_to_move([row, col])
        if move not in self.board.availables:
            raise ValueError("illegal move")
        self.board.do_move(move)
        self._refresh_end()
        if not self.ended:
            self._ai_move()
        return self.snapshot()

    def _ai_move(self) -> None:
        move = self.ai.get_action(self.board)
        self.board.do_move(move)
        self._refresh_end()

    def _refresh_end(self) -> None:
        self.ended, self.winner = self.board.game_end()

    def snapshot(self) -> dict[str, object]:
        cells = [[0 for _ in range(self.board.width)] for _ in range(self.board.height)]
        for move, player in self.board.states.items():
            row, col = self.board.move_to_location(move)
            cells[row][col] = player
        return {
            "width": self.board.width,
            "height": self.board.height,
            "n_in_row": self.board.n_in_row,
            "cells": cells,
            "current_player": self.board.current_player,
            "human_player": self.human_player_id,
            "ai_player": self.ai_player_id,
            "ended": self.ended,
            "winner": self.winner,
            "last_move": self.board.last_move,
        }
