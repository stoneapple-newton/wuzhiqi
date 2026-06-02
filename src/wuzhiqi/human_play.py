"""Command-line human vs trained policy-value model."""

from __future__ import annotations

import argparse
from pathlib import Path

from wuzhiqi.game import Board, Game
from wuzhiqi.mcts import MCTSPlayer
from wuzhiqi.network import PolicyValueNet
from wuzhiqi.utils import load_config


class HumanPlayer:
    def __init__(self) -> None:
        self.player = 0

    def set_player_ind(self, player: int) -> None:
        self.player = player

    def get_action(self, board: Board) -> int:
        while True:
            raw_move = input("Your move as row,col: ").strip()
            try:
                row, col = (int(part.strip()) for part in raw_move.split(",", maxsplit=1))
            except ValueError:
                print("Invalid input. Use row,col, for example: 2,3")
                continue

            move = board.location_to_move([row, col])
            if move in board.availables:
                return move
            print("Illegal move. Choose an empty point inside the board.")

    def __str__(self) -> str:
        return f"Human {self.player}"


def default_model_path() -> Path:
    best_policy = Path("checkpoints/best_policy.pt")
    if best_policy.exists():
        return best_policy
    return Path("checkpoints/current_policy.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Gomoku against a trained Wuzhiqi model.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--model", default=str(default_model_path()), help="Path to a .pt policy checkpoint.")
    parser.add_argument("--playouts", type=int, default=None, help="MCTS simulations per AI move.")
    parser.add_argument(
        "--human-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the human moves first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")

    width = int(config.get("board_width", config.get("board_size", 6)))
    height = int(config.get("board_height", config.get("board_size", 6)))
    n_in_row = int(config.get("n_in_row", 4))
    mcts_cfg = config.get("mcts", {})
    n_playout = args.playouts or int(mcts_cfg.get("simulations", 400))

    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)
    policy_net = PolicyValueNet(width, height, model_file=model_path)
    ai_player = MCTSPlayer(
        policy_net.policy_value_fn,
        c_puct=float(mcts_cfg.get("cpuct", 5.0)),
        n_playout=n_playout,
    )
    human = HumanPlayer()

    print(f"Loaded model: {model_path}")
    print(f"Board: {height}x{width}, target: {n_in_row} in a row, AI playouts: {n_playout}")
    print("Rows and columns are zero-based. Enter moves like: 2,3")
    start_player = 0 if args.human_first else 1
    game.start_play(human, ai_player, start_player=start_player, is_shown=1)


if __name__ == "__main__":
    main()
