"""Evaluation helpers for Gomoku agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wuzhiqi.game import Board, Game
from wuzhiqi.heuristic import HeuristicPlayer
from wuzhiqi.mcts import MCTSPlayer, PureMCTSPlayer
from wuzhiqi.network import PolicyValueNet
from wuzhiqi.utils import load_config, set_seed


@dataclass
class EvaluationResult:
    wins: int
    losses: int
    draws: int
    games: int

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games


def evaluate_checkpoint(
    model_file: str | Path,
    config_path: str | Path = "config/default.yaml",
    games: int | None = None,
) -> EvaluationResult:
    config = load_config(config_path)
    seed = config.get("evaluation", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))

    width = int(config.get("board_width", config.get("board_size", 6)))
    height = int(config.get("board_height", config.get("board_size", 6)))
    n_in_row = int(config.get("n_in_row", 4))
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)
    policy_net = PolicyValueNet(width, height, model_file=model_file)
    mcts_cfg = config.get("mcts", {})
    eval_cfg = config.get("evaluation", {})
    current_player = MCTSPlayer(
        policy_net.policy_value_fn,
        c_puct=float(mcts_cfg.get("cpuct", 5.0)),
        n_playout=int(mcts_cfg.get("simulations", 400)),
    )
    opponent = str(eval_cfg.get("opponent", "pure_mcts"))
    if opponent == "heuristic":
        opponent_player = HeuristicPlayer(search_radius=int(eval_cfg.get("heuristic_search_radius", 2)))
    elif opponent == "pure_mcts":
        opponent_player = PureMCTSPlayer(
            c_puct=5.0,
            n_playout=int(eval_cfg.get("pure_mcts_playouts", 1000)),
        )
    else:
        raise ValueError(f"unknown evaluation opponent: {opponent}")
    num_games = games or int(eval_cfg.get("games", 10))
    wins = losses = draws = 0
    for i in range(num_games):
        winner = game.start_play(current_player, opponent_player, start_player=i % 2, is_shown=0)
        if winner == -1:
            draws += 1
        elif winner == 1:
            wins += 1
        else:
            losses += 1
    return EvaluationResult(wins=wins, losses=losses, draws=draws, games=num_games)


def main() -> None:
    config = load_config()
    model_file = config.get("evaluation", {}).get("model_file", "checkpoints/current_policy.pt")
    result = evaluate_checkpoint(model_file)
    print(result)


if __name__ == "__main__":
    main()
