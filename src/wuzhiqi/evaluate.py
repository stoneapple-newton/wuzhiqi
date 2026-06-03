"""Evaluation helpers for Gomoku agents."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class MatchResult:
    agent1_wins: int
    agent2_wins: int
    draws: int
    games: int
    agent1_first: dict[str, int]
    agent1_second: dict[str, int]
    game_lengths: list[int]
    runtime_seconds: float
    agent1_label: str
    agent2_label: str

    @property
    def score_rate(self) -> float:
        return (self.agent1_wins + 0.5 * self.draws) / self.games

    @property
    def average_game_length(self) -> float:
        return sum(self.game_lengths) / self.games if self.games else 0.0

    @property
    def confidence_interval_95(self) -> float:
        if self.games <= 0:
            return 0.0
        score = self.score_rate
        return 1.96 * ((score * (1 - score) / self.games) ** 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent1_label": self.agent1_label,
            "agent2_label": self.agent2_label,
            "agent1_wins": self.agent1_wins,
            "agent2_wins": self.agent2_wins,
            "draws": self.draws,
            "games": self.games,
            "score_rate": self.score_rate,
            "confidence_interval_95": self.confidence_interval_95,
            "average_game_length": self.average_game_length,
            "game_lengths": self.game_lengths,
            "agent1_first": self.agent1_first,
            "agent1_second": self.agent1_second,
            "runtime_seconds": self.runtime_seconds,
        }


def make_model_player(
    model_file: str | Path,
    width: int,
    height: int,
    c_puct: float,
    playouts: int,
) -> MCTSPlayer:
    policy_net = PolicyValueNet(width, height, model_file=model_file)
    return MCTSPlayer(policy_net.policy_value_fn, c_puct=c_puct, n_playout=playouts)


def evaluate_match(
    agent1,
    agent2,
    width: int,
    height: int,
    n_in_row: int,
    games: int,
    agent1_label: str = "agent1",
    agent2_label: str = "agent2",
) -> MatchResult:
    started = time.perf_counter()
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)
    agent1_wins = agent2_wins = draws = 0
    agent1_first = {"wins": 0, "losses": 0, "draws": 0}
    agent1_second = {"wins": 0, "losses": 0, "draws": 0}
    game_lengths: list[int] = []

    for index in range(games):
        start_player = index % 2
        winner = game.start_play(agent1, agent2, start_player=start_player, is_shown=0)
        length = len(board.states)
        game_lengths.append(length)
        agent1_player_id = board.players[0]
        bucket = agent1_first if start_player == 0 else agent1_second
        if winner == -1:
            draws += 1
            bucket["draws"] += 1
        elif winner == agent1_player_id:
            agent1_wins += 1
            bucket["wins"] += 1
        else:
            agent2_wins += 1
            bucket["losses"] += 1
        for agent in (agent1, agent2):
            reset = getattr(agent, "reset_player", None)
            if reset:
                reset()

    return MatchResult(
        agent1_wins=agent1_wins,
        agent2_wins=agent2_wins,
        draws=draws,
        games=games,
        agent1_first=agent1_first,
        agent1_second=agent1_second,
        game_lengths=game_lengths,
        runtime_seconds=time.perf_counter() - started,
        agent1_label=agent1_label,
        agent2_label=agent2_label,
    )


def evaluate_checkpoint_match(
    model_file: str | Path,
    opponent: str = "heuristic",
    opponent_model_file: str | Path | None = None,
    config_path: str | Path = "config/default.yaml",
    games: int | None = None,
    playouts: int | None = None,
) -> MatchResult:
    config = load_config(config_path)
    seed = config.get("evaluation", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))
    width = int(config.get("board_width", config.get("board_size", 6)))
    height = int(config.get("board_height", config.get("board_size", 6)))
    n_in_row = int(config.get("n_in_row", 4))
    mcts_cfg = config.get("mcts", {})
    eval_cfg = config.get("evaluation", {})
    c_puct = float(mcts_cfg.get("cpuct", 5.0))
    agent_playouts = int(playouts or mcts_cfg.get("simulations", 400))
    agent1 = make_model_player(model_file, width, height, c_puct, agent_playouts)
    agent1_label = Path(model_file).name
    if opponent == "heuristic":
        agent2 = HeuristicPlayer(search_radius=int(eval_cfg.get("heuristic_search_radius", 2)))
        agent2_label = f"heuristic_radius:{agent2.search_radius}"
    elif opponent == "pure_mcts":
        agent2 = PureMCTSPlayer(c_puct=5.0, n_playout=int(eval_cfg.get("pure_mcts_playouts", 1000)))
        agent2_label = f"pure_mcts_playouts:{eval_cfg.get('pure_mcts_playouts', 1000)}"
    elif opponent == "model":
        if opponent_model_file is None:
            raise ValueError("opponent_model_file is required for model opponent")
        agent2 = make_model_player(opponent_model_file, width, height, c_puct, agent_playouts)
        agent2_label = Path(opponent_model_file).name
    else:
        raise ValueError(f"unknown opponent: {opponent}")
    return evaluate_match(
        agent1,
        agent2,
        width=width,
        height=height,
        n_in_row=n_in_row,
        games=games or int(eval_cfg.get("games", 10)),
        agent1_label=agent1_label,
        agent2_label=agent2_label,
    )


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
