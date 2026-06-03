import json

from wuzhiqi.dashboard.games import HumanVsModelSession
from wuzhiqi.dashboard.logs import parse_log, summarize_records
from wuzhiqi.evaluate import evaluate_match
from wuzhiqi.heuristic import HeuristicPlayer


def test_dashboard_parses_jsonl_training_log(tmp_path) -> None:
    log_path = tmp_path / "training.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "init", "batch": 0}),
                json.dumps(
                    {
                        "event": "batch",
                        "batch": 1,
                        "episode_len": 7,
                        "kl": 0.01,
                        "lr_multiplier": 1.0,
                        "effective_lr": 0.001,
                        "loss": 2.0,
                        "policy_loss": 1.4,
                        "value_loss": 0.6,
                        "entropy": 3.0,
                        "replay_buffer": 56,
                        "explained_var_old": 0.0,
                        "explained_var_new": 0.1,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = parse_log(log_path)
    summary = summarize_records(records)

    assert len(records) == 1
    assert summary["summary"]["latest_batch"] == 1
    assert summary["summary"]["policy_loss"]["latest"] == 1.4


def test_dashboard_parses_text_training_log(tmp_path) -> None:
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "batch i:2, episode_len:9,start_player:1,winner:2\n"
        "kl:0.02000,lr_multiplier:0.667,effective_lr:0.0003333,loss:3.0,"
        "policy_loss:2.5,value_loss:0.5,entropy:2.7,replay_buffer:72,"
        "explained_var_old:0.000,explained_var_new:0.200\n",
        encoding="utf-8",
    )

    records = parse_log(log_path)

    assert records[0]["batch"] == 2
    assert records[0]["episode_len"] == 9
    assert records[0]["value_loss"] == 0.5


def test_evaluate_match_reports_color_split_and_lengths() -> None:
    result = evaluate_match(
        HeuristicPlayer(),
        HeuristicPlayer(),
        width=6,
        height=6,
        n_in_row=4,
        games=2,
        agent1_label="h1",
        agent2_label="h2",
    )

    assert result.games == 2
    assert len(result.game_lengths) == 2
    assert result.agent1_first["wins"] + result.agent1_first["losses"] + result.agent1_first["draws"] == 1
    assert result.agent1_second["wins"] + result.agent1_second["losses"] + result.agent1_second["draws"] == 1
    assert 0.0 <= result.score_rate <= 1.0


def test_human_vs_model_session_accepts_human_move_with_heuristic(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
board_width: 6
board_height: 6
n_in_row: 4
mcts:
  simulations: 1
  cpuct: 5.0
evaluation:
  seed: 123
""",
        encoding="utf-8",
    )
    session = HumanVsModelSession(config_path=config_path, use_heuristic=True)

    state = session.human_move(2, 2)

    assert state["cells"][2][2] == state["human_player"]
    assert len(session.board.states) == 2
