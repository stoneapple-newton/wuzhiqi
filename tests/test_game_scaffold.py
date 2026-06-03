import numpy as np
import pytest

from wuzhiqi.game import Board, Gomoku
from wuzhiqi.heuristic import HeuristicPlayer
from wuzhiqi.human_play import HumanPlayer
from wuzhiqi.mcts import MCTSPlayer, uniform_policy_value_fn
from wuzhiqi.network import PolicyValueNet
from wuzhiqi.train import TrainPipeline


def test_reset_starts_empty_with_black_to_move() -> None:
    game = Gomoku(board_size=15, n_in_row=5)

    assert game.current_player == 1
    assert len(game.board) == 15
    assert all(cell == 0 for row in game.board for cell in row)


def test_legal_moves_are_all_points_on_empty_board() -> None:
    game = Gomoku(board_size=15, n_in_row=5)

    assert game.legal_moves() == list(range(225))


def test_board_default_is_template_smoke_setup() -> None:
    board = Board()

    assert board.width == 6
    assert board.height == 6
    assert board.n_in_row == 4
    assert board.availables == list(range(36))


@pytest.mark.parametrize(
    "moves",
    [
        [0, 6, 1, 7, 2, 8, 3],
        [0, 1, 6, 2, 12, 3, 18],
        [0, 1, 7, 2, 14, 3, 21],
        [3, 0, 8, 1, 13, 2, 18],
    ],
)
def test_board_detects_four_in_a_row(moves: list[int]) -> None:
    board = Board(width=6, height=6, n_in_row=4)

    for move in moves:
        board.do_move(move)

    assert board.game_end() == (True, 1)


def test_board_rejects_occupied_move() -> None:
    board = Board()
    board.do_move(0)

    with pytest.raises(ValueError):
        board.do_move(0)


def test_current_state_shape_and_planes() -> None:
    board = Board()
    board.do_move(0)

    state = board.current_state()

    assert state.shape == (4, 6, 6)
    assert np.sum(state[1]) == 1.0
    assert np.sum(state[2]) == 1.0


def test_mcts_returns_full_board_policy_with_legal_mass() -> None:
    board = Board()
    player = MCTSPlayer(uniform_policy_value_fn, n_playout=2)

    move, probs = player.get_action(board, return_prob=1)

    assert move in range(36)
    assert probs.shape == (36,)
    assert np.isclose(np.sum(probs), 1.0)


def test_policy_value_net_forward_and_train_step() -> None:
    board = Board()
    net = PolicyValueNet(6, 6)
    state_batch = [board.current_state()]
    mcts_probs = np.ones((1, 36), dtype=np.float32) / 36

    action_probs, values = net.policy_value(state_batch)
    loss, policy_loss, value_loss, entropy = net.train_step(state_batch, mcts_probs, [0.0], lr=0.001)

    assert action_probs.shape == (1, 36)
    assert values.shape == (1, 1)
    assert loss > 0
    assert policy_loss > 0
    assert value_loss >= 0
    assert entropy > 0


def test_training_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = str(tmp_path).replace("\\", "/")
    checkpoint_file = str(checkpoint_path).replace("\\", "/")
    config_path.write_text(
        f"""
board_width: 6
board_height: 6
n_in_row: 4
checkpoint_dir: {checkpoint_dir}
mcts:
  simulations: 1
  cpuct: 5.0
training:
  batch_size: 2
  learning_rate: 0.002
  weight_decay: 0.0001
  replay_buffer_size: 10
  temperature: 1.0
  epochs: 1
  kl_target: 0.02
  check_freq: 1
  game_batch_num: 2
  resume: false
  checkpoint_path: {checkpoint_file}
self_play:
  games_per_iteration: 1
evaluation:
  games: 1
  seed: 123
  pure_mcts_playouts: 1
""",
        encoding="utf-8",
    )
    pipeline = TrainPipeline(config_path)
    board = Board()
    replay_item = (board.current_state(), np.ones(36, dtype=np.float32) / 36, 1.0)
    pipeline.data_buffer.append(replay_item)
    pipeline.current_batch = 7
    pipeline.lr_multiplier = 0.5
    pipeline.best_win_ratio = 0.6
    pipeline.pure_mcts_playout_num = 2000

    pipeline.save_checkpoint()

    config_path.write_text(config_path.read_text(encoding="utf-8").replace("resume: false", "resume: true"))
    resumed = TrainPipeline(config_path)

    assert resumed.current_batch == 7
    assert resumed.lr_multiplier == 0.5
    assert resumed.best_win_ratio == 0.6
    assert resumed.pure_mcts_playout_num == 2000
    assert len(resumed.data_buffer) == 1


def test_init_model_without_resume_resets_training_state(tmp_path) -> None:
    model_path = tmp_path / "policy.pt"
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = str(tmp_path).replace("\\", "/")
    checkpoint_file = str(checkpoint_path).replace("\\", "/")
    model_file = str(model_path).replace("\\", "/")

    seed_config = tmp_path / "seed_config.yaml"
    seed_config.write_text(
        f"""
board_width: 6
board_height: 6
n_in_row: 4
checkpoint_dir: {checkpoint_dir}
mcts:
  simulations: 1
  cpuct: 5.0
training:
  batch_size: 2
  learning_rate: 0.002
  weight_decay: 0.0001
  replay_buffer_size: 10
  temperature: 1.0
  epochs: 1
  kl_target: 0.02
  check_freq: 1
  game_batch_num: 2
  resume: false
  checkpoint_path: {checkpoint_file}
self_play:
  games_per_iteration: 1
evaluation:
  games: 1
  seed: 123
  pure_mcts_playouts: 1
""",
        encoding="utf-8",
    )
    seeded = TrainPipeline(seed_config)
    seeded.current_batch = 7
    seeded.lr_multiplier = 0.5
    seeded.data_buffer.append((Board().current_state(), np.ones(36, dtype=np.float32) / 36, 1.0))
    seeded.policy_value_net.save_model(model_path)
    seeded.save_checkpoint()

    config_path.write_text(
        f"""
board_width: 6
board_height: 6
n_in_row: 4
checkpoint_dir: {checkpoint_dir}
init_model: {model_file}
mcts:
  simulations: 1
  cpuct: 5.0
training:
  batch_size: 2
  learning_rate: 0.002
  weight_decay: 0.0001
  replay_buffer_size: 10
  temperature: 1.0
  epochs: 1
  kl_target: 0.02
  check_freq: 1
  game_batch_num: 2
  resume: false
  checkpoint_path: {checkpoint_file}
  current_model_path: {model_file}
  best_model_path: {model_file}
  best_checkpoint_path: {checkpoint_file}
self_play:
  games_per_iteration: 1
evaluation:
  games: 1
  seed: 123
  pure_mcts_playouts: 1
""",
        encoding="utf-8",
    )
    reset = TrainPipeline(config_path)

    assert reset.current_batch == 0
    assert reset.lr_multiplier == 1.0
    assert len(reset.data_buffer) == 0
    assert reset.checkpoint_path == checkpoint_path
    assert reset.current_model_path == model_path
    assert reset.best_model_path == model_path
    assert reset.best_checkpoint_path == checkpoint_path


def test_human_player_parses_valid_move(monkeypatch) -> None:
    board = Board()
    human = HumanPlayer()
    monkeypatch.setattr("builtins.input", lambda _: "2,3")

    assert human.get_action(board) == board.location_to_move([2, 3])


def test_heuristic_player_takes_immediate_win() -> None:
    board = Board(width=6, height=6, n_in_row=4)
    for move in [0, 6, 1, 7, 2, 8]:
        board.do_move(move)
    player = HeuristicPlayer()

    assert player.get_action(board) == 3


def test_heuristic_player_blocks_immediate_loss() -> None:
    board = Board(width=6, height=6, n_in_row=4)
    for move in [6, 0, 7, 1, 12, 2]:
        board.do_move(move)
    player = HeuristicPlayer()

    assert player.get_action(board) == 3
