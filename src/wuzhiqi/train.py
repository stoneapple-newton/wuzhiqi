"""Training pipeline adapted from the AlphaZero_Gomoku template."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from wuzhiqi.game import Board, Game
from wuzhiqi.heuristic import HeuristicPlayer
from wuzhiqi.mcts import MCTSPlayer, PureMCTSPlayer
from wuzhiqi.network import PolicyValueNet
from wuzhiqi.utils import augment_play_data, load_config, set_seed


class TrainPipeline:
    def __init__(self, config_path: str | Path = "config/default.yaml", init_model: str | None = None):
        self.config = load_config(config_path)
        seed = self.config.get("evaluation", {}).get("seed")
        if seed is not None:
            set_seed(int(seed))

        self.board_width = int(self.config.get("board_width", self.config.get("board_size", 6)))
        self.board_height = int(self.config.get("board_height", self.config.get("board_size", 6)))
        self.n_in_row = int(self.config.get("n_in_row", 4))
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        mcts_cfg = self.config.get("mcts", {})
        training_cfg = self.config.get("training", {})
        self.learn_rate = float(training_cfg.get("learning_rate", 2e-3))
        self.lr_multiplier = 1.0
        self.temp = float(training_cfg.get("temperature", 1.0))
        self.n_playout = int(mcts_cfg.get("simulations", 400))
        self.c_puct = float(mcts_cfg.get("cpuct", 5.0))
        self.buffer_size = int(training_cfg.get("replay_buffer_size", 10000))
        self.batch_size = int(training_cfg.get("batch_size", 512))
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = int(self.config.get("self_play", {}).get("games_per_iteration", 1))
        self.epochs = int(training_cfg.get("epochs", 5))
        self.kl_targ = float(training_cfg.get("kl_target", 0.02))
        self.check_freq = int(training_cfg.get("check_freq", 50))
        self.game_batch_num = int(training_cfg.get("game_batch_num", 1500))
        self.current_batch = 0
        self.best_win_ratio = 0.0
        evaluation_cfg = self.config.get("evaluation", {})
        self.eval_opponent = str(evaluation_cfg.get("opponent", "pure_mcts"))
        self.pure_mcts_playout_num = int(evaluation_cfg.get("pure_mcts_playouts", 1000))
        self.heuristic_search_radius = int(evaluation_cfg.get("heuristic_search_radius", 2))
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_path = Path(training_cfg.get("checkpoint_path", self.checkpoint_dir / "training_checkpoint.pt"))
        self.current_model_path = Path(training_cfg.get("current_model_path", self.checkpoint_dir / "current_policy.pt"))
        self.best_model_path = Path(training_cfg.get("best_model_path", self.checkpoint_dir / "best_policy.pt"))
        self.best_checkpoint_path = Path(
            training_cfg.get("best_checkpoint_path", self.checkpoint_dir / "best_training_checkpoint.pt")
        )
        self.resume = bool(training_cfg.get("resume", False))

        model_file = init_model or self.config.get("init_model")
        self.policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            model_file=model_file,
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
            dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
            exploration_fraction=float(mcts_cfg.get("exploration_fraction", 0.25)),
        )
        self.episode_len = 0
        if self.resume:
            self.load_checkpoint(self.checkpoint_path)

    def get_equi_data(self, play_data):
        return augment_play_data(play_data, self.board_width, self.board_height)

    def checkpoint_state(self) -> dict:
        state = {
            "board_width": self.board_width,
            "board_height": self.board_height,
            "n_in_row": self.n_in_row,
            "current_batch": self.current_batch,
            "episode_len": self.episode_len,
            "best_win_ratio": self.best_win_ratio,
            "pure_mcts_playout_num": self.pure_mcts_playout_num,
            "lr_multiplier": self.lr_multiplier,
            "data_buffer": list(self.data_buffer),
            "model_state_dict": self.policy_value_net.get_policy_param(),
            "optimizer_state_dict": self.policy_value_net.optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def save_checkpoint(self, checkpoint_path: str | Path | None = None) -> None:
        path = Path(checkpoint_path or self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)
        print(f"saved training checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"resume checkpoint not found: {path}; starting from configured model")
            return

        checkpoint = torch.load(path, map_location=self.policy_value_net.device, weights_only=False)
        expected_shape = (
            checkpoint.get("board_width"),
            checkpoint.get("board_height"),
            checkpoint.get("n_in_row"),
        )
        actual_shape = (self.board_width, self.board_height, self.n_in_row)
        if expected_shape != actual_shape:
            raise ValueError(f"Checkpoint board config {expected_shape} does not match current config {actual_shape}")

        self.policy_value_net.load_policy_param(checkpoint["model_state_dict"])
        self.policy_value_net.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_batch = int(checkpoint.get("current_batch", 0))
        self.episode_len = int(checkpoint.get("episode_len", 0))
        self.best_win_ratio = float(checkpoint.get("best_win_ratio", 0.0))
        self.pure_mcts_playout_num = int(checkpoint.get("pure_mcts_playout_num", self.pure_mcts_playout_num))
        self.lr_multiplier = float(checkpoint.get("lr_multiplier", 1.0))
        self.data_buffer = deque(checkpoint.get("data_buffer", []), maxlen=self.buffer_size)

        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and "torch_cuda_random_state_all" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["torch_cuda_random_state_all"]])

        print(
            "resumed training checkpoint: {}, batch:{}, replay_buffer:{}, best_win_ratio:{:.3f}".format(
                path, self.current_batch, len(self.data_buffer), self.best_win_ratio
            )
        )

    def collect_selfplay_data(self, n_games: int = 1) -> None:
        for _ in range(n_games):
            _, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            self.data_buffer.extend(self.get_equi_data(play_data))

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = 0.0
        loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        for _ in range(self.epochs):
            loss, policy_loss, value_loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier,
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = float(
                np.mean(
                    np.sum(
                        old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                        axis=1,
                    )
                )
            )
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        old_var = np.var(np.array(winner_batch) - old_v.flatten())
        new_var = np.var(np.array(winner_batch) - new_v.flatten())
        target_var = np.var(np.array(winner_batch))
        explained_var_old = 1 - old_var / target_var if target_var > 0 else 0.0
        explained_var_new = 1 - new_var / target_var if target_var > 0 else 0.0
        target_values, target_counts = np.unique(np.array(winner_batch, dtype=np.float32), return_counts=True)
        target_count_map = {float(value): int(count) for value, count in zip(target_values, target_counts)}
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},effective_lr:{:.7f},loss:{},"
            "policy_loss:{},value_loss:{},entropy:{},replay_buffer:{},"
            "target_counts:-1={},0={},1={},"
            "explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl,
                self.lr_multiplier,
                self.learn_rate * self.lr_multiplier,
                loss,
                policy_loss,
                value_loss,
                entropy,
                len(self.data_buffer),
                target_count_map.get(-1.0, 0),
                target_count_map.get(0.0, 0),
                target_count_map.get(1.0, 0),
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def policy_evaluate(self, n_games: int | None = None) -> float:
        n_games = n_games or int(self.config.get("evaluation", {}).get("games", 10))
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
        )
        if self.eval_opponent == "heuristic":
            opponent_player = HeuristicPlayer(search_radius=self.heuristic_search_radius)
            opponent_label = f"heuristic_radius:{self.heuristic_search_radius}"
        elif self.eval_opponent == "pure_mcts":
            opponent_player = PureMCTSPlayer(c_puct=5, n_playout=self.pure_mcts_playout_num)
            opponent_label = f"pure_mcts_playouts:{self.pure_mcts_playout_num}"
        else:
            raise ValueError(f"unknown evaluation opponent: {self.eval_opponent}")
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player,
                opponent_player,
                start_player=i % 2,
                is_shown=0,
            )
            win_cnt[winner] += 1
        win_ratio = (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(
            "opponent:{}, win: {}, lose: {}, tie:{}".format(
                opponent_label, win_cnt[1], win_cnt[2], win_cnt[-1]
            )
        )
        return float(win_ratio)

    def run(self) -> None:
        try:
            for i in range(self.current_batch, self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                self.current_batch = i + 1
                print(f"batch i:{self.current_batch}, episode_len:{self.episode_len}")
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                if self.current_batch % self.check_freq == 0:
                    print(f"current self-play batch: {self.current_batch}")
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(self.current_model_path)
                    self.save_checkpoint()
                    if win_ratio > self.best_win_ratio:
                        print("New best policy")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(self.best_model_path)
                        self.save_checkpoint(self.best_checkpoint_path)
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
            if self.current_batch >= self.game_batch_num:
                self.policy_value_net.save_model(self.current_model_path)
                self.save_checkpoint()
        except KeyboardInterrupt:
            self.policy_value_net.save_model(self.current_model_path)
            self.save_checkpoint()
            print("\nquit")


def main() -> None:
    TrainPipeline().run()


if __name__ == "__main__":
    main()
