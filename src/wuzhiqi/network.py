"""PyTorch policy-value network for AlphaZero-style Gomoku."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - exercised only without deps
    torch = None
    nn = None
    F = None
    optim = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for wuzhiqi.network") from _TORCH_IMPORT_ERROR


def set_learning_rate(optimizer: "optim.Optimizer", lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if nn is not None:

    class Net(nn.Module):
        """Compact policy-value CNN adapted from the AlphaZero_Gomoku template."""

        def __init__(self, board_width: int, board_height: int) -> None:
            super().__init__()
            self.board_width = board_width
            self.board_height = board_height
            self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
            self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
            self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
            self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
            self.val_fc2 = nn.Linear(64, 1)

        def forward(self, state_input):
            x = F.relu(self.conv1(state_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x_act = F.relu(self.act_conv1(x))
            x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
            x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
            x_val = F.relu(self.val_conv1(x))
            x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
            x_val = F.relu(self.val_fc1(x_val))
            x_val = torch.tanh(self.val_fc2(x_val))
            return x_act, x_val

else:

    class Net:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


class PolicyValueNet:
    """Policy-value network wrapper used by MCTS and the trainer."""

    def __init__(
        self,
        board_width: int,
        board_height: int,
        model_file: str | Path | None = None,
        device: str | None = None,
        weight_decay: float = 1e-4,
    ) -> None:
        _require_torch()
        self.board_width = board_width
        self.board_height = board_height
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.l2_const = weight_decay
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            values = value.cpu().numpy()
        return act_probs, values

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, 4, self.board_width, self.board_height)
        )
        self.policy_value_net.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(current_state).float().to(self.device)
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            value_scalar = float(value.cpu().numpy()[0][0])
        return zip(legal_positions, act_probs[legal_positions]), value_scalar

    def train_step(self, state_batch, mcts_probs, winner_batch, lr: float):
        self.policy_value_net.train()
        state_tensor = torch.as_tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        mcts_probs_tensor = torch.as_tensor(np.array(mcts_probs), dtype=torch.float32, device=self.device)
        winner_tensor = torch.as_tensor(np.array(winner_batch), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        log_act_probs, value = self.policy_value_net(state_tensor)
        value_loss = F.mse_loss(value.view(-1), winner_tensor)
        policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_act_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))
        return float(loss.item()), float(entropy.item())

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def load_policy_param(self, net_params) -> None:
        self.policy_value_net.load_state_dict(net_params)

    def save_model(self, model_file: str | Path) -> None:
        model_path = Path(model_file)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_policy_param(), model_path)


AlphaZeroNet = Net
