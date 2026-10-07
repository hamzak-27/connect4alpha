"""The neural network: one glance at a board, two answers.

  policy  for each of the 7 columns, how promising is playing there?
  value   who is winning, from -1 (the player to move loses) to +1 (wins)?

The search in ``mcts.py`` gets these answers by playing thousands of random
games. The network gives them instantly, from patterns it has learned. That
is the difference between working something out and knowing it.

Both answers come from one network with a shared "body" and two small
"heads", because judging moves and judging positions need the same
understanding of the board (where the threats are, who controls the centre).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import COLS, ROWS, Connect4


def encode(canonical: np.ndarray) -> np.ndarray:
    """Turn a board (as seen by the player to move) into the network's input:
    two 6x7 grids of 0s and 1s, "where are my discs" and "where are theirs".

    Separate grids work better than one grid of +1/-1 because the first layer
    can then look for "my three in a row" and "their three in a row" directly."""
    return np.stack([canonical == 1, canonical == -1]).astype(np.float32)


def mirror(planes: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The left-right mirror image of a position, with the move preferences
    mirrored to match. Connect Four is symmetric, so every position we collect
    gives a second training example for free."""
    return planes[..., ::-1].copy(), policy[::-1].copy()


class ResidualBlock(nn.Module):
    """Two convolutions plus a shortcut that adds the input back on.

    The shortcut lets a block learn "the input, plus a small correction",
    which is what makes networks many layers deep trainable (He et al., 2016)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class AZNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.channels, self.blocks = channels, blocks
        # Body: convolutions slide a small 3x3 window over the board, so a
        # pattern learned in one corner is recognised everywhere.
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        # Policy head: one score per column.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, COLS))
        # Value head: a single number squashed into [-1, 1].
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
            nn.Linear(ROWS * COLS, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy scores before softmax, value), for a batch of boards."""
        h = self.body(self.stem(planes))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def loss_fn(net: AZNet, planes, target_policy, target_value):
    """How wrong the network is on a batch, as two parts added together.

    policy part: cross-entropy between the network's move preferences and the
                 target preferences (low when it favours the same moves);
    value part:  squared error between its "who is winning" and what happened.
    """
    logits, value = net(planes)
    policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    return policy_loss + value_loss, policy_loss.item(), value_loss.item()


def predict(net: AZNet, game: Connect4) -> tuple[np.ndarray, float]:
    """The network's answers for one position: a probability for each column
    (zero for full columns) and the value for the player to move."""
    net.eval()
    return predict_prepared(net, game)


@torch.inference_mode()
def predict_prepared(net: AZNet, game: Connect4) -> tuple[np.ndarray, float]:
    """``predict`` for a network already in evaluation mode (the fast path
    used inside the search, which calls this thousands of times)."""
    planes = torch.from_numpy(encode(game.canonical())).unsqueeze(0)
    logits, value = net(planes)
    probs = torch.softmax(logits[0], dim=0).numpy()
    legal = np.zeros(COLS, dtype=bool)
    legal[game.legal_moves()] = True
    probs = np.where(legal, probs, 0.0)
    total = probs.sum()
    probs = probs / total if total > 0 else legal / legal.sum()
    return probs, float(value[0])


def save(net: AZNet, path) -> None:
    torch.save({"channels": net.channels, "blocks": net.blocks, "state": net.state_dict()}, path)


def load(path) -> AZNet:
    ckpt = torch.load(path, map_location="cpu")
    net = AZNet(ckpt["channels"], ckpt["blocks"])
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net
