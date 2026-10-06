"""Turning games into training examples.

Each position in a finished game becomes one example with three parts:
  planes   the board as the network sees it
  policy   how the search split its simulations across the 7 columns
  value    how the game ended for the player who was about to move (+1/0/-1)
"""
from __future__ import annotations

import numpy as np

from .game import COLS, Connect4
from .mcts import search
from .network import encode

SAMPLED_MOVES = 8  # early moves are sampled, for variety; later ones are the search's best


def mcts_game(simulations: int, seed: int):
    """One game of the tree search playing itself. Returns the examples from it."""
    rng = np.random.default_rng(seed)
    game = Connect4()
    boards, policies, movers = [], [], []
    while not game.done:
        root = search(game, simulations, rng)
        visits = np.zeros(COLS, dtype=np.float32)
        for child in root.children:
            visits[child.move] = child.visits
        policy = visits / visits.sum()
        boards.append(encode(game.canonical()))
        policies.append(policy)
        movers.append(game.player)
        if game.moves < SAMPLED_MOVES:
            # Without this every game would open the same way and the network
            # would only ever see a narrow slice of possible positions.
            move = int(rng.choice(COLS, p=policy))
        else:
            move = int(visits.argmax())
        game = game.play(move)
    values = np.array([game.result_for(m) for m in movers], dtype=np.float32)
    return np.stack(boards), np.stack(policies), values
