"""Tree search guided by the neural network (the AlphaZero search).

Same skeleton as ``mcts.py``: select, expand, evaluate, back up. Two things
change, and they are the whole idea:

  * No random playouts. When the search reaches a new position it asks the
    network "who is winning here?" and uses that number directly.
  * No blind exploration. The network's policy says which moves deserve
    attention, so simulations go to plausible moves first.

The network makes the search efficient; the search corrects the network's
mistakes by actually looking ahead. Each covers the other's weakness.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .game import COLS, Connect4

# An evaluator answers two questions about a position: how promising is each
# column (probabilities, zero for illegal moves), and how good is the position
# for the player about to move (-1 to +1).
Evaluator = Callable[[Connect4], tuple[np.ndarray, float]]


class AZNode:
    __slots__ = ("game", "parent", "move", "prior", "children", "visits", "value_sum")

    def __init__(self, game: Connect4, parent: "AZNode | None" = None,
                 move: int | None = None, prior: float = 1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior                 # the network's first impression of this move
        self.children: list[AZNode] = []   # empty until the position has been evaluated
        self.visits = 0
        # As in mcts.py: results summed from the point of view of the player
        # who made the move into this node.
        self.value_sum = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def puct_score(parent: AZNode, child: AZNode, c_puct: float) -> float:
    """Average result so far, plus an exploration bonus scaled by the prior.

    Compared with the UCB rule in mcts.py, the bonus is multiplied by the
    network's prior: a move the network likes gets explored early, and a move
    it dismisses needs strong evidence before it gets many simulations. The
    bonus still shrinks as a move is visited, so over time the results of
    actually looking ahead outweigh the first impression."""
    explore = c_puct * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    return child.mean_value + explore


def az_search(game: Connect4, evaluate: Evaluator, simulations: int,
              c_puct: float = 1.5) -> AZNode:
    """Run a network-guided search from ``game`` and return the root."""
    root = AZNode(game)
    for _ in range(simulations):
        node = root

        # 1. SELECT: follow the best PUCT score down to a position not yet evaluated.
        while node.children:
            node = max(node.children, key=lambda ch: puct_score(node, ch, c_puct))

        # 2. EXPAND and EVALUATE. ``value`` is for the player to move at ``node``.
        if node.game.done:
            # A finished game needs no opinion: the result is known exactly.
            value = float(node.game.result_for(node.game.player))
        else:
            priors, value = evaluate(node.game)
            node.children = [AZNode(node.game.play(m), node, m, float(priors[m]))
                             for m in node.game.legal_moves()]

        # 3. BACK UP. The player who moved *into* this node is the opponent of
        # the player to move here, so the sign flips at every level going up.
        value = -value
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent
    return root


def visit_policy(root: AZNode) -> np.ndarray:
    """How the search divided its simulations among the columns. This is the
    search's considered opinion, and usually better than the network's raw
    policy, which is what makes it a good training target later."""
    visits = np.zeros(COLS, dtype=np.float32)
    for child in root.children:
        visits[child.move] = child.visits
    return visits / visits.sum()


def best_move(root: AZNode) -> int:
    return max(root.children, key=lambda ch: ch.visits).move


def describe(root: AZNode) -> str:
    """The network's first impression of each move next to the search's verdict."""
    lines = []
    for ch in sorted(root.children, key=lambda ch: ch.move):
        lines.append(f"  column {ch.move}: first impression {ch.prior:4.0%}, "
                     f"after search {ch.visits / max(root.visits - 1, 1):4.0%} of simulations, "
                     f"estimated win chance {(ch.mean_value + 1) / 2:.0%}")
    return "\n".join(lines)


def network_evaluator(net) -> Evaluator:
    from .network import predict

    return lambda game: predict(net, game)


def uniform_evaluator(game: Connect4) -> tuple[np.ndarray, float]:
    """Knows nothing: every legal move equally likely, every position even.
    Useful for testing the search on its own."""
    priors = np.zeros(COLS, dtype=np.float32)
    legal = game.legal_moves()
    priors[legal] = 1.0 / len(legal)
    return priors, 0.0
