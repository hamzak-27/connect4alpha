"""Monte Carlo tree search (MCTS): choosing a move by imagining many games.

No learning happens here. For each move decision the player builds a tree of
"what if" continuations, one simulation at a time. Every simulation has four
phases:

  1. SELECT    walk down the tree from the current position, at each step
               picking the move that best balances "has worked well so far"
               against "has hardly been tried";
  2. EXPAND    on reaching a position with an untried move, add it to the tree;
  3. SIMULATE  from there, play random moves to the end of the game;
  4. BACK UP   send the result back up the path, so every position on it
               remembers one more game and who won it.

After all simulations, play the move that was tried most often. Good moves
attract more simulations, so the visit count is the search's verdict.
"""
from __future__ import annotations

import math

import numpy as np

from .game import COLS, ROWS, Connect4, _makes_four


class Node:
    """One position in the search tree, with what the search has learned about it."""

    __slots__ = ("game", "parent", "move", "children", "untried", "visits", "value_sum")

    def __init__(self, game: Connect4, parent: "Node | None" = None, move: int | None = None):
        self.game = game
        self.parent = parent
        self.move = move                    # the move that led here from the parent
        self.children: list[Node] = []
        self.untried = game.legal_moves()   # moves not yet added to the tree
        self.visits = 0
        # Total result of all simulations through this node, counted from the
        # point of view of the player who *made the move into it*. That is the
        # player choosing between this node and its siblings, so a high average
        # means "this was a good move to make".
        self.value_sum = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def ucb_score(parent: Node, child: Node, c: float) -> float:
    """Upper confidence bound: average result so far, plus a bonus that is
    large for moves tried rarely and shrinks as they are tried more.

    The bonus is what stops the search from fixating on the first move that
    happened to win: a neglected move keeps gaining appeal until it is
    checked again."""
    explore = c * math.sqrt(math.log(parent.visits) / child.visits)
    return child.mean_value + explore


def random_playout(game: Connect4, rng: np.random.Generator) -> int:
    """Play random moves from ``game`` to the end. Returns the winner (+1/-1) or 0.

    Works on one scratch copy of the board instead of creating a new position
    per move, because this is the part of the search that runs most."""
    if game.done:
        return game.winner
    board = game.board.copy()
    heights = (board != 0).sum(axis=0)  # discs already in each column
    player, moves = game.player, game.moves
    while moves < ROWS * COLS:
        open_cols = [c for c in range(COLS) if heights[c] < ROWS]
        col = open_cols[int(rng.integers(len(open_cols)))]
        row = heights[col]
        board[row, col] = player
        if _makes_four(board, row, col):
            return player
        heights[col] += 1
        moves += 1
        player = -player
    return 0


def search(game: Connect4, simulations: int, rng: np.random.Generator, c: float = 1.4) -> Node:
    """Run MCTS from ``game`` and return the root of the finished tree."""
    root = Node(game)
    for _ in range(simulations):
        node = root

        # 1. SELECT: follow the most promising known moves down the tree.
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ucb_score(node, ch, c))

        # 2. EXPAND: try one new move from this position.
        if node.untried:
            move = node.untried.pop(int(rng.integers(len(node.untried))))
            child = Node(node.game.play(move), parent=node, move=move)
            node.children.append(child)
            node = child

        # 3. SIMULATE: finish the game with random moves.
        winner = random_playout(node.game, rng)

        # 4. BACK UP: every position on the path records the outcome.
        while node is not None:
            node.visits += 1
            mover = -node.game.player  # who made the move into this node
            node.value_sum += winner * mover
            node = node.parent
    return root


def best_move(root: Node) -> int:
    """The most-visited move. More reliable than the highest average, which
    can belong to a move that was tried twice and got lucky."""
    return max(root.children, key=lambda ch: ch.visits).move


def describe(root: Node) -> str:
    """What the search thinks of each move, for display."""
    lines = []
    for ch in sorted(root.children, key=lambda ch: ch.move):
        win_chance = (ch.mean_value + 1) / 2
        lines.append(f"  column {ch.move}: tried {ch.visits:5d} times, "
                     f"estimated win chance {win_chance:.0%}")
    return "\n".join(lines)
