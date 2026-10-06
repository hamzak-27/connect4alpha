"""Players, and matches between them.

A *player* is any function that takes a position and returns a column. Putting
every kind of player behind that one shape means a random mover, a search, and
later a trained network can all be matched against each other the same way.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .game import Connect4
from .mcts import best_move, search

Player = Callable[[Connect4], int]


def random_player(rng: np.random.Generator) -> Player:
    return lambda game: int(rng.choice(game.legal_moves()))


def mcts_player(simulations: int, rng: np.random.Generator, c: float = 1.4) -> Player:
    return lambda game: best_move(search(game, simulations, rng, c))


def play_game(first: Player, second: Player) -> int:
    """One game. Returns +1 if ``first`` won, -1 if ``second`` won, 0 for a draw."""
    game = Connect4()
    players = {1: first, -1: second}
    while not game.done:
        game = game.play(players[game.player](game))
    return game.winner


def play_match(a: Player, b: Player, games: int) -> dict:
    """``games`` games between ``a`` and ``b``, alternating who moves first.

    Going first is an advantage in Connect Four, so each player gets it
    exactly half the time; otherwise the match would be measuring the seat,
    not the player."""
    wins = draws = losses = 0
    for i in range(games):
        result = play_game(a, b) if i % 2 == 0 else -play_game(b, a)
        wins += result == 1
        draws += result == 0
        losses += result == -1
    score = (wins + 0.5 * draws) / games  # a draw counts as half a win
    low, high = wilson_interval(score, games)
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "score": score, "low": low, "high": high}


def wilson_interval(share: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a win rate measured over ``n`` games. With few games
    the true rate could be far from what was observed; this says how far."""
    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
