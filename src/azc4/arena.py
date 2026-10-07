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


def network_player(net) -> Player:
    """The network on its own, with no search: it plays whichever column it
    rates highest. This is pure instinct, one look at the board."""
    from .network import predict

    return lambda game: int(predict(net, game)[0].argmax())


def az_player(net, simulations: int, c_puct: float = 1.5) -> Player:
    """The network and the search together: instinct guiding the planning."""
    from .azsearch import az_search, best_move as az_best_move, network_evaluator

    evaluate = network_evaluator(net)
    return lambda game: az_best_move(az_search(game, evaluate, simulations, c_puct))


def random_opening(plies: int, rng: np.random.Generator) -> Connect4:
    """A position reached by ``plies`` random moves (never a finished game)."""
    while True:
        game = Connect4()
        for _ in range(plies):
            game = game.play(int(rng.choice(game.legal_moves())))
            if game.done:
                break
        if not game.done:
            return game


def play_game(first: Player, second: Player, start: Connect4 | None = None) -> int:
    """One game from ``start`` (default: the empty board). ``first`` plays the
    side that moves first in the whole game, ``second`` the other side.
    Returns +1 if ``first`` won, -1 if ``second`` won, 0 for a draw."""
    game = start or Connect4()
    players = {1: first, -1: second}
    while not game.done:
        game = game.play(players[game.player](game))
    return game.winner


def play_match(a: Player, b: Player, games: int, rng: np.random.Generator | None = None,
               opening_plies: int = 0) -> dict:
    """``games`` games between ``a`` and ``b``, alternating who moves first.

    Going first is an advantage in Connect Four, so each player gets it
    exactly half the time; otherwise the match would be measuring the seat,
    not the player.

    ``opening_plies`` starts each pair of games from a random opening. Two
    players with no randomness in them would otherwise play the same game
    over and over, and 100 copies of one game are one piece of evidence, not
    100. Each opening is played twice, once with each player on each side,
    so a lopsided opening cannot favour either of them."""
    wins = draws = losses = 0
    start = None
    for i in range(games):
        if i % 2 == 0:
            start = random_opening(opening_plies, rng) if opening_plies else None
            result = play_game(a, b, start)
        else:
            result = -play_game(b, a, start)
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
