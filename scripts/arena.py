"""Matches between players, to measure how strong each one is.

    python scripts/arena.py mcts:200 random --games 100
    python scripts/arena.py mcts:1000 mcts:100 --games 50

A player is "random" or "mcts:<simulations per move>". The two players take
turns going first. The score counts a draw as half a win.
"""
import argparse

import numpy as np

from azc4.arena import mcts_player, play_match, random_player


def make_player(spec: str, rng: np.random.Generator):
    if spec == "random":
        return random_player(rng)
    kind, _, n = spec.partition(":")
    if kind == "mcts" and n.isdigit():
        return mcts_player(int(n), rng)
    raise SystemExit(f"unknown player '{spec}' (use 'random' or 'mcts:<simulations>')")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    r = play_match(make_player(args.a, rng), make_player(args.b, rng), args.games)
    print(f"{args.a} vs {args.b}, {r['games']} games: "
          f"{r['wins']} wins, {r['draws']} draws, {r['losses']} losses "
          f"-> score {r['score']:.1%} (95% interval {r['low']:.1%} to {r['high']:.1%})")
