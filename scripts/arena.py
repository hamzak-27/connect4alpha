"""Matches between players, to measure how strong each one is.

    python scripts/arena.py mcts:200 random --games 100
    python scripts/arena.py mcts:1000 mcts:100 --games 50

A player is "random", "mcts:<simulations per move>", or "net:<path to a saved
network>" (the network playing with no search). The two players take
turns going first. The score counts a draw as half a win.
"""
import argparse

import numpy as np

from azc4.arena import mcts_player, network_player, play_match, random_player
from azc4.network import load


def make_player(spec: str, rng: np.random.Generator):
    if spec == "random":
        return random_player(rng)
    kind, _, n = spec.partition(":")
    if kind == "net":
        return network_player(load(n))
    if kind == "mcts" and n.isdigit():
        return mcts_player(int(n), rng)
    raise SystemExit(f"unknown player '{spec}' (use 'random', 'mcts:<simulations>' or 'net:<path>')")


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
