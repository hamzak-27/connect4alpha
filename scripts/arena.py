"""Matches between players, to measure how strong each one is.

    python scripts/arena.py mcts:200 random
    python scripts/arena.py az:100 mcts:400 --openings 4
    python scripts/arena.py az:50 net --net runs/imitate/net.pt --openings 4

Players:
    random        moves at random
    mcts:<n>      tree search with n random-playout simulations per move (no learning)
    net           the network alone: plays its favourite move, no search
    az:<n>        the network guiding a search of n simulations per move

--openings K starts each pair of games from K random moves. Use it whenever
both players are free of randomness (net, az), or every game will be the same.
The two players take turns going first. The score counts a draw as half a win.
"""
import argparse

import numpy as np

from azc4.arena import az_player, mcts_player, network_player, play_match, random_player
from azc4.network import load


def make_player(spec: str, rng: np.random.Generator, net_path: str):
    kind, _, n = spec.partition(":")
    if kind == "random":
        return random_player(rng)
    if kind == "net":
        return network_player(load(n or net_path))
    if kind == "mcts" and n.isdigit():
        return mcts_player(int(n), rng)
    if kind == "az" and n.isdigit():
        return az_player(load(net_path), int(n))
    raise SystemExit(f"unknown player '{spec}' (use random, mcts:<n>, net or az:<n>)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--net", default="runs/imitate/net.pt", help="saved network for net / az players")
    p.add_argument("--openings", type=int, default=0, help="random moves at the start of each pair of games")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    r = play_match(make_player(args.a, rng, args.net), make_player(args.b, rng, args.net),
                   args.games, rng, args.openings)
    print(f"{args.a} vs {args.b}, {r['games']} games: "
          f"{r['wins']} wins, {r['draws']} draws, {r['losses']} losses "
          f"-> score {r['score']:.1%} (95% interval {r['low']:.1%} to {r['high']:.1%})")
