"""Self-play: the agent generates its own training data by playing itself.

One round of the AlphaZero loop:

    1. PLAY    the current network, guided by search, plays games against itself;
    2. LEARN   the network is trained to predict what the search concluded
               (policy target) and who ended up winning (value target);
    3. REPEAT  with the improved network.

Why this improves anything: the search looks ahead, so its verdict on a
position is better than the network's first impression of it. Training pulls
the first impression towards the verdict. Next round the search starts from
better impressions, so its verdicts are better still.

Nothing here knows what good Connect Four looks like. The only outside
information is the rules and who won.
"""
from __future__ import annotations

import numpy as np
import torch

from .arena import az_player, mcts_player, play_match
from .azsearch import az_search, network_evaluator, visit_policy
from .game import COLS, Connect4
from .network import encode, load

TEMPERATURE_MOVES = 10   # how many opening moves are sampled instead of played greedily
NOISE_ALPHA = 1.0        # shape of the random noise added to the root's priors
NOISE_FRACTION = 0.25    # how much of the root's priors is replaced by noise


def selfplay_game(evaluate, simulations: int, rng: np.random.Generator):
    """One game of the agent against itself. Returns one training example per
    position: the board, the search's move distribution, and the final result
    from the point of view of the player who was to move."""
    game = Connect4()
    boards, policies, movers = [], [], []
    while not game.done:
        # Noise at the root makes the search occasionally try moves the
        # network currently dismisses. Without it the agent would keep
        # replaying the lines it already believes in and never discover
        # that a neglected move is good.
        root = az_search(game, evaluate, simulations,
                         noise=(rng, NOISE_ALPHA, NOISE_FRACTION))
        policy = visit_policy(root)
        boards.append(encode(game.canonical()))
        policies.append(policy)
        movers.append(game.player)
        if game.moves < TEMPERATURE_MOVES:
            # Early on, pick moves in proportion to how much the search liked
            # them, so different games explore different openings.
            move = int(rng.choice(COLS, p=policy))
        else:
            move = int(policy.argmax())
        game = game.play(move)
    values = np.array([game.result_for(m) for m in movers], dtype=np.float32)
    return np.stack(boards), np.stack(policies), values, game.winner


def selfplay_worker(net_path: str, games: int, simulations: int, seed: int) -> dict:
    """Play ``games`` self-play games with the saved network. Runs in a worker process."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    evaluate = network_evaluator(load(net_path))
    planes, policies, values, winners, lengths = [], [], [], [], []
    for _ in range(games):
        b, p, v, winner = selfplay_game(evaluate, simulations, rng)
        planes.append(b), policies.append(p), values.append(v)
        winners.append(winner), lengths.append(len(v))
    return {"planes": np.concatenate(planes), "policies": np.concatenate(policies),
            "values": np.concatenate(values), "winners": winners, "lengths": lengths}


def eval_worker(net_path: str, simulations: int, opponent_simulations: int, games: int,
                seed: int) -> dict:
    """A short match: the saved network with search, against plain tree search."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    return play_match(az_player(load(net_path), simulations),
                      mcts_player(opponent_simulations, rng), games, rng, opening_plies=4)
