"""Step 3: teach the network to imitate the tree search.

    python scripts/imitate.py --games 600 --sims 400

1. The search from step 2 plays games against itself; every position is saved
   with what the search thought of it and how the game ended.
2. The network is trained to give the same answers from one look at the board.
3. The network then plays on its own, with no search at all, against a random
   player and against the search, to see how much it absorbed.

This is ordinary supervised learning, not yet reinforcement learning: the
network copies a teacher. In step 5 the teacher becomes the network itself.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from azc4.arena import mcts_player, network_player, play_match, random_player
from azc4.data import mcts_game
from azc4.network import AZNet, loss_fn, mirror, save


def generate(games: int, sims: int, workers: int, path: Path):
    if path.exists():
        d = np.load(path)
        print(f"loaded {len(d['values'])} saved positions from {path}")
        return d["planes"], d["policies"], d["values"], d["game_ids"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(mcts_game, [sims] * games, range(games), chunksize=4))
    planes = np.concatenate([r[0] for r in results])
    policies = np.concatenate([r[1] for r in results])
    values = np.concatenate([r[2] for r in results])
    game_ids = np.concatenate([np.full(len(r[2]), i) for i, r in enumerate(results)])
    np.savez_compressed(path, planes=planes, policies=policies, values=values, game_ids=game_ids)
    print(f"generated {games} games, {len(values)} positions -> {path}")
    return planes, policies, values, game_ids


def with_mirrors(planes, policies, values):
    flipped = [mirror(p, pi) for p, pi in zip(planes, policies)]
    return (np.concatenate([planes, np.stack([f[0] for f in flipped])]),
            np.concatenate([policies, np.stack([f[1] for f in flipped])]),
            np.concatenate([values, values]))


@torch.no_grad()
def evaluate(net, planes, policies, values) -> dict:
    net.eval()
    x, pi, v = (torch.from_numpy(a) for a in (planes, policies, values))
    loss, policy_loss, value_loss = loss_fn(net, x, pi, v)
    logits, value = net(x)
    decided = v != 0
    return {"loss": loss.item(), "policy_loss": policy_loss, "value_loss": value_loss,
            # how often the network's favourite move is the search's favourite
            "move_agreement": (logits.argmax(1) == pi.argmax(1)).float().mean().item(),
            # in games that had a winner, how often it names the right side
            "winner_accuracy": (torch.sign(value[decided]) == v[decided]).float().mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=600)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--match-games", type=int, default=100)
    args = p.parse_args()

    out = Path("runs") / "imitate"
    out.mkdir(parents=True, exist_ok=True)
    planes, policies, values, game_ids = generate(
        args.games, args.sims, args.workers, out / f"mcts{args.sims}_{args.games}games.npz")

    # Hold out whole games, not single positions: positions from one game are
    # near-copies of each other, so splitting them would leak answers.
    rng = np.random.default_rng(args.seed)
    held_out = rng.permutation(args.games)[: args.games // 10]
    is_val = np.isin(game_ids, held_out)
    train = with_mirrors(planes[~is_val], policies[~is_val], values[~is_val])
    val = (planes[is_val], policies[is_val], values[is_val])
    print(f"training on {len(train[2])} positions (with mirror images), "
          f"checking on {len(val[2])} from {len(held_out)} unseen games")

    torch.manual_seed(args.seed)
    net = AZNet()
    print(f"network: {sum(p.numel() for p in net.parameters()):,} weights")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    x, pi, v = (torch.from_numpy(a) for a in train)
    before = evaluate(net, *val)
    print(f"untrained: move agreement {before['move_agreement']:.1%}, "
          f"winner accuracy {before['winner_accuracy']:.1%}")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(len(v))
        for i in range(0, len(v), args.batch_size):
            idx = order[i: i + args.batch_size]
            loss, _, _ = loss_fn(net, x[idx], pi[idx], v[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch % 4 == 0 or epoch == args.epochs:
            m = evaluate(net, *val)
            print(f"epoch {epoch:2d}: unseen-game loss {m['loss']:.3f} "
                  f"(policy {m['policy_loss']:.3f}, value {m['value_loss']:.3f}), "
                  f"move agreement {m['move_agreement']:.1%}, "
                  f"winner accuracy {m['winner_accuracy']:.1%}")
    save(net, out / "net.pt")
    print(f"saved {out / 'net.pt'}")

    print("\nThe network playing with no search (it just plays its favourite move):")
    net_only = network_player(net)
    for name, opponent in [("random", random_player(rng)), ("mcts:50", mcts_player(50, rng)),
                           ("mcts:400", mcts_player(400, rng))]:
        r = play_match(net_only, opponent, args.match_games)
        print(f"  vs {name:9s} {r['wins']:3d} wins, {r['draws']:2d} draws, {r['losses']:3d} losses "
              f"-> score {r['score']:.1%} ({r['low']:.1%} to {r['high']:.1%})")


if __name__ == "__main__":  # required on Windows for parallel workers
    main()
