"""Train an agent by self-play, starting from a network that knows nothing.

    python scripts/train.py --name run1
    python scripts/train.py --name run1          # run again to continue where it stopped

Each iteration: play games against itself, train on the recent games, save.
Everything is saved per iteration in runs/selfplay/<name>/, so a stopped run
loses at most one iteration.
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from azc4.network import AZNet, load, loss_fn, save
from azc4.selfplay import eval_worker, selfplay_worker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--games", type=int, default=120, help="self-play games per iteration")
    p.add_argument("--sims", type=int, default=100, help="search simulations per move in self-play")
    p.add_argument("--workers", type=int, default=4, help="0 = run in this process")
    p.add_argument("--window", type=int, default=8,
                   help="train on the games of this many most recent iterations")
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--eval-opponent", type=int, default=200,
                   help="simulations of the plain-search opponent in check matches")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_parallel(pool, fn, jobs):
    """Run ``fn(*job)`` for each job, in the pool if there is one."""
    if pool is None:
        return [fn(*job) for job in jobs]
    return [f.result() for f in [pool.submit(fn, *job) for job in jobs]]


def split(total: int, parts: int) -> list[int]:
    """Divide ``total`` games as evenly as possible among ``parts`` workers."""
    parts = max(1, min(parts, total))
    return [total // parts + (i < total % parts) for i in range(parts)]


def train_network(net, optim, data: list[dict], steps: int, batch_size: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    planes = torch.from_numpy(np.concatenate([d["planes"] for d in data]))
    policies = torch.from_numpy(np.concatenate([d["policies"] for d in data]))
    values = torch.from_numpy(np.concatenate([d["values"] for d in data]))
    net.train()
    policy_losses, value_losses = [], []
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(values), batch_size))
        x, pi, v = planes[idx], policies[idx], values[idx]
        # Mirror half of each batch left-to-right: a free second example.
        flip = torch.from_numpy(rng.random(batch_size) < 0.5)
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        pi = torch.where(flip[:, None], pi.flip(-1), pi)
        loss, policy_loss, value_loss = loss_fn(net, x, pi, v)
        optim.zero_grad()
        loss.backward()
        optim.step()
        policy_losses.append(policy_loss), value_losses.append(value_loss)
    net.eval()
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def main():
    args = parse_args()
    out = Path("runs") / "selfplay" / args.name
    out.mkdir(parents=True, exist_ok=True)
    net_file = lambda i: out / f"net_{i:03d}.pt"
    data_file = lambda i: out / f"data_{i:03d}.npz"

    # ---- start fresh, or continue from the last finished iteration
    done = 0
    while net_file(done + 1).exists() and data_file(done + 1).exists():
        done += 1
    torch.manual_seed(args.seed)
    if done == 0:
        net = AZNet(args.channels, args.blocks)
        save(net, net_file(0))   # iteration 0: a network with random weights
    else:
        net = load(net_file(done))
        print(f"continuing after iteration {done}")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    if done and (out / "optim.pt").exists():
        optim.load_state_dict(torch.load(out / "optim.pt"))

    log_path = out / "log.csv"
    fields = ["iteration", "games_total", "positions", "avg_length", "first_player_wins",
              "draws", "policy_loss", "value_loss", "eval_score", "eval_low", "eval_high",
              "minutes"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 0 else None
    workers = max(args.workers, 1)
    start = time.time()
    try:
        for it in range(done + 1, args.iterations + 1):
            # 1. PLAY: the current network plays itself.
            jobs = [(str(net_file(it - 1)), n, args.sims, args.seed * 100_000 + it * 100 + w)
                    for w, n in enumerate(split(args.games, workers))]
            results = run_parallel(pool, selfplay_worker, jobs)
            new = {k: np.concatenate([r[k] for r in results])
                   for k in ("planes", "policies", "values")}
            winners = np.array([w for r in results for w in r["winners"]])
            lengths = np.array([n for r in results for n in r["lengths"]])
            np.savez_compressed(data_file(it), **new)

            # 2. LEARN: train on the games of the most recent iterations. Older
            # games came from a weaker player, so they are dropped.
            recent = [dict(np.load(data_file(i))) for i in range(max(1, it - args.window + 1), it + 1)]
            rng = np.random.default_rng(args.seed * 1000 + it)
            policy_loss, value_loss = train_network(net, optim, recent, args.train_steps,
                                                    args.batch_size, rng)
            save(net, net_file(it))
            torch.save(optim.state_dict(), out / "optim.pt")

            # 3. CHECK (now and then): a short match against plain tree search.
            row = {"iteration": it, "games_total": it * args.games, "positions": len(new["values"]),
                   "avg_length": round(float(lengths.mean()), 1),
                   "first_player_wins": round(float((winners == 1).mean()), 3),
                   "draws": round(float((winners == 0).mean()), 3),
                   "policy_loss": round(policy_loss, 4), "value_loss": round(value_loss, 4),
                   "minutes": round((time.time() - start) / 60, 1)}
            line = (f"iteration {it:2d}: {row['games_total']:5d} games, "
                    f"avg length {row['avg_length']:4.1f}, policy loss {policy_loss:.3f}, "
                    f"value loss {value_loss:.3f}")
            if it % args.eval_every == 0 or it == args.iterations:
                jobs = [(str(net_file(it)), args.eval_sims, args.eval_opponent, 2 * (n // 2 or 1),
                         args.seed * 7919 + it * 10 + w)
                        for w, n in enumerate(split(args.eval_games, workers))]
                parts = run_parallel(pool, eval_worker, jobs)
                from azc4.arena import wilson_interval
                games = sum(p["games"] for p in parts)
                score = sum(p["wins"] + 0.5 * p["draws"] for p in parts) / games
                low, high = wilson_interval(score, games)
                row.update(eval_score=round(score, 3), eval_low=round(low, 3), eval_high=round(high, 3))
                line += (f" | search({args.eval_sims}) vs plain search({args.eval_opponent}): "
                         f"{score:.0%} ({low:.0%} to {high:.0%})")
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
            print(line + f" | {row['minutes']:.0f} min", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"final network: {net_file(args.iterations)}")


if __name__ == "__main__":  # required on Windows for parallel workers
    main()
