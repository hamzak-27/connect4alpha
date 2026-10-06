"""Play Connect Four in the terminal.

    python scripts/play.py            you (X) against a player that moves at random
    python scripts/play.py --demo     watch two random players, then see who tends to win
"""
import argparse

import numpy as np

from azc4.game import Connect4


def random_player(game: Connect4, rng: np.random.Generator) -> int:
    return int(rng.choice(game.legal_moves()))


def human_player(game: Connect4) -> int:
    while True:
        answer = input(f"Your move {game.legal_moves()}: ").strip()
        if answer.isdigit() and int(answer) in game.legal_moves():
            return int(answer)
        print("Not a legal column, try again.")


def demo(games: int = 2000) -> None:
    rng = np.random.default_rng(0)
    tally = {1: 0, -1: 0, 0: 0}
    lengths = []
    for i in range(games):
        game = Connect4()
        while not game.done:
            game = game.play(random_player(game, rng))
        tally[game.winner] += 1
        lengths.append(game.moves)
        if i == 0:
            print("One finished random game:\n" + game.render() + "\n")
    print(f"{games} games between two random players:")
    print(f"  first player wins  {tally[1] / games:.1%}")
    print(f"  second player wins {tally[-1] / games:.1%}")
    print(f"  draws              {tally[0] / games:.1%}")
    print(f"  average length     {np.mean(lengths):.1f} moves")


def play_human() -> None:
    rng = np.random.default_rng()
    game = Connect4()
    while not game.done:
        print("\n" + game.render())
        move = human_player(game) if game.player == 1 else random_player(game, rng)
        game = game.play(move)
    print("\n" + game.render())
    print({1: "You win!", -1: "You lose.", 0: "Draw."}[game.winner])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    demo() if args.demo else play_human()
