import numpy as np

from azc4.arena import mcts_player, play_match, random_player, wilson_interval
from azc4.game import Connect4
from azc4.mcts import Node, best_move, random_playout, search


def play_moves(moves) -> Connect4:
    game = Connect4()
    for m in moves:
        game = game.play(m)
    return game


def test_takes_a_win_when_one_is_available():
    # X has three in a row on the bottom (columns 0-2); column 3 wins at once.
    game = play_moves([0, 0, 1, 1, 2, 2])
    root = search(game, 300, np.random.default_rng(0))
    assert best_move(root) == 3


def test_blocks_the_opponents_winning_move():
    # X threatens column 3. It is O's turn, and anything but 3 loses next move.
    game = play_moves([0, 0, 1, 1, 2])
    assert game.player == -1
    root = search(game, 800, np.random.default_rng(0))
    assert best_move(root) == 3


def test_every_simulation_is_counted_once():
    root = search(Connect4(), 200, np.random.default_rng(0))
    assert root.visits == 200
    assert sum(ch.visits for ch in root.children) == 200  # each one went through exactly one move
    assert all(-1.0 <= ch.mean_value <= 1.0 for ch in root.children)


def test_playout_from_a_finished_game_returns_its_winner():
    won = play_moves([0, 0, 1, 1, 2, 2, 3])
    assert random_playout(won, np.random.default_rng(0)) == 1
    winners = {random_playout(Connect4(), np.random.default_rng(s)) for s in range(40)}
    assert winners <= {1, -1, 0} and {1, -1} <= winners


def test_playout_does_not_change_the_position():
    game = play_moves([3, 3])
    before = game.board.copy()
    random_playout(game, np.random.default_rng(0))
    assert (game.board == before).all() and game.moves == 2


def test_value_is_from_the_point_of_view_of_whoever_moved():
    # After X's winning move the node is a certain win for X, who made the move.
    won = play_moves([0, 0, 1, 1, 2, 2, 3])
    root = search(play_moves([0, 0, 1, 1, 2, 2]), 50, np.random.default_rng(0))
    winning_child = next(ch for ch in root.children if ch.move == 3)
    assert winning_child.game.winner == won.winner == 1
    assert winning_child.mean_value == 1.0


def test_search_beats_random_play():
    rng = np.random.default_rng(0)
    result = play_match(mcts_player(60, rng), random_player(rng), games=20)
    assert result["wins"] >= 18


def test_wilson_interval_narrows_with_more_games():
    low_few, high_few = wilson_interval(0.6, 10)
    low_many, high_many = wilson_interval(0.6, 1000)
    assert low_few < low_many < 0.6 < high_many < high_few


def test_openings_are_shared_by_both_seats_and_never_finished():
    from azc4.arena import random_opening

    rng = np.random.default_rng(0)
    for _ in range(50):
        game = random_opening(6, rng)
        assert game.moves == 6 and not game.done

    # Record the starting position each deterministic player is asked to move from.
    seen = []

    def recorder(game):
        if game.moves in (4, 5):
            seen.append((game.moves, game.board.tobytes()))
        return game.legal_moves()[0]

    play_match(recorder, recorder, games=4, rng=np.random.default_rng(1), opening_plies=4)
    starts = [board for moves, board in seen if moves == 4]
    assert len(starts) == 4
    assert starts[0] == starts[1] and starts[2] == starts[3]   # each opening is played twice
    assert starts[0] != starts[2]                              # and the openings differ
