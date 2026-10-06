"""The rules must be exactly right: an agent trained on a buggy game learns the bugs."""
import numpy as np
import pytest

from azc4.game import COLS, ROWS, Connect4


def play_moves(moves) -> Connect4:
    game = Connect4()
    for m in moves:
        game = game.play(m)
    return game


def brute_force_has_four(board: np.ndarray, player: int) -> bool:
    """A slow, obviously-correct check: look at every possible line of four."""
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player
                       for rr, cc in cells):
                    return True
    return False


def test_new_game():
    game = Connect4()
    assert game.player == 1 and not game.done
    assert game.legal_moves() == list(range(COLS))


def test_discs_stack_and_turns_alternate():
    game = play_moves([3, 3, 3])
    assert list(game.board[:3, 3]) == [1, -1, 1]
    assert game.player == -1


def test_play_does_not_change_the_original_position():
    start = Connect4()
    start.play(0)
    assert not start.board.any() and start.player == 1


@pytest.mark.parametrize("moves, winner", [
    ([0, 0, 1, 1, 2, 2, 3], 1),                    # across
    ([0, 1, 0, 1, 0, 1, 0], 1),                    # up
    ([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3], 1),        # diagonal rising to the right
    ([6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3], 1),        # diagonal rising to the left
    ([6, 0, 0, 1, 1, 2, 2, 3], -1),                # the second player can win too
])
def test_wins_in_every_direction(moves, winner):
    game = play_moves(moves)
    assert game.done and game.winner == winner
    assert game.legal_moves() == []
    assert game.result_for(winner) == 1 and game.result_for(-winner) == -1


def test_three_in_a_row_is_not_a_win():
    assert not play_moves([0, 0, 1, 1, 2]).done


def test_full_column_and_finished_game_reject_moves():
    game = play_moves([0, 0, 0, 0, 0, 0])
    assert 0 not in game.legal_moves()
    with pytest.raises(ValueError):
        game.play(0)
    with pytest.raises(ValueError):
        play_moves([0, 0, 1, 1, 2, 2, 3]).play(4)


def test_canonical_view_is_from_the_mover():
    game = play_moves([3])            # player 1 has a disc at the bottom of column 3
    assert game.player == -1
    assert game.board[0, 3] == 1      # true board: first player's disc
    assert game.canonical()[0, 3] == -1  # to the player now moving, it is the opponent's


def test_full_board_without_a_line_is_a_draw():
    moves = [3, 3, 5, 2, 1, 2, 0, 0, 5, 0, 1, 4, 6, 0, 2, 4, 3, 5, 5, 2, 4, 2, 2, 3, 0, 1, 4, 4,
             1, 0, 4, 6, 6, 5, 6, 6, 3, 3, 5, 6, 1, 1]
    game = play_moves(moves)
    assert game.moves == ROWS * COLS and game.winner == 0
    assert game.result_for(1) == 0 and game.result_for(-1) == 0
    assert game.legal_moves() == []


def test_random_games_agree_with_brute_force():
    rng = np.random.default_rng(0)
    winners = set()
    for _ in range(300):
        game = Connect4()
        while not game.done:
            # before the final move, nobody may already have four in a row
            assert not brute_force_has_four(game.board, 1)
            assert not brute_force_has_four(game.board, -1)
            game = game.play(int(rng.choice(game.legal_moves())))
        assert game.moves <= ROWS * COLS
        if game.winner != 0:
            assert brute_force_has_four(game.board, game.winner)
            assert not brute_force_has_four(game.board, -game.winner)
        winners.add(game.winner)
    assert {1, -1} <= winners  # both sides win some games
