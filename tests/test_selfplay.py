import numpy as np

from azc4.azsearch import network_evaluator, uniform_evaluator
from azc4.game import COLS, ROWS
from azc4.network import AZNet, save
from azc4.selfplay import eval_worker, selfplay_game, selfplay_worker


def test_selfplay_game_gives_one_consistent_example_per_position():
    planes, policies, values, winner = selfplay_game(uniform_evaluator, 20, np.random.default_rng(0))
    n = len(values)
    assert planes.shape == (n, 2, ROWS, COLS) and policies.shape == (n, COLS)
    assert np.allclose(policies.sum(axis=1), 1)
    assert planes[0].sum() == 0                      # the first position is the empty board
    assert planes[1].sum() == 1                      # then one disc, and so on
    assert values[-1] >= 0                           # whoever moved last did not lose
    if winner != 0:                                  # labels alternate with the player to move
        assert (values[::-1][::2] == 1).all() and (values[::-1][1::2] == -1).all()
    # a full column is never given any probability
    full = planes[:, 0, ROWS - 1, :] + planes[:, 1, ROWS - 1, :] > 0
    assert (policies[full] == 0).all()


def test_selfplay_is_repeatable_for_a_seed_and_varies_between_seeds(tmp_path):
    save(AZNet(channels=8, blocks=1), tmp_path / "net.pt")
    a = selfplay_worker(str(tmp_path / "net.pt"), 2, 12, seed=1)
    b = selfplay_worker(str(tmp_path / "net.pt"), 2, 12, seed=1)
    c = selfplay_worker(str(tmp_path / "net.pt"), 2, 12, seed=2)
    assert np.array_equal(a["planes"], b["planes"]) and a["winners"] == b["winners"]
    assert a["lengths"] != c["lengths"] or not np.array_equal(a["planes"], c["planes"])
    assert len(a["values"]) == sum(a["lengths"])


def test_eval_worker_plays_the_requested_games(tmp_path):
    save(AZNet(channels=8, blocks=1), tmp_path / "net.pt")
    r = eval_worker(str(tmp_path / "net.pt"), 8, 8, games=4, seed=0)
    assert r["games"] == 4 and r["wins"] + r["draws"] + r["losses"] == 4
