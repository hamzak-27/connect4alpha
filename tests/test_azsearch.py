import numpy as np

from azc4.azsearch import (AZNode, az_search, best_move, network_evaluator, puct_score,
                           uniform_evaluator, visit_policy)
from azc4.game import COLS, Connect4
from azc4.network import AZNet


def play_moves(moves) -> Connect4:
    game = Connect4()
    for m in moves:
        game = game.play(m)
    return game


def test_finds_an_immediate_win_even_with_no_knowledge():
    root = az_search(play_moves([0, 0, 1, 1, 2, 2]), uniform_evaluator, 100)
    assert best_move(root) == 3
    winning = next(ch for ch in root.children if ch.move == 3)
    assert winning.mean_value == 1.0   # a certain win for the player who made the move


def test_blocks_the_opponents_win_even_with_no_knowledge():
    game = play_moves([0, 0, 1, 1, 2])   # O to move; X threatens column 3
    root = az_search(game, uniform_evaluator, 300)
    assert best_move(root) == 3
    # every other move is seen to lose: its value for O is negative
    assert all(ch.mean_value < 0 for ch in root.children if ch.move != 3)


def test_visits_add_up_and_policy_is_a_distribution():
    root = az_search(Connect4(), uniform_evaluator, 150)
    assert root.visits == 150
    assert sum(ch.visits for ch in root.children) == 149   # the first simulation only evaluated the root
    policy = visit_policy(root)
    assert policy.shape == (COLS,) and abs(policy.sum() - 1) < 1e-6


def test_full_columns_get_no_visits():
    game = play_moves([0, 0, 0, 0, 0, 0])
    root = az_search(game, uniform_evaluator, 60)
    assert visit_policy(root)[0] == 0
    assert all(ch.move != 0 for ch in root.children)


def test_prior_steers_early_exploration():
    parent = AZNode(Connect4())
    parent.visits = 10
    liked = AZNode(Connect4().play(3), parent, 3, prior=0.9)
    ignored = AZNode(Connect4().play(0), parent, 0, prior=0.01)
    assert puct_score(parent, liked, 1.5) > puct_score(parent, ignored, 1.5)
    # but enough bad results override a good first impression
    liked.visits, liked.value_sum = 50, -40.0
    assert puct_score(parent, liked, 1.5) < puct_score(parent, ignored, 1.5)


def test_search_overrules_a_wrong_first_impression():
    # An evaluator that is badly wrong: it loves column 6 and hates the winning column 3.
    def misguided(game):
        priors, value = uniform_evaluator(game)
        legal = game.legal_moves()
        priors[:] = 0
        priors[legal] = 0.02
        if 6 in legal:
            priors[6] = 1 - 0.02 * (len(legal) - 1)
        return priors, value

    root = az_search(play_moves([0, 0, 1, 1, 2, 2]), misguided, 200)
    assert best_move(root) == 3


def test_search_is_repeatable_and_works_with_a_real_network():
    net = AZNet(channels=16, blocks=1)
    a = visit_policy(az_search(play_moves([3, 3]), network_evaluator(net), 40))
    b = visit_policy(az_search(play_moves([3, 3]), network_evaluator(net), 40))
    assert np.array_equal(a, b)   # no randomness: same position, same answer


def test_root_noise_changes_only_the_roots_priors_and_keeps_them_a_distribution():
    game = play_moves([3, 3])
    plain = az_search(game, uniform_evaluator, 30)
    noisy = az_search(game, uniform_evaluator, 30, noise=(np.random.default_rng(0), 1.0, 0.25))
    plain_priors = [ch.prior for ch in plain.children]
    noisy_priors = [ch.prior for ch in noisy.children]
    assert plain_priors != noisy_priors
    assert abs(sum(noisy_priors) - 1) < 1e-6 and min(noisy_priors) > 0
    # one level down nothing was touched: still the evaluator's uniform priors
    grandchildren = [g for ch in noisy.children for g in ch.children]
    assert grandchildren and all(abs(g.prior - 1 / 7) < 1e-6 for g in grandchildren)
