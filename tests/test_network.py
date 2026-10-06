import numpy as np
import torch

from azc4.data import mcts_game
from azc4.game import COLS, ROWS, Connect4
from azc4.network import AZNet, encode, load, loss_fn, mirror, predict, save


def play_moves(moves) -> Connect4:
    game = Connect4()
    for m in moves:
        game = game.play(m)
    return game


def test_encoding_separates_my_discs_from_theirs():
    game = play_moves([3, 4])          # X at column 3, O at column 4, X to move
    planes = encode(game.canonical())
    assert planes.shape == (2, ROWS, COLS)
    assert planes[0, 0, 3] == 1 and planes[0].sum() == 1   # mine (X is to move)
    assert planes[1, 0, 4] == 1 and planes[1].sum() == 1   # theirs


def test_mirror_flips_board_and_policy_together():
    planes = encode(play_moves([0, 6, 0]).canonical())
    policy = np.array([0.7, 0.1, 0.1, 0.1, 0, 0, 0], dtype=np.float32)
    m_planes, m_policy = mirror(planes, policy)
    assert m_planes[1, 0, 6] == 1 and m_planes[1, 1, 6] == 1  # the column-0 stack is now in column 6
    assert m_policy[6] == np.float32(0.7)
    back_planes, back_policy = mirror(m_planes, m_policy)
    assert (back_planes == planes).all() and (back_policy == policy).all()


def test_outputs_have_the_right_shape_and_range():
    net = AZNet(channels=16, blocks=1)
    logits, value = net(torch.zeros(5, 2, ROWS, COLS))
    assert logits.shape == (5, COLS) and value.shape == (5,)
    assert (value.abs() <= 1).all()


def test_predict_never_suggests_a_full_column():
    game = play_moves([0, 0, 0, 0, 0, 0])   # column 0 is full
    probs, value = predict(AZNet(channels=16, blocks=1), game)
    assert probs[0] == 0 and abs(probs.sum() - 1) < 1e-6
    assert -1 <= value <= 1


def test_network_can_learn_a_small_batch():
    torch.manual_seed(0)
    planes, policies, values = mcts_game(simulations=30, seed=0)
    x, pi, v = (torch.from_numpy(a) for a in (planes, policies, values))
    net = AZNet(channels=16, blocks=1)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        net.train()
        loss, _, _ = loss_fn(net, x, pi, v)
        first = first or loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert loss.item() < 0.7 * first


def test_examples_from_a_game_are_consistent():
    planes, policies, values = mcts_game(simulations=30, seed=1)
    assert len(planes) == len(policies) == len(values)
    assert np.allclose(policies.sum(axis=1), 1)
    assert set(np.unique(values)) <= {-1.0, 0.0, 1.0}
    # players alternate, so in a decided game the labels alternate in sign
    if values[-1] != 0:
        assert (values[::-1][::2] == values[-1]).all() and (values[::-1][1::2] == -values[-1]).all()
    # the player who made the last move won (or it was a draw), never lost
    assert values[-1] >= 0


def test_save_and_load_round_trip(tmp_path):
    net = AZNet(channels=16, blocks=1)
    save(net, tmp_path / "n.pt")
    game = play_moves([3, 3, 2])
    a, b = predict(net, game), predict(load(tmp_path / "n.pt"), game)
    assert np.allclose(a[0], b[0]) and abs(a[1] - b[1]) < 1e-6
