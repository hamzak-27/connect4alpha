# Roadmap

Goal: an agent that learns Connect Four from nothing but playing against itself
(the AlphaZero method), strong enough that a person cannot easily beat it, with a
measured account of how it got there.

One step at a time; each step ends with something that runs and is tested.

- [x] **1. The game.** Rules, win detection, a terminal to play in. *Done: 13 tests,
  including 300 random games checked against a brute-force win detector.*
- [ ] **2. Search without learning.** Monte Carlo tree search (MCTS) that plans by
  playing out random games. Already a decent player, and the baseline to beat.
- [ ] **3. The neural network.** One network with two outputs: which moves look
  promising (policy) and who is winning (value).
- [ ] **4. Search guided by the network.** Replace random playouts with the network's
  judgement (the AlphaZero search).
- [ ] **5. Self-play training.** The agent plays itself, learns from the games, and
  repeats.
- [ ] **6. Measuring strength.** Elo ratings across training, matches against the
  pure-search player from step 2, several seeds.
- [ ] **7. A playable demo.** Play against the trained agent and see what it is thinking.
