# Roadmap

Goal: an agent that learns Connect Four from nothing but playing against itself
(the AlphaZero method), strong enough that a person cannot easily beat it, with a
measured account of how it got there.

One step at a time; each step ends with something that runs and is tested.

- [x] **1. The game.** Rules, win detection, a terminal to play in. *Done: 13 tests,
  including 300 random games checked against a brute-force win detector.*
- [x] **2. Search without learning.** Monte Carlo tree search (MCTS) that plans by
  playing out random games. *Done: with 50 simulations per move it beats a random player
  98.5% of the time, with 200 it won 100 of 100. More thinking keeps helping: 200
  simulations beat 50 with an 89.5% score, and 1000 beat 200 with 80.8%. This
  1000-simulation player is the baseline the learned agent has to beat.*
- [ ] **3. The neural network.** One network with two outputs: which moves look
  promising (policy) and who is winning (value).
- [ ] **4. Search guided by the network.** Replace random playouts with the network's
  judgement (the AlphaZero search).
- [ ] **5. Self-play training.** The agent plays itself, learns from the games, and
  repeats.
- [ ] **6. Measuring strength.** Elo ratings across training, matches against the
  pure-search player from step 2, several seeds.
- [ ] **7. A playable demo.** Play against the trained agent and see what it is thinking.
