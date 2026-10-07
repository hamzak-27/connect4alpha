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
- [x] **3. The neural network.** One network with two outputs: which moves look
  promising (policy) and who is winning (value). *Done: trained to imitate 600 games of
  400-simulation search. On unseen games its favourite move matches the search's 49% of
  the time (11% untrained). Playing with no search at all it beat random 100-0, scored
  82% against 50-simulation search and 29% against its 400-simulation teacher. The value
  head learned little from single-game outcomes; self-play should fix that.*
- [x] **4. Search guided by the network.** Replace random playouts with the network's
  judgement (the AlphaZero search). *Done, using the step 3 imitation network and random
  openings. Guided search with 50 simulations scored 73% against the network alone and
  75% against plain search with 50. It did not beat stronger plain search: 38.5% with 100
  simulations against 400, and 41.7% with 200 against the 1000-simulation baseline (the
  network alone scores 22.5% against 400). Search clearly improves on the network, but
  this network's weak value head holds it back. Self-play training is the fix.*
- [ ] **5. Self-play training.** The agent plays itself, learns from the games, and
  repeats.
- [ ] **6. Measuring strength.** Elo ratings across training, matches against the
  pure-search player from step 2, several seeds.
- [ ] **7. A playable demo.** Play against the trained agent and see what it is thinking.
