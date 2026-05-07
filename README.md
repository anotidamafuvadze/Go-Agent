# Go Agent

A Go-playing AI agent built in Python that combines Monte Carlo Tree Search (MCTS) with a trained neural value network. The core idea is that random rollouts — the standard way MCTS evaluates positions — are noisy, since a randomly played game rarely resembles how strong players would actually continue. Replacing rollouts with a network trained on real game outcomes lets the agent evaluate positions more accurately within the same time budget.

## How It Works

**Hybrid Evaluation**

Rather than replacing rollouts entirely, the agent uses a hybrid approach: every fifth MCTS iteration evaluates the leaf node using the value network, while the remaining four use random rollouts. This prevents the network from fully dominating when its predictions are uncertain and keeps the agent grounded in real game dynamics.

**Feature Engineering**

The value network takes 21 hand-crafted board features as input, extended from a baseline of 11. The added features are:

- **Center control** — scores stones by proximity to the center, since central positions tend to have more board influence
- **Capture threat** — counts stones adjacent to opponent pieces and therefore under immediate capture risk
- **Edge stones** — counts stones on the board boundary, which have fewer liberties and are generally weaker positions
- **Game phase indicator** — derived from the fraction of empty spaces remaining, helping the network distinguish early, middle, and endgame positions

All features are computed relative to the current player and work on both 5x5 and 9x9 boards without modification.

**Self-Play Data Generation**

For 9x9, no labeled training data was available, so a self-play pipeline bootstraps it. The process is: train a value network, use it to build a stronger agent, have that agent play itself, then retrain on the combined dataset. In practice the gains are modest — an imperfect agent tends to repeat the same mistakes, so self-play reinforces those patterns rather than introducing new signal. The 9x9 network is weaker than the 5x5 one as a result.

## Results

The hybrid value network approach improved win rate from ~50% with plain MCTS to ~60% on 5x5 boards.

## What Didn't Work

A policy network was trained alongside the value network to guide which actions MCTS explores first — an idea inspired by AlphaGo. It was trained via imitation learning on winning players' moves. In every configuration tested (full ranking, top-3 priority, 15–30% guided expansion), the policy network hurt performance, dropping win rate as low as 10%. The output distribution was near-uniform with a maximum action probability of ~0.08, meaning the network was barely better than random. The likely cause is that the features describe the current board state rather than the result of each candidate move, so the network lacked the information needed to distinguish good moves from bad ones. Policy guidance is disabled in the final agent.

## Stack

Python, PyTorch, NumPy

