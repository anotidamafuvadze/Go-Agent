[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pU4J2i_g)
# README

Tell us about your implementation!

Answer the conceptual questions!

# Task 1.1

Running 100 games of Greedy vs Random, the greedy agent won 56% of the time and the random agent won 44%. These results are not surprising for two reasons. First, the greedy agent uses a heuristic to guide its decisions toward moves more likely to lead to a win. A random agent has no such guidance, so any wins it achieves are essentially luck. We would therefore expect the greedy agent to win more often. Second, the margin isn't overwhelming, which also makes sense. The heuristic used (difference in stone count) is a weak signal for winning in Go since controlling territory matters far more than raw stone count. So while the greedy agent is better than random, its advantage is modest because the heuristic doesn't strongly correlate with actually winning the game.

# Task 1.2

Playing against the greedy agent, it was immediately clear that it plays poorly. It frequently made moves that seemed obviously suboptimal, such as placing stones in positions that gained no territory and left itself vulnerable to capture. This is not surprising to me given that the agent relies solely on a simple stone-count heuristic. Since this heuristic doesn't capture territory control or other key aspects of Go strategy, the agent has no real sense of which positions are strong or how to build toward a winning board state.

# Task 1.3 

After shuffling the actions, the greedy agent's win rate jumped from 56% to 67% against the random agent. The shuffle altered performance because the original greedy agent was deterministic. It always evaluated actions in the same fixed order, meaning ties in heuristic value were always broken the same way. Since the simple stone-count heuristic could assign the same value to many moves the agent was effectively playing the same moves every game rather than exploring the full action space. Shuffling introduces randomness in tie-breaking, allowing the agent to explore a wider variety of moves and find ones that lead to better outcomes more often.

# Task 1.5

For minimax, a search depth of 3 is the best choice. Depths 1, 2, and 3 all stayed within the time constraint, with depth 3 finishing with around 16 seconds remaining, depth 2 with 19 seconds, and depth 1 with 21 seconds. While depth 1 achieved a 100% win rate against random and depth 2 achieved 90%, depth 3 offers a better balance between search quality and time usage, using more of the available time to make more informed decisions.

For alpha-beta pruning, depth 3 is also the best choice. Time remaining stayed consistently between 18 and 22 seconds across all tested depths, meaning the pruning kept move times very low. Depth 2 had an 80% win rate while other depths achieved 100%, suggesting that deeper search leads to more consistent performance. Since alpha-beta prunes branches that cannot affect the final decision, it can search deeper than minimax in the same amount of time, making depth 3 a reliable and safe choice for both agents

# Task 2b

- Iterative Deepening with a simple heuristic outperformed MCTS, winning 60% of the games (6/10) compared to MCTS’s 40%, while also having slightly faster average move times. Both methods performed similarly as Black, but the heuristic approach still held a small edge.

- The figure I created compares the visit counts allocated by the MCTS agent to its single explored action from the starting board state under two search time budgets: 0.1 seconds and 1.0 seconds. MCTS works by repeatedly selecting promising nodes in a search tree, expanding them, simulating random playouts to a terminal state, and backpropagating the result to update visit counts and values up the tree. The visit count for an action reflects how much of the agent's compute budget was spent exploring that line of play. With 0.1 seconds of search, the agent completed roughly 800 simulations, while with 1.0 seconds it completed nearly 19,000, about 24 times more. Since both runs converged on the same single action from the start state, this shows that even with limited search time MCTS identifies the same move, but with far less confidence. The large difference in visit counts shows that more search time allows the agent to test its chosen action more thoroughly through random playouts, making its decision increasingly reliable even if the chosen action stays the same.

# Final Agent Explaination

- My final agent is ValueMCTSAgent, which combines MCTS with a trained value network. The core motivation was that random rollouts are noisy (a randomly played game often looks nothing like how strong players would actually continue from a given position). By replacing rollouts with a neural network that has seen thousands of real game outcomes, the agent can evaluate positions more accurately within the same time budget. Rather than replacing rollouts entirely, I used a hybrid approach where every third MCTS iteration still uses a random rollout while the other two use the value network. This prevents the network from fully dominating when its predictions are uncertain and keeps the agent grounded in real game outcomes.

- To support the value network, I extended the original 11 board features to 21. The original features captured stone counts, liberty counts, and basic board coverage. I added center control, which scores stones based on proximity to the center since central positions tend to have more board influence; stones at risk, which counts how many of your stones are adjacent to opponent stones and therefore under capture threat; edge stones, which counts stones on the board boundary since those positions have fewer liberties and are generally weaker; and a game phase indicator based on the fraction of empty spaces remaining, which helps the network distinguish early, middle, and endgame positions. All 21 features are computed relative to the current player and are board-size agnostic, meaning they work correctly on both 5x5 and 9x9 boards without modification.

- I also trained a policy network intended to guide which actions get explored first during MCTS expansion. Since the agent only expands one child per iteration and is time-limited, the order in which actions are explored matters. I trained the policy network using imitation learning on the provided dataset, keeping only moves made by the winning player under the assumption that winners played better moves on average. I experimented with several configurations: using the full policy ranking to order all expansions, using only the top 3 ranked moves first and shuffling the rest, and using the policy only 15-30% of the time. In every configuration the policy network hurt performance. Win rate dropped from around 60% to as low as 10% when policy-guided expansion was fully enabled. Inspecting the output showed near-uniform action probabilities with a maximum of around 0.08 meaning the network was barely better than random. I increased the network size from 64-32 hidden units to 256-128-64 and added more features, but the output distribution remained unreliable. Therefore policy guidance is fully disabled in the final submission.

- To improve the value network beyond what the provided dataset offered, I generated additional training data through self-play. The idea is to create a feedback loop: train a value network, use it to build a stronger ValueMCTSAgent, have that agent play games against itself, collect the resulting game states and outcomes as new training data, then retrain the network on the combined dataset. A stronger agent generates higher quality games, which in turn produce better training signal. In practice the improvement from self-play was modest because an imperfect agent tends to make similar mistakes repeatedly, which means the self-play games reinforce those mistakes rather than introducing new information. For the 9x9 board I had no provided dataset at all, so self-play was the only training option. The 9x9 value network is therefore weaker than the 5x5 one and the 9x9 agent falls back to plain MCTS when the network predictions are unreliable.

- Overall the value network improved the 5x5 agent's win rate from around 50% with plain MCTS to around 60% with the hybrid approach. The policy network was the most ambitious component of the design and the one that ultimately did not work. If I had more time I would try computing features on a per-action basis — for example, evaluating the board state after each candidate move rather than the current state — which would give the policy network the information it needs to actually distinguish good moves from bad ones.

Hours taken: 20

Collaborators:

Known Bugs:

AI Use Description:




