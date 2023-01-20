# BA-POSGMCP experiments to UAI paper

## Driving

Each experiment run against every fixed KLR policy `[klr_k0, klr_k1, klr_k2, klr_k3]` with the following parameters:

- `num_sims = [10, 50, 100, 500, 1000]`
- `num_episodes = 100`

Experiment list:

- [ ] BAPOSGMCP
  - [ ] Greedy MetaPi, Truncated
  - [ ] Softmax MetaPi, Truncated
  - [ ] Uniform MetaPi, Truncated
  - [ ] Best MetaPi, Not Truncated
  - [ ] Fixed Pis (5), Truncated
  - [ ] Random Pi, Not Truncated
- [ ] UCB MCP
    - [ ] Best MetaPi, Truncated
	- [ ] Best MetaPi, Not Truncated
	- [ ] Random Pi, Not Truncated
- [ ] Meta Pis
    - [ ] Greedy
	- [ ] Softmax
	- [ ] Uniform


## LevelBasedForaging

Each experiment run against every fixed heuristic policy `[heuristic1, heuristic2, heuristic2, heuristic3]` with the following parameters:

- `num_sims = [10, 50, 100, 500, 1000]`
- `num_episodes = 100`

Experiment list:

- [ ] BAPOSGMCP
  - [ ] Greedy MetaPi, Not Truncated
  - [ ] Softmax MetaPi, Not Truncated
  - [ ] Uniform MetaPi, Not Truncated
  - [ ] Random Pi, Not Truncated
- [ ] UCB MCP
    - [ ] Best MetaPi, Not Truncated
	- [ ] Random Pi, Not Truncated
- [ ] Meta Pis
    - [ ] Greedy
	- [ ] Softmax
	- [ ] Uniform


## PredatorPrey-P2

Each experiment run against every fixed policy `[sp_seed0, sp_seed1, sp_seed2, sp_seed3, sp_seed4]` with the following parameters:

- `num_sims = [10, 50, 100, 500, 1000]`
- `num_episodes = 100`

Experiment list:

- [ ] BAPOSGMCP
  - [ ] Greedy MetaPi, Truncated
  - [ ] Softmax MetaPi, Truncated
  - [ ] Uniform MetaPi, Truncated
  - [ ] Random Pi, Not Truncated
- [ ] UCB MCP
    - [ ] Best MetaPi, Truncated
	- [ ] Random Pi, Not Truncated
- [ ] Meta Pis
    - [ ] Greedy
	- [ ] Softmax
	- [ ] Uniform


## PredatorPrey-P4

Each experiment run against every fixed policy `[sp_seed0, sp_seed1, sp_seed2, sp_seed3, sp_seed4]` with the following parameters:

- `num_sims = [10, 50, 100, 500, 1000]`
- `num_episodes = 100`

Experiment list:

- [ ] BAPOSGMCP
  - [ ] Greedy MetaPi, Truncated
  - [ ] Softmax MetaPi, Truncated
  - [ ] Uniform MetaPi, Truncated
  - [ ] Random Pi, Not Truncated
- [ ] UCB MCP
    - [ ] Best MetaPi, Truncated
	- [ ] Random Pi, Not Truncated
- [ ] Meta Pis
    - [ ] Greedy
	- [ ] Softmax
	- [ ] Uniform


## PursuitEvasion

Each experiment run against every fixed policy `[sp_seed0, sp_seed1, sp_seed2, sp_seed3, sp_seed4]` with the following parameters:

- `num_sims = [10, 50, 100, 500, 1000]`
- `num_episodes = 100`

Experiment list **Evader (agent 0)**:

- [ ] BAPOSGMCP
  - [ ] Greedy MetaPi, Truncated
  - [ ] Softmax MetaPi, Truncated
  - [ ] Uniform MetaPi, Truncated
  - [ ] Random Pi, Not Truncated
- [ ] UCB MCP
    - [ ] Best MetaPi, Truncated
	- [ ] Random Pi, Not Truncated
- [ ] Meta Pis
    - [ ] Greedy
	- [ ] Softmax
	- [ ] Uniform


Experiment list **Pursuer (agent 1)**:

- [ ] BAPOSGMCP
  - [ ] Greedy MetaPi, Truncated
  - [ ] Softmax MetaPi, Truncated
  - [ ] Uniform MetaPi, Truncated
  - [ ] Random Pi, Not Truncated
- [ ] UCB MCP
    - [ ] Best MetaPi, Truncated
	- [ ] Random Pi, Not Truncated
- [ ] Meta Pis
    - [ ] Greedy
	- [ ] Softmax
	- [ ] Uniform
