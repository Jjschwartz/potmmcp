# Experiments

## Running experiments

The experiment for each environment can by run by running the environments run script. For example to run the `driving` env experiments run the ``run_driving_exp.py`` script with python. You can see the available command line options (e.g. num episodes, num seeds, etc) using the following command:

```
python run_driving_exp.py --help

```

## Environment sizes

### Driving

- Grid size = 14 x 14
- Num free cells = (14 x 14) - 96 = 100
- Num agents = 2
- Num goal locations = 4

**States**

```
= (pos * goal)*(pos-1 * goal-1)*(direction * speed * crashed/dest_reached/none)**2
= (100*4)*(99*3)*(4*4*3)**2
= 273,715,200
~ 2.7*10**8
```

**Actions** = 5 (Do nothing, Acc, Dec, left, right)

**Observations**

Each car observes 14 adjacent cells which can include one other vehicle, one destination

```
= pos_vehicle * pos_dest * (wall/empty)^14 cells
= 15 * 15 * 2^14
= 3,686,400
~ 3.6*10**6
```

### PE (Evader) and PE (Pursuer)

- Grid size = 16 x 16
- Num free coods = 145
- Num goal = 3

**States**

```
= evader_states * pursuer_states
= (pos * dir * goal * min_goal_dist) * (pos * dir)
= (145 * 4 * 3 * 23) * (145 * 4)
= 23,211,600
```

**Actions** = 4 (Forward, Backward, Left, Right)

**Observations**

```
= adj_cells * seed * heard * evader_start_coord, pursuer_start_coord * goal_coord
= 2**4 * 2 * 2 * 6 * 2 * 6
= 4,608
```

Coord observations are constant throught an episode so only influence branching for the initial observation.
Given agents know their initial location and the map is fixed and known the branching is `2*2=4`.


### PP (two-agents)

**States**

```
= pred1_pos * pred2_pos * prey1_pos * prey2_pos * prey3_pos * caught^num_prey
= (100*99*98*97*96) * 2^3
= 7.2×10^10
```

Note, this is a slight overestimate since since agents cannot be in the same pos at the same time.
But you get the idea.

**Actions** = 5 (do_nothing, up, down,  left, right)

**Observations**

= 2 cells in each direction = 5x5 grid

Each cell (except the middle one) can either have the other predator, one of the three prey, be empty, or be a wall/out of bounds.

Empty/wall configs = 4 per corner + 2 per side + 1 center = 4*4 + 2*4 + 1 = 25

```
= other_pred * prey1 * prey2 * prey3 * empty/wall configs
~ 25 * 24 * 23 * 22 * 25
= 7,590,000
= 7*10**6
```

### PP (four-agents)

**States**

```
= pred1_pos * pred2_pos * pred3_pos * pred4_pos * prey1_pos * prey2_pos * prey3_pos * caught^num_prey
= (100*99*98*97*96*95*94) * 2^3
= 6.45×10^14
```

Note, this is a slight overestimate since since agents cannot be in the same pos at the same time.
But you get the idea.

**Actions** = 5 (do_nothing, up, down, left, right)

**Observations**

= 2 cells in each direction = 5x5 grid

Each cell (except the middle one) can either have the other predator, one of the three prey, be empty, or be a wall/out of bounds.

Empty/wall configs = 4 per corner + 2 per side + 1 center = 4*4 + 2*4 + 1 = 25

```
= other_pred1 * other_pred2 * other_pred3 * prey1 * prey2 * prey3 * empty/wall configs
~ 25 * 24 * 23 * 22 * 21 * 20 * 19 * 25
= 60,568,200,000
= 6*10**10
```
