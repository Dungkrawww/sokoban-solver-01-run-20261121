# Sokoban Solver — make.fun sokoban-solver-01-run-20261121

Push-optimal Sokoban solver in a single `main.ts` (Node 22, no dependencies).

## Idea

The generator builds each level as one hallway (the player's row) plus 1–4
independent rooms, each opening onto the hallway through a single door, and a
box pushed out of a room can never come back — so every box stays in its room
and a level's push-optimal count is the sum of its rooms' optima.

The solver decomposes the level into rooms, solves each room push-optimally
with A* over pushes (admissible min-assignment heuristic, corner / wall-run /
2×2-freeze deadlock pruning), and stitches the rooms together with hallway
walks. It solves every room twice (player must be able to leave / free) and
picks the visit order that minimises total pushes — only the last room may end
with the player trapped inside. Deterministic: no clock, no randomness, only
expansion caps, so two runs write identical bytes.

## Results

- Practice set: **1,000,000 / 1,000,000** (all 5 instances at their push
  optimum P\*) via `grade-local`, L1 regrade byte-identical, ~1 s for the
  whole set (limit 60 s).
- 3 extra sets drawn with the kit's own generator (15 unseen levels): all at
  P\*, 1,000,000 each.

## Run

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
