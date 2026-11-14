# Pipes, Leaks, and the Graph Beneath (Finland)

**Flag: `SIGMOID_HOTLINE`**

## How it was found
1. Clean the town names so every town is counted once:
   - strip whitespace and trailing dots, casefold, drop diacritics (`Jyvaskyla` == `Jyväskylä`);
   - map the Swedish names of bilingual towns to the Finnish name (`Helsingfors` -> `Helsinki`, `Åbo` -> `Turku`, ...).
2. Build an undirected graph with one edge per unordered town pair. Route type and repeated logs are ignored, and there are no self-loops.
   This matches the self-check from the task: 116 towns, 448 connections, and the top-8 degrees are all distinct.
3. The flag is the first letters of the 7 highest-degree towns: **HOTLINE**.
   A raw group-by on the uncleaned `from_town` gives a different, wrong leader.

## Run
Put `routes.csv` from the task next to `solve.py`.
```
pip install pandas
python finland/solve.py
```
