# Black Box (Germany)

**Submission: `main.py`**. The platform runs it once against the hidden function and releases the flag at a score of 70 or more.

On the public practice function it scores **100 / 100 using 93 of the 150 queries**.
The hidden function is wired differently, so we also tested on 460 random functions of the same family
(random gate pair, weights and signs, random interaction pair, jump size, switching region from 2% to 25% of the cube).
All 460 scored 100 / 100 with the final code. In the last 300 runs no run used more than 98 queries, and every run used fewer than 120.
These variants were made up by us from the task description, so they show the method is robust, not that the hidden function is covered.

## Why one dial at a time fails
The function is a sum of main effects, one product term and one hard step. Turning one dial at a time with the others at the middle
never enters the region where the step is active and sees every dial as a straight line, which hides both the interaction and the jump.

## What `main.py` does
1. **Design (43 queries).** 16 cube corners (a half-fraction), the centre and 26 random interior points.
   Whatever the switching surface is, the half-space it switches on contains the corner that maximises its normal vector,
   so a corner always lands in the active region and the jump cannot hide.
2. **Smooth part.** A robust fit (RANSAC, with a 1e-4 tolerance because the output is noise-free) of
   a constant, five linear terms and one product term, trying all 10 variable pairs, on the interior points only.
   The corners are deliberately left out of this fit: if one gate variable dominates, the active region contains every corner with that variable at 1,
   and on 0/1 values a step looks exactly like a linear term, so the corners alone would blur the jump into a main effect.
3. **Jump.** Points that are off the smooth fit by the same constant are the active ones. That constant is the jump size,
   and any point can now be labelled active or inactive from a single query.
4. **Gate variables.** Bisect between an interior inactive point and an active point to land within 0.003 of the switching surface,
   then nudge each of the five variables up and down. The ones that flip the label are the gate variables, and the flip direction gives the sign of their weight.
5. **Boundary.** For the two gate variables, bisect along the edges and the diagonal of their unit square (corners with known labels)
   and fit a line through the crossing points. The result is normalised so the largest coefficient is 1.
6. **Verification.** Ten random points are queried 0.03 on either side of the fitted boundary, with all other variables random.
   If any disagrees, the code falls back to a general fit (bisect many segments, fit a hyperplane, and keep the sparsest one that explains them).
7. **Safety.** Every query goes through a guard that refuses to spend the 149th query, and any error returns a valid answer instead of crashing.

The reported answer contains the relevant and inert variables, the interaction (the coefficient of the `xi*xj` monomial after expansion),
the discontinuity with the boundary, threshold, `active_when` and jump size, an expression for the model, and predictions for the 300 challenge points.

## Run
```
pip install numpy==2.2.4
python3 evaluation_script.py main.py --eval-type train
```
