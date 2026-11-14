# The 300-Year Old Tiger Nest Painting (Bhutan)

`main.py` reassembles a picture that was cut into vertical strips and shuffled. It reads the strips from the `input` folder next to it
and writes `reconstructed_order.json` and `reconstructed_image.png` next to itself. It uses only `numpy`, `scipy` and `opencv-python`.
On the demo strips (192 strips of 8 x 1024 px) it recovers the exact order (match 1.00, SSIM 1.00) in under 5 seconds.
`reconstructed_image.png` and `reconstructed_order.json` in this folder are the output for the demo strips.

## Method
1. **Edge cost.** For every ordered pair (strip i on the left, strip j on the right) we measure how well the right edge of i continues into the left edge of j.
   A cheap first pass uses an L1 distance between linearly extrapolated edge columns. For the 10 best candidates of every strip (in both directions)
   a Mahalanobis gradient compatibility is computed: the colour step across the seam is compared with the strip's own edge gradients, in units of that strip's own
   colour variation. This measure is much more reliable than a plain edge difference on smooth regions, and restricting it to candidates makes it 12 times faster.
2. **Normalisation.** Each cost is divided by the geometric mean of the two best alternatives of its row and column. Without this, strips in smooth regions get low costs
   for every neighbour, and minimising the total cost prefers cheap wrong joins there over good joins in textured regions.
3. **Chain.** The cheapest left-to-right chain through all strips is built by greedy edge joining (each strip gets one successor and one predecessor, no cycles),
   then improved by moving blocks of up to 32 strips (and reversed blocks) while the total cost drops, within a time budget of 9 seconds.
4. **Output.** The strips are concatenated in chain order.

## Validation
Besides the demo strips, the same code was cut into 24 test puzzles from four different pictures (strip widths 4 to 16 px, 128 to 300 strips).
The order was exact in 21 of 24, in at most 6.3 seconds. The 3 misses are all on a blurred picture and are confined to the ends of the chain
(one strip rotated from the start to the end, or the first 3 to 5 strips shuffled), where nearly identical flat strips cannot be told apart.

## Limitations
- **Direction.** A chain read right to left is as good as the same chain read left to right, so edge continuity cannot tell the two apart.
  The code uses a weak prior taken from the demo picture: the half of the picture that is darker goes on the left.
  On the demo strips this gives the correct direction; on other pictures it is a guess.
- **Flat borders.** If the first or last strips of the picture are almost featureless, their order among themselves can be wrong.

## Run
```
pip install -r requirements.txt
mkdir input
(copy the strip PNGs into input)
python main.py
```
