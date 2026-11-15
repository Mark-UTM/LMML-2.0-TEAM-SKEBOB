# Twins Separated at Birth (India)

`main.py` reads `test-input.zip` and writes `output.csv` (`image_a,image_b`). It needs `numpy` and `opencv-python`; it runs on all CPU cores, about 1 minute for 4,000 images.

## The catch
Duplicates are resized, recompressed or slightly cropped copies of one photograph. The archive also holds images with the same geometry as another one but with a very different colour grade (grayscale, sunset tint, replaced sky). The reference treats these as distinct photographs, so geometry alone gives a precision of only 0.65.

## Method
1. **Candidates.** Each image gets a 32x24 grayscale thumbnail; the 20 nearest neighbours by cosine similarity form the candidate pairs (97.8% of the true pairs are inside).
2. **Geometry.** SIFT matches (ratio test 0.8) are fitted with a RANSAC similarity transform (4 px). Pairs with fewer than 12 inliers are dropped.
3. **Photometry.** The second image is warped onto the first and compared: mean and 90th percentile colour difference, grayscale and edge correlation, hue/saturation/value difference, contrast difference, residual of the inliers, aspect-ratio mismatch, overlap.
4. **Decision.** A logistic regression over these 27 values (coefficients are stored in `main.py`) keeps pairs with probability above 0.5.

## Result on the training archive
5-fold cross-validated F1 is 0.977; running `main.py` on the training archive gives precision 0.979, recall 0.977, F1 0.978 (the pass mark is 0.88).
For a local check, rename the training archive to `test-input.zip` and run `python main.py`.
