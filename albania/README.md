# Looks like an anomaly! (Albania)

**Submission: `detector.py`**. The flag is released by the platform once the score reaches 70/100.
On the training stream our local copy of the scoring formula gives **93.96**
(F1 0.938; recall: point 1.00, drift 0.91, collective 0.95). It runs in about 0.2 s, using the standard library only.

## What the anomalies look like in the training data
The values below are z-scores per zone, against normal readings.
- **point**: one sensor spikes for one minute (vibration, acoustic or CO₂, z ≈ 14–25);
- **drift**: one sensor ramps linearly over about 60 minutes (humidity, CO₂ or pressure, from z ≈ 0 to about 11), then snaps back;
- **collective**: vibration, acoustic and CO₂ are all moderately raised together (z ≈ 4–5 each) for 30–35 minutes.

The two zones alternate in blocks of 900 minutes and have different noise levels.
Temperature and pressure wander slowly (lag-1 autocorrelation ≈ 0.75); the other sensors are white noise.

## Detector
1. **Baseline per zone.** The mean is learned from the first 60 readings of each zone, the variance from the same warm-up,
   and both then track slowly (exponential weight 0.01). They are updated only on readings judged normal, so an anomaly is
   never absorbed into the baseline.
2. **Point:** any sensor with |z| ≥ 7.
3. **Smoothed deviation:** an EWMA (α = 0.25) of the clipped z per sensor, divided by its own normal-time spread.
   This normalization is what keeps the slowly wandering temperature and pressure from raising false alarms.
4. **Drift:** any one of three tests fires:
   - smoothed deviation ≥ 2.1;
   - two-sided CUSUM (k = 0.5, h = 9);
   - linear-trend t-statistic ≥ 5.5 over the last 20 minutes, with the recent level ≥ 1.2 in the same direction.
     This test catches ramps earlier.
5. **Collective:** at least 2 sensors with smoothed deviation ≥ 1.8 at the same time.
6. **Fast end:** if no sensor currently has |z| ≥ 1.8, drift and collective states are reset.
   Drifts end abruptly, and this avoids false alarms right after them.

The score is 1.0 for an anomaly and 0.0 otherwise.

## How the parameters were chosen
A grid search on the training stream alone peaks sharply, which means it overfits. So we also generated 12 synthetic streams
(`synth.py`) from the training data: anomalies are removed and replaced by normal behaviour, then new anomalies of the three
kinds are inserted at random places, on random sensors and with random signs. The chosen parameters maximise the worst case
across these streams.

| | training | synthetic, mean | synthetic, worst |
|---|---|---|---|
| sharp grid optimum | 94.23 | 78.3 | 67.2 |
| **chosen** | **93.96** | **80.8** | **75.9** |

## Run
Put `train-input.csv` and `train-output.csv` from the task next to the scripts.
```
python evaluate.py    # streams the training data through Detector and prints the task's score formula
```
