# predbound: how far can a "predictability bound" be trusted?

Mantis builds "living, predictive models of individual people" and its page leads with a published figure: "a 93% upper bound on the predictability of an individual's location, nearly invariant across millions of people" (Song et al., Science 2010; https://mantisbiotech.com/). That kind of bound is computed from a person's sequence with an entropy estimator and an inequality (Fano). It says how well *any* predictor could do, if the estimate is right. This prototype checks the machinery on processes whose true entropy and best accuracy are known.

## What it does
`predbound/bound.py`: the Fano inversion S = H(Pi) + (1 - Pi) log2(N - 1), the Lempel-Ziv / Kontoyiannis entropy estimator, exact entropy rate and best one-step accuracy for a Markov chain. `predbound/run.py`: two worlds with N = 20 places:
- a **first-order Markov routine** (random transition matrix with a "stay" probability), where the true entropy rate, the true Fano bound and the true best accuracy (sum pi_i max_j P_ij) are exact;
- a **weekly schedule with noise** (each hour of the week has a usual place; 20% of hours are a random place), where a predictor that knows the clock gets 81% but the bare sequence does not say what the clock is.
Sequences of 300, 1,000 and 4,000 steps; the LZ estimate of the bound is compared with the truth. In held-out, 40% of positions are also dropped and filled with the most frequent place, as a careless pipeline might.

## Results (mean over 12 worlds per split; Pi values, 1.0 = perfectly predictable)
Dev: seeds 0-11. Held-out: seeds 1000-1011, with the gap-filling variant (fixed in git before the first run).

| Held-out | n | true bound (exact entropy) | best achievable accuracy | LZ-estimated bound |
|---|---|---|---|---|
| routine (Markov) | 300 / 1,000 / 4,000 | 0.579 | 0.370 | 0.662 / 0.647 / 0.640 |
| routine, 40% gaps filled | 300 / 1,000 / 4,000 | 0.579 | 0.370 | 0.710 / 0.694 / 0.679 |
| weekly schedule + noise | 300 / 1,000 / 4,000 | 0.810 | 0.810 | 0.628 / 0.695 / 0.725 |
| weekly schedule, 40% gaps filled | 300 / 1,000 / 4,000 | 0.810 | 0.810 | 0.688 / 0.671 / 0.668 |

Dev: routine true bound 0.664, LZ 0.710 / 0.704 / 0.698; schedule 0.81 true, LZ 0.621 / 0.703 / 0.722.
- The estimate is an *over*estimate of the bound for a Markov routine at every length (+8 points at n = 300, +6 at n = 4,000): short sequences look more predictable than they are.
- The bound is not the accuracy: for the routine the best first-order predictor gets 0.37 while the bound says 0.58, so "93% predictable" does not mean a model gets 93%.
- Without the clock the estimator *under*estimates a schedule-driven life (0.63-0.73 vs 0.81): a predictor that knows the time of week can beat the bound computed from the sequence alone.
- Filling missing data with the most frequent place inflates a Markov routine's estimate (0.64 to 0.68-0.71) and distorts the schedule one in the other direction.

## Read this before believing the table
Synthetic chains and a plain LZ estimator (not the exact variants in every paper); real mobility is not Markov or a noisy schedule, and wearable physiology is not a symbol sequence. Nothing here is Mantis's method or uses its data; the figure in the page is quoted from the literature, not tested.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m predbound.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Mantis.
