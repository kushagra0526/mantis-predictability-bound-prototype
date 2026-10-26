import json

import numpy as np

from . import bound

LENGTHS = [300, 1000, 4000]


def routine_world(seed, N=20):
    rng = np.random.default_rng(seed)
    P = bound.routine_matrix(N, rng, stay=rng.uniform(0.2, 0.7))
    pi, H, best = bound.markov_entropy_and_best(P)
    return rng, P, H, best, bound.fano_pi(H, N), None


def weekly_world(seed, N=20, eps=0.2):
    """A weekly schedule (hour-of-week -> place) with noise: predictable if the clock is known, which the bare sequence does not say."""
    rng = np.random.default_rng(seed)
    sched = rng.integers(0, N, 168)
    H = -(1 - eps + eps / N) * np.log2(1 - eps + eps / N) - (N - 1) * (eps / N) * np.log2(eps / N)
    best = 1 - eps + eps / N
    return rng, sched, float(H), float(best), bound.fano_pi(float(H), N), eps


def draw_weekly(rng, sched, eps, n, N=20):
    out = []
    for t in range(n):
        out.append(int(sched[t % 168]) if rng.random() > eps else int(rng.integers(0, N)))
    return out


def fill_gaps(seq, rng, share):
    """Drop a share of positions (no signal) and fill them with the most frequent place, as a careless pipeline might."""
    seq = list(seq)
    top = max(set(seq), key=seq.count)
    keep = rng.random(len(seq)) > share
    return [x if k else top for x, k in zip(seq, keep)]


def score(seeds, held):
    rows = {}
    for s in seeds:
        for name, world in (("routine (first-order Markov)", routine_world), ("weekly schedule + noise", weekly_world)):
            if held and name.startswith("routine"):
                pass
            rng, obj, H, best, true_bound, eps = world(s)
            for n in LENGTHS:
                seq = bound.sample_markov(obj, n, rng) if eps is None else draw_weekly(rng, obj, eps, n)
                est = bound.fano_pi(bound.lz_entropy(seq), 20)
                r = rows.setdefault((name, n), {"true_bound": [], "best_accuracy": [], "lz_bound": []})
                r["true_bound"].append(true_bound)
                r["best_accuracy"].append(best)
                r["lz_bound"].append(est)
                if held:
                    g = rows.setdefault((name + ", 40% gaps filled with the top place", n), {"true_bound": [], "best_accuracy": [], "lz_bound": []})
                    g["true_bound"].append(true_bound)
                    g["best_accuracy"].append(best)
                    g["lz_bound"].append(bound.fano_pi(bound.lz_entropy(fill_gaps(seq, rng, 0.4)), 20))
    return {f"{k[0]} | n={k[1]}": {m: round(float(np.mean(v)), 3) for m, v in d.items()} for k, d in sorted(rows.items())}


def main():
    res = {"dev": score(range(0, 12), False), "held_out": score(range(1000, 1012), True)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for m, d in v.items():
            print("  ", m, d)


if __name__ == "__main__":
    main()
