"""Predictability bound of a symbol sequence (Song et al., Science 2010 style) checked against processes with known truth.

Max predictability Pi solves S = H(Pi) + (1 - Pi) log2(N - 1) (Fano), where S is the entropy rate. S is estimated from a finite
sequence with the Lempel-Ziv / Kontoyiannis estimator S_hat = n / sum_i (L_i / log2 i), L_i = length of the shortest substring starting at i
that has not appeared before i.
"""
import math

import numpy as np


def fano_pi(S, N):
    """Invert S = H(Pi) + (1 - Pi) log2(N - 1) for Pi in [1/N, 1] (S in bits)."""
    if S >= math.log2(N):
        return 1.0 / N
    lo, hi = 1.0 / N, 1.0 - 1e-12
    f = lambda p: -p * math.log2(p) - (1 - p) * math.log2(1 - p) + (1 - p) * math.log2(N - 1) - S
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def lz_entropy(seq):
    """seq: list of small ints. Returns bits per symbol."""
    s = bytes(seq)
    n = len(s)
    total = 0.0
    for i in range(1, n):
        L = 1
        prefix = s[:i]
        while i + L <= n and prefix.find(s[i:i + L]) != -1:
            L += 1
        total += L / math.log2(i + 1)
    return (n - 1) / total


def markov_entropy_and_best(P):
    """Stationary distribution, entropy rate (bits) and best one-step accuracy (sum pi_i max_j P_ij) of a transition matrix."""
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    pi = pi / pi.sum()
    H = float(-(pi[:, None] * P * np.log2(np.where(P > 0, P, 1))).sum())
    return pi, H, float((pi * P.max(1)).sum())


def sample_markov(P, n, rng, start=0):
    seq, x = [], start
    for _ in range(n):
        seq.append(x)
        x = int(rng.choice(len(P), p=P[x]))
    return seq


def routine_matrix(N, rng, stay=0.5, concentration=0.3):
    P = rng.dirichlet(np.full(N, concentration), size=N)
    P = (1 - stay) * P
    P[np.arange(N), np.arange(N)] += stay
    return P / P.sum(1, keepdims=True)
