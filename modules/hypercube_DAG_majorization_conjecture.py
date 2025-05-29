"""
Code to accompany "On the Probability of Existence of Pairwise Stable Networks in Large Random Network Formation Settings"

This module has two parts.  Part 1 performs generates a superset of admissable degree sequences of a DAG on the n-hypercube and then confirms that no member fails the conjectured majorization condition.
Part 2 genrates random DAGs on the n-hypercube and confirms that they satisfy the conjectured majorization condition.
"""


#    _                     __
#   (_)_ _  ___  ___  ____/ /____
#  / /  ' \/ _ \/ _ \/ __/ __(_-<
# /_/_/_/_/ .__/\___/_/  \__/___/
#        /_/
# ________________________________________________________________________________________________________________________________________

import numpy as np
import time
from sympy import binomial
import enlighten
import os
from itertools import product, chain, repeat, combinations
from inspect import signature
from functools import lru_cache
from collections import Counter
from itertools import product
import random
import math
from typing import List, Tuple

# __________
from math import e
import os


try:
    from graph_tool.all import *

    np.float128 = float
except:
    pass

import warnings


import os

import importlib, sys


os.chdir("../conjecture_functions/")


# This formtran module must be installed via f2py
# This module checks the minimum number of sinks condition for a hypercube in-degree sequence,
# and generates all sequences that satisfy the condition.

# Sinks-to-faces inequality ― holds for every integer t ―────────────────────
#
#           ⎛ d(v) ⎞
#   Σ_{v∈V} ⎜      ⎟  ≥  f_t        (sinks-to-faces)
#           ⎝  t   ⎠
#
# Sources-to-faces inequality ― likewise, for every integer t ―──────────────
#
#           ⎛ n − d(v) ⎞
#   Σ_{v∈V} ⎜          ⎟ ≥  f_t     (sources-to-faces)
#           ⎝    t     ⎠
#
# Here “⎛ x ⎞ / ⎝ t ⎠” denotes the binomial coefficient “x choose t”.

from diophantine_omp import diophantine_mod as diophantine_mod_omp


#    ___           __    ___
#   / _ \___ _____/ /_  <  /
#  / ___/ _ `/ __/ __/  / /
# /_/   \_,_/_/  \__/  /_/
# ________________________________________________________________________________________________________________________________________


def is_mean_preserving_spread(candidate, base, tol=1e-8):
    """
    Check whether 'candidate' is a mean-preserving spread (MPS) of 'base', i.e. whether base majorizes candidate.

    For two collections of numbers (which can be either lists or NumPy arrays),
    candidate is an MPS of base if:
      1. They have (approximately) the same mean.
      2. When sorted in descending order, the partial sums of base are less than or
         equal to the corresponding partial sums of candidate (with strict inequality
         for at least one partial sum, although this version only checks non-violation
         of the condition).

    This is equivalent to saying that base is majorized by candidate.

    Parameters:
        base (list or np.ndarray): The original set of values.
        candidate (list or np.ndarray): The set to test whether it is an MPS of base.
        tol (float): Tolerance for floating-point comparisons.

    Returns:
        bool: True if candidate is a mean-preserving spread of base, False otherwise.

    Raises:
        ValueError: If the two inputs do not have the same number of elements.
    """
    # Convert lists to numpy arrays, if needed.
    base_arr = np.asarray(base, dtype=float)
    cand_arr = np.asarray(candidate, dtype=float)

    if base_arr.shape != cand_arr.shape:
        raise ValueError("Both inputs must have the same shape or number of elements.")

    n = base_arr.size  # works for 1-d arrays
    mean_base = np.mean(base_arr)
    mean_candidate = np.mean(cand_arr)

    # Check that means are approximately equal.
    if abs(mean_base - mean_candidate) > tol:
        return False

    # Sort both arrays in descending order.
    base_sorted = np.sort(base_arr)[::-1]
    cand_sorted = np.sort(cand_arr)[::-1]

    # Compute cumulative sums.
    cum_base = np.cumsum(base_sorted)
    cum_cand = np.cumsum(cand_sorted)

    # For indices 0 to n-2, check that cumulative sum of base is <= that of candidate.
    # (The last cumulative sum must be equal since the means are equal.)
    if np.any(cum_cand[:-1] > cum_base[:-1] + tol):
        return False

    return True


# ---------- 3. helper to ‘expand’ a solution ----------
def expand_counts(counts):
    """
    (n_{m‑1}, …, n₀)  ⟼  (m‑1 repeated n_{m‑1} times, …, 1 repeated n₁ times, 0 repeated n₀ times).

    Works for any length ``counts``.
    """
    m = len(counts)
    return tuple(
        chain.from_iterable(
            repeat(val, cnt)  # repeat the value …
            for val, cnt in zip(
                range(m - 1, -1, -1), counts
            )  # … that matches each count
        )
    )


def check_Harper_constraints(indeg):
    """
    Return True iff the in‑degree multiset `indeg` could belong to an
    *acyclic* orientation of the n‑dimensional hypercube Q_n.

    Necessary conditions tested
    ---------------------------
    (A) 0 ≤ dᵢ ≤ n  for every vertex
    (B) Σ dᵢ = n·2ⁿ⁻¹   (total edge count of Q_n)
    (C) For every k,   Σ_{i≤k} d_{(i)} ≤ M_n(k),
        where d_{(i)} are the in‑degrees in non‑increasing order and
        M_n(k) is the maximum number of edges inside *any* k‑vertex
        induced sub‑graph of Q_n (Harper).
    """
    m = len(indeg)
    # ---------- deduce n --------------------------------------------------
    n = m.bit_length() - 1
    if m != 1 << n:
        raise ValueError("length of indegree tuple must be a power of two")

    d = tuple(indeg)
    # ---------- condition (A) --------------------------------------------
    if any(x < 0 or x > n for x in d):
        return False

    # ---------- condition (B) --------------------------------------------
    if sum(d) != n * (1 << (n - 1)):
        return False

    # ---------- compute M_n(k) via Harper’s ordering ---------------------
    #   The “initial segment in binary order” attains the maximum.
    #   So list the vertices 0…2ⁿ−1 and add edges greedily.
    #
    #   For speed, pre‑compute, for each vertex, the list of earlier
    #   neighbours in binary order.
    M = [0] * (m + 1)  # M[k] will hold the bound for size k
    T = [0] * (m + 1)
    internal_edges = 0
    in_degs_sum = 0

    in_degs = [0] * m  # in_degs[v] = weight of vertex v
    earlier = [0] * m  # earlier[v] = number of neighbours < v

    for v in range(m):
        # a neighbour differs in exactly one bit ⇒ v ^ (1<<i)
        earlier[v] = sum(1 for i in range(n) if (v ^ (1 << i)) < v)
        in_degs[v] = n - (v).bit_count()

    for k in range(1, m + 1):
        internal_edges += earlier[k - 1]
        M[k] = internal_edges

        in_degs_sum += in_degs[k - 1]
        T[k] = in_degs_sum

    M = M[1:]
    T = T[1:]
    # ---------- prefix‑sum test (C) --------------------------------------
    # print(M)
    # print(T)

    return bool(np.all(np.cumsum(sorted(d)) <= M)) & bool(
        np.all(np.cumsum(sorted(d)[::-1]) >= T)
    )


def acyclic_orientation_exists(deg):
    """
    Performs a depth-first search (DFS) to determine whether there exists
    a directed acyclic graph (DAG) orientation of the n-dimensional hypercube
    Q_n with the specified in-degrees.

    Parameters
    ----------
    deg : tuple[int]
        Target in‑degree of every vertex, indexed by its binary label.
        Its length must be a power of two, say 2ⁿ.

    Returns
    -------
    bool
        True  ⇔  there exists a DAG orientation of Q_n with those in‑degrees.
        False ⇔  impossible (all topological orders fail).
    """
    m = len(deg)
    if m == 0 or m & (m - 1):
        raise ValueError("len(deg) must be 2**n for some n ≥ 1")

    n = m.bit_length() - 1  # dimension

    # quick sanity checks: degree bounds and total edges
    if any(d < 0 or d > n for d in deg):
        return False
    if sum(deg) != n * (m // 2):
        return False

    # histogram of remaining counts; index = desired in‑degree
    remaining = [0] * (n + 1)
    for d in deg:
        remaining[d] += 1
    remaining = tuple(remaining)  # make hashable

    # ---------- adjacency lists of Q_n -----------------------------------
    adj = [[] for _ in range(m)]
    for v in range(m):
        for i in range(n):
            adj[v].append(v ^ (1 << i))

    # ---------- DFS with memoised states ---------------------------------
    @lru_cache(maxsize=None)
    def dfs(mask, rem):  # rem is a tuple like remaining
        """
        mask : bitset of already placed vertices in the topological order
        rem  : tuple of remaining multiplicities for in‑degrees 0 … n
        """
        if mask == (1 << m) - 1:  # all 2ⁿ vertices placed
            return all(x == 0 for x in rem)

        # heuristic: try vertices that create the *fewest* branches first
        for v in range(m):
            if mask >> v & 1:  # already placed
                continue
            indeg = sum((mask >> nb) & 1 for nb in adj[v])
            if indeg <= n and rem[indeg]:  # still “slots” for this indeg?
                new_mask = mask | (1 << v)
                new_rem = list(rem)
                new_rem[indeg] -= 1
                if dfs(new_mask, tuple(new_rem)):
                    return True
        return False

    return dfs(0, remaining)


def repeated_binomial_sequence(n: int) -> list[int]:
    """
    Return a list in which each integer k (0 ≤ k ≤ n) appears
    C(n, k) = “n choose k” times.

    Examples
    --------
    >>> repeated_binomial_sequence(2)
    [0, 1, 1, 2]

    >>> repeated_binomial_sequence(3)
    [0, 1, 1, 1, 2, 2, 3]
    """
    seq = []
    for k in range(n + 1):
        seq.extend(repeat(k, math.comb(n, k)))  # repeat k, C(n,k) times
    return seq


if __name__ == "__main__":
    # Generate all in-degree sequences satisyfing the minimum number of sinks condition for a hypercube DAG
    # Note that depending on your system, this may take hours to days for n > 5.
    n = 4
    sols, nsols = diophantine_mod_omp.diophantine_solutions(n, max_sols=n * 10**6)
    sols = sols[:nsols]

    # pickle_object(sols, "diophantine_solutions_n_6.pkl")

    sols_tup = [tuple(x) for x in sols.tolist()]
    sols_full = [expand_counts(x) for x in sols_tup]

    valid = [a for a in sols_full if check_Harper_constraints(a)]

    canonical_sequence = repeated_binomial_sequence(n)

    valid_not_maj = [
        a for a in valid if not is_mean_preserving_spread(canonical_sequence, a)
    ]

    valid_not_maj_possible = [a for a in valid_not_maj if acyclic_orientation_exists(a)]


#    ___           __    ___
#   / _ \___ _____/ /_  |_  |
#  / ___/ _ `/ __/ __/ / __/
# /_/   \_,_/_/  \__/ /____/
# # ________________________________________________________________________________________________________________________________________


def random_hypercube_dag(
    d: int, *, seed: int | None = None
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Generate a random DAG on the d-dimensional hyper-cube.

    Parameters
    ----------
    d     : int
        Dimension of the hyper-cube (number of bits).  The graph has
        N = 2**d vertices, labelled 0 … N-1 in binary.
    seed  : int, optional
        Seed for reproducibility.

    Returns
    -------
    order : list[int]
        The random permutation used as a topological order.
    dag   : (N,N) np.ndarray
        Adjacency matrix (0/1, int) of the directed acyclic graph.
        dag[u,v] == 1  ⇔  edge u → v.
    undirected : (N,N) np.ndarray
        The symmetric 0/1 adjacency matrix of the undirected hyper-cube.
        Returned in case you also want it; delete this output if not
        needed.
    """
    if d < 0 or d > 20:  # memory guard (2**20 = 1,048,576 vertices)
        raise ValueError("d must be between 0 and 20 (inclusive)")

    N = 1 << d  # number of vertices: 2**d
    if seed is not None:
        random.seed(seed)

    # --- Build undirected adjacency matrix ---------------------------------
    undirected = np.zeros((N, N), dtype=int)

    # Each vertex v is adjacent to v ^ (1<<i) for every bit position i.
    for v in range(N):
        for i in range(d):
            u = v ^ (1 << i)
            undirected[v, u] = 1
            undirected[u, v] = 1  # ensure symmetry

    # --- Random topological order ------------------------------------------
    order = list(range(N))
    random.shuffle(order)
    rank = {v: i for i, v in enumerate(order)}

    # --- Orient edges -------------------------------------------------------
    dag = np.zeros_like(undirected)
    rows, cols = np.triu_indices(N, k=1)
    for u, v in zip(rows, cols):
        if undirected[u, v]:
            if rank[u] < rank[v]:
                dag[u, v] = 1
            else:
                dag[v, u] = 1

    return dag


# Set the parameters for the hypercube DAG majorization conjecture test
n = 9
samples_to_generate = 10**3


if __name__ == "__main__":
    base_sequence = repeated_binomial_sequence(n)

    for _ in range(samples_to_generate):
        dag = random_hypercube_dag(n, seed=_)
        assert is_mean_preserving_spread(base_sequence, dag.sum(axis=0)), (
            f"Sample {_} failed majorization condition."
        )
