"""Quantum amplitude estimation of flood-exceedance risk, written directly in Qiskit.

Question answered: "what is the probability that the threat driver (72-hour rainfall or peak river
discharge) exceeds the flood threshold?", given a forecast ensemble.

Amplitude estimation in plain words: load the forecast distribution into a few qubits, flag the
outcomes above the threshold on one extra "objective" qubit, and then, instead of just measuring
many times (classical Monte Carlo), apply a Grover-style operator Q that rotates the state so the
probability of reading the flag becomes sin^2((2k+1) theta). Measuring at several k and fitting
theta gives the exceedance probability a = sin^2(theta) with fewer uses of the loading circuit than
plain sampling would need for the same precision. We use the maximum-likelihood variant
(Suzuki et al., Quantum Inf. Process. 19, 75, 2020), which needs no quantum phase estimation.

Honest caveat: the distribution here is an 8-bin histogram of at most ~50 ensemble members, so the
exact answer is trivial to compute classically. The quantum part is real and verified against that
exact answer, but offers no practical advantage at this size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, XGate
from qiskit_aer import AerSimulator

N_STATE_QUBITS = 3          # 8 bins. The threshold sits on a bin edge, so the exceedance probability is exact.
_COMPILED: dict = {}


@dataclass
class Discretised:
    probs: np.ndarray        # (2^n,) bin probabilities
    width: float             # bin width in driver units
    k_threshold: int         # first bin index counted as "exceeds"
    threshold: float
    exact: float             # exact exceedance probability of the discretised distribution

    @property
    def edges(self) -> np.ndarray:
        return np.arange(len(self.probs) + 1) * self.width


def discretise(samples, threshold: float, n_qubits: int = N_STATE_QUBITS) -> Discretised:
    """Histogram ensemble members on 2^n equal bins starting at 0, with the threshold exactly on a bin edge.
    Members beyond the last edge go into the top bin."""
    x = np.asarray(samples, dtype=float)
    nb = 2 ** n_qubits
    top = max(float(x.max()) * 1.05, threshold * nb / (nb - 1))
    m = int(np.clip(np.floor(nb * threshold / top), 1, nb - 1))   # threshold sits on edge m
    width = threshold / m
    idx = np.clip(np.floor(x / width + 1e-9).astype(int), 0, nb - 1)
    probs = np.bincount(idx, minlength=nb) / len(x)
    return Discretised(probs=probs, width=width, k_threshold=m, threshold=threshold, exact=float(probs[m:].sum()))


# --------------------------------------------------------------------------- circuits
def mark_geq(n: int, k: int) -> QuantumCircuit:
    """Flip the objective qubit (index n) when the n-qubit register value i satisfies i >= k.

    Uses at most n + 1 multi-controlled X gates on disjoint bit patterns and no ancilla qubits:
    i >= k  iff  i == k, or for some bit j where k has a 0, i agrees with k on all higher bits and has a 1 at j.
    """
    qc = QuantumCircuit(n + 1, name=f">={k}")
    if k <= 0:
        qc.x(n)
        return qc
    if k >= 2 ** n:
        return qc
    bits = [(k >> j) & 1 for j in range(n)]

    def add(controls: list[int], values: list[int]):
        state = sum(v << pos for pos, v in enumerate(values))
        qc.append(XGate().control(len(controls), ctrl_state=state), controls + [n])

    for j in range(n - 1, -1, -1):
        if bits[j] == 0:
            ctrl = list(range(j, n))
            add(ctrl, [1] + bits[j + 1:])
    add(list(range(n)), bits)                                   # the i == k case
    return qc


def a_operator(d: Discretised) -> QuantumCircuit:
    """A|0> = sum_i sqrt(p_i)|i> (|0> or |1> on the objective qubit, 1 when bin i exceeds the threshold)."""
    n = int(np.log2(len(d.probs)))
    qc = QuantumCircuit(n + 1, name="A")
    qc.append(StatePreparation(np.sqrt(d.probs)), range(n))
    qc.compose(mark_geq(n, d.k_threshold), inplace=True)
    return qc


def grover_operator(a: QuantumCircuit) -> QuantumCircuit:
    """Q = A S_0 A^dagger S_chi  (S_chi flips the sign of flagged states, S_0 of |0...0>)."""
    n_tot = a.num_qubits
    qc = QuantumCircuit(n_tot, name="Q")
    qc.z(n_tot - 1)                                            # S_chi
    qc.compose(a.inverse(), inplace=True)                      # A^dagger
    qc.x(range(n_tot))                                         # S_0
    qc.h(n_tot - 1)
    qc.mcx(list(range(n_tot - 1)), n_tot - 1)
    qc.h(n_tot - 1)
    qc.x(range(n_tot))
    qc.compose(a, inplace=True)                                # A
    return qc


def qae_circuit(d: Discretised, k: int, measure: bool = True) -> QuantumCircuit:
    a = a_operator(d)
    q = grover_operator(a)
    qc = QuantumCircuit(a.num_qubits, 1) if measure else QuantumCircuit(a.num_qubits)
    qc.compose(a, inplace=True)
    for _ in range(k):
        qc.compose(q, inplace=True)
    if measure:
        qc.measure(a.num_qubits - 1, 0)
    return qc


# --------------------------------------------------------------------------- estimation
def _mle(powers, hits, shots) -> float:
    theta = np.linspace(1e-6, np.pi / 2 - 1e-6, 40001)
    ll = np.zeros_like(theta)
    for k, h in zip(powers, hits):
        pk = np.clip(np.sin((2 * k + 1) * theta) ** 2, 1e-12, 1 - 1e-12)
        ll += h * np.log(pk) + (shots - h) * np.log(1 - pk)
    return float(np.sin(theta[int(np.argmax(ll))]) ** 2)


def estimate(d: Discretised, powers=(0, 1, 2, 4, 8), shots: int = 64, seed: int = 1234, noise=None) -> dict:
    """Maximum-likelihood amplitude estimation on the Aer simulator (shot-based)."""
    sim = AerSimulator(noise_model=noise, seed_simulator=seed)
    hits, depth, cx = [], 0, 0
    for k in powers:
        key = (d.probs.tobytes(), d.k_threshold, k)
        if key not in _COMPILED:                                 # transpile each circuit once
            _COMPILED[key] = transpile(qae_circuit(d, k), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                                       seed_transpiler=1234)
        tc = _COMPILED[key]
        depth, cx = max(depth, tc.depth()), max(cx, int(tc.count_ops().get("cx", 0)))
        counts = sim.run(tc, shots=shots).result().get_counts()
        hits.append(int(counts.get("1", 0)))
    a_hat = _mle(powers, hits, shots)
    queries = int(shots * sum(2 * k + 1 for k in powers))                      # uses of A (or its inverse)
    fisher = shots * sum((2 * k + 1) ** 2 for k in powers) / max(a_hat * (1 - a_hat), 1e-9)
    return {
        "estimate": a_hat,
        "exact": d.exact,
        "abs_error": abs(a_hat - d.exact),
        "std_error_ideal": float(1 / np.sqrt(fisher)),
        "queries": queries,
        "classical_mc_std_error_same_queries": float(np.sqrt(max(d.exact * (1 - d.exact), 1e-9) / queries)),
        "powers": list(powers),
        "shots_per_circuit": shots,
        "hits": hits,
        "qubits": int(np.log2(len(d.probs))) + 1,
        "max_depth": int(depth),
        "max_cx": int(cx),
    }


def classical_monte_carlo(d: Discretised, n_samples: int, seed: int = 1234) -> float:
    """Baseline: draw n_samples bins from the same distribution and count exceedances."""
    rng = np.random.default_rng(seed)
    return float((rng.choice(len(d.probs), size=n_samples, p=d.probs) >= d.k_threshold).mean())
