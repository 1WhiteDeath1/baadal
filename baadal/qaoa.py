"""QAOA solver written directly in Qiskit (no Qiskit Optimization / Qiskit Algorithms dependency).

QAOA in plain words: start from an equal superposition of every possible plan,
then alternate two operations p times:
  * a "cost" step  exp(-i * gamma * H_C)  that phase-shifts each plan by its QUBO energy, and
  * a "mixer" step exp(-i * beta * sum X) that lets amplitude flow between plans that differ by one bit.
A classical optimiser tunes the 2p angles (gamma, beta) so that measuring the circuit
returns low-energy (good, feasible) plans with high probability.

Simulation uses Qiskit Aer (exact statevector for training, shot-based sampling for the
final answer, optional depolarising-noise model). Everything is seeded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from scipy.optimize import minimize

from .qubo import QUBO


def qaoa_circuit(h: np.ndarray, J: np.ndarray, p: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build the depth-p QAOA ansatz for the Ising Hamiltonian H = sum h_i Z_i + sum J_ij Z_i Z_j."""
    n = len(h)
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)
    qc = QuantumCircuit(n, name=f"QAOA(p={p})")
    qc.h(range(n))
    for layer in range(p):
        # Cost layer: RZ(theta) = exp(-i theta Z / 2), RZZ(theta) = exp(-i theta ZZ / 2)
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gammas[layer] * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    qc.rzz(2 * gammas[layer] * J[i, j], i, j)
        # Mixer layer: exp(-i beta X) on every qubit
        for i in range(n):
            qc.rx(2 * betas[layer], i)
    return qc, gammas, betas


def cvar(energies: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """Conditional value at risk: mean energy of the best alpha-fraction of outcomes
    (Barkoutsos et al., Quantum 4, 256, 2020). alpha = 1 gives the ordinary expectation."""
    if alpha >= 1.0:
        return float(energies @ probs)
    order = np.argsort(energies)
    e, pr = energies[order], probs[order]
    cum = np.cumsum(pr)
    k = int(np.searchsorted(cum, alpha))
    mass_before = cum[k - 1] if k > 0 else 0.0
    total = float(e[:k] @ pr[:k]) + float(e[k]) * (alpha - mass_before)
    return total / alpha


@dataclass
class QAOAResult:
    p: int
    params: list[float]
    objective: float
    n_evals: int
    seconds: float
    probs: np.ndarray


class QAOASolver:
    def __init__(self, qubo: QUBO, seed: int = 1234, alpha: float = 0.25):
        self.qubo = qubo
        self.seed = seed
        self.alpha = alpha
        h, J, _ = qubo.to_ising()
        self.scale = float(max(np.abs(h).max(), np.abs(J).max()))
        self.h, self.J = h / self.scale, J / self.scale   # normalised so angles live in a sane range
        self.energies = qubo.energy_all()                # original units, index k <-> Qiskit bit order
        self.sim = AerSimulator(method="statevector", seed_simulator=seed)
        self._cache: dict[int, tuple] = {}

    @property
    def n_qubits(self) -> int:
        return len(self.h)

    def _compiled(self, p: int):
        if p not in self._cache:
            qc, g, b = qaoa_circuit(self.h, self.J, p)
            sv = qc.copy()
            sv.save_statevector()
            tsv = transpile(sv, self.sim, optimization_level=0)
            self._cache[p] = (qc, g, b, tsv)
        return self._cache[p]

    def probabilities(self, params: np.ndarray, p: int) -> np.ndarray:
        _, g, b, tsv = self._compiled(p)
        binding = {g[k]: params[k] for k in range(p)} | {b[k]: params[p + k] for k in range(p)}
        bound = tsv.assign_parameters(binding)
        sv = self.sim.run(bound).result().get_statevector()
        return np.abs(np.asarray(sv)) ** 2

    def objective(self, params: np.ndarray, p: int) -> float:
        return cvar(self.energies, self.probabilities(params, p), self.alpha)

    def optimize(self, p_max: int = 4, grid: tuple[int, int] = (16, 12), maxiter_p1: int = 200,
                 maxiter_per_layer: int = 100) -> list[QAOAResult]:
        """Frozen training recipe (chosen on the calibration instance exceptional / 220 PKR m):

        * p = 1: grid scan over (gamma, beta), then COBYLA from the best grid point.
        * p > 1: two warm starts, keep the better final CVaR:
            - INTERP: interpolate the p-1 angles (Zhou et al., PRX 10, 021067, 2020);
            - zero-pad: previous angles plus a new layer at gamma = beta = 0. This start reproduces
              the p-1 objective exactly, so the optimised CVaR never gets worse as p grows.
        """
        results: list[QAOAResult] = []
        t0 = time.perf_counter()
        evals = 0
        best = None
        for gm in np.linspace(0.05, np.pi, grid[0]):
            for bt in np.linspace(0.05, np.pi / 2, grid[1]):
                val = self.objective(np.array([gm, bt]), 1)
                evals += 1
                if best is None or val < best[0]:
                    best = (val, np.array([gm, bt]))
        for p in range(1, p_max + 1):
            if p == 1:
                starts, maxiter, rhobeg = [best[1]], maxiter_p1, 0.2
            else:
                prev = np.array(results[-1].params)
                g, b = prev[: p - 1], prev[p - 1:]
                starts = [np.concatenate([_interp(g), _interp(b)]),
                          np.concatenate([g, [0.0], b, [0.0]])]
                maxiter, rhobeg = maxiter_per_layer * p, 0.1
            winner = None
            for x0 in starts:
                counter = {"n": 0}

                def f(x, p=p, counter=counter):
                    counter["n"] += 1
                    return self.objective(x, p)

                res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": rhobeg})
                evals += counter["n"]
                if winner is None or res.fun < winner.fun:
                    winner = res
            if results and winner.fun > results[-1].objective:   # numerical safety: keep monotone
                padded = np.concatenate([np.array(results[-1].params[: p - 1]), [0.0],
                                         np.array(results[-1].params[p - 1:]), [0.0]])
                winner.x, winner.fun = padded, results[-1].objective
            results.append(QAOAResult(p=p, params=list(map(float, winner.x)), objective=float(winner.fun),
                                      n_evals=evals, seconds=time.perf_counter() - t0,
                                      probs=self.probabilities(np.asarray(winner.x), p)))
        return results

    # ---- sampling and circuit statistics --------------------------------
    def bound_measured_circuit(self, params, p: int) -> QuantumCircuit:
        qc, g, b, _ = self._compiled(p)
        binding = {g[k]: params[k] for k in range(p)} | {b[k]: params[p + k] for k in range(p)}
        m = qc.assign_parameters(binding)
        m.measure_all()
        return m

    def sample(self, params, p: int, shots: int = 512, noise: NoiseModel | None = None) -> dict[int, int]:
        """Return counts keyed by basis-state index k (bit i of k = qubit i)."""
        circ = self.bound_measured_circuit(params, p)
        sim = AerSimulator(noise_model=noise, seed_simulator=self.seed)
        tc = transpile(circ, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                       seed_transpiler=self.seed) if noise is not None else transpile(circ, sim)
        counts = sim.run(tc, shots=shots).result().get_counts()
        # Qiskit count keys are bitstrings with qubit 0 on the RIGHT.
        return {int(key.replace(" ", ""), 2): v for key, v in counts.items()}

    def circuit_stats(self, params, p: int) -> dict:
        circ = self.bound_measured_circuit(params, p)
        logical = circ.count_ops()
        tc = transpile(circ, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=self.seed)
        ops = tc.count_ops()
        return {
            "qubits": circ.num_qubits,
            "p": p,
            "logical_rzz": int(logical.get("rzz", 0)),
            "logical_depth": circ.depth(),
            "transpiled_depth": tc.depth(),
            "transpiled_cx": int(ops.get("cx", 0)),
            "transpiled_1q": int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))),
            "note": "all-to-all connectivity assumed; real heavy-hex hardware would add SWAPs",
        }


def _interp(v: np.ndarray) -> np.ndarray:
    """INTERP: extend p optimal angles to p+1 by linear interpolation."""
    p = len(v)
    out = np.zeros(p + 1)
    for i in range(p + 1):
        left = v[i - 1] if i - 1 >= 0 else 0.0
        right = v[i] if i < p else 0.0
        out[i] = (i / p) * left + ((p - i) / p) * right
    return out


def simple_noise_model(p1: float = 1e-3, p2: float = 1e-2, readout: float = 0.02) -> NoiseModel:
    """Generic depolarising + readout noise of roughly current superconducting-device magnitude.
    Not a model of any specific IBM backend."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm
