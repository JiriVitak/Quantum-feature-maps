from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Layout
from qiskit.circuit import ParameterVector   # <<< NOVO
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import RZZGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from itertools import combinations
from collections import Counter

import json

import numpy as np

SEED = 42
np.random.seed(SEED)

# -------- parametrized circuit (same topology, angles vary per row) -----
def build_quench_circuit_param(num_qubits, edges_act, m=1):
    """
    edges_act: list of edges (i,j) with J_ij != 0 (logical indices).
    Uses:
      - one parameter per qubit and per step (terms Y_i)
      - one parameter per edge and per step (same angle in YZ and ZY)
    Returns:
      - the circuit
      - param_order has the format [ thy(0,0), thy(0,1), ..., thy(0,n-1),
                                    thy(1,0), ..., thy(m-1,n-1),
                                    the(0,0), ..., the(0,n_edges-1),
                                    ...,
                                    the(m-1,n_edges-1) ]

    """
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    n = num_qubits
    n_edges = len(edges_act)

    theta_y = ParameterVector("thy", m * n)
    theta_e = ParameterVector("the", m * n_edges)

    def idx_y(step, i):
        return step * n + i

    def idx_e(step, e_idx):
        return step * n_edges + e_idx

    for step in range(m):
        # terms Y_i
        for i in range(n):
            th = theta_y[idx_y(step, i)]
            add_exp_minus_i_theta_Y(qc, i, th)

        # terms YZ/ZY per edge
        for e_idx, (i, j) in enumerate(edges_act):
            th = theta_e[idx_e(step, e_idx)]
            add_exp_minus_i_theta_YZ(qc, i, j, th)
            add_exp_minus_i_theta_ZY(qc, i, j, th)

    # implicit order of parameters: all thy then all the
    param_order = list(theta_y) + list(theta_e)
    return qc, param_order

def build_quantum_features_all(
    X_q_all, J_dict, perm, phys_nodes, tau, m, backend, estimator, simulation=False
):
    """
    Generates the quantum feature matrix Xq_all (N x n_q) in A SINGLE BATCH.
    """
    N, n = X_q_all.shape

    # reorder features -> logical qubits
    X_q_reordered = X_q_all[:, perm]

    # active edges (with Jij != 0), (min(i, j), max(i, j)) is to always have i < j
    edges_act = sorted(
        {(min(i, j), max(i, j)) for (i, j) in J_dict.keys()}
    )

    # parametrized circuit
    qc_param, param_order = build_quench_circuit_param(n, edges_act, m=m)

    # transpile once
    if not simulation:
        initial_layout = Layout(
            {qc_param.qubits[i]: phys_nodes[i] for i in range(n)}
        )
        qc_t = transpile(
            qc_param,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=0,
            seed_transpiler=SEED,
            routing_method="none",
        )
    else:
        pm = generate_preset_pass_manager(
            optimization_level=0, backend=backend, seed_transpiler=SEED
        )
        qc_t = pm.run(qc_param)

    if simulation == False:
        test_circuit_t(qc_t, backend)

    # observables Z_i and Z_iZ_j
    # obs_list, pairs_2q = Z_observables_edges(n, edges_act)
    obs_list, pairs_2q = Z_observables(n)
    obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]

    # It is necessary to add an extra dimension to the observables
    obs_isa_broadcast = [[obs] for obs in obs_isa]  # shape (n_obs, 1)    

    # parameter matrix for ALL rows
    theta_values_all = make_theta_matrix_all(
        X_q_reordered, J_dict, edges_act, tau, m
    )


    # after qc_t and theta_values_all are ready, it is necessary to reorder the columns of theta_values_all
    # so that it matches the order of the parameters of the transpiled circuit qc_t

    param_t = list(qc_t.parameters)          # REAL order used by the Estimator
    pos = {p: k for k, p in enumerate(param_order)}  # order used in the parameter matrix

    idx = [pos[p] for p in param_t]          # how to permute columns
    theta_values_all = theta_values_all[:, idx]


    # single job 
    job = estimator.run([(qc_t, obs_isa_broadcast, theta_values_all)])

    result = job.result()[0]
    evs = np.asarray(result.data.evs, dtype=float)   # should be something like (n_obs, N_samples) 

    # features per row (N_samples, n_obs)
    Xq_all = evs.T

    return Xq_all, pairs_2q

def build_quantum_features_all_job(
    X_q_all, J_dict, perm, phys_nodes, tau, m, backend, estimator, simulation=False
):
    """
    Generates the quantum feature matrix Xq_all (N x n_q) in A SINGLE BATCH.
    """
    N, n = X_q_all.shape

    # reorder features -> logical qubits
    X_q_reordered = X_q_all[:, perm]

    # active edges (with Jij != 0), (min(i, j), max(i, j)) is to always have i < j
    edges_act = sorted(
        {(min(i, j), max(i, j)) for (i, j) in J_dict.keys()}
    )

    # parametrized circuit
    qc_param, param_order = build_quench_circuit_param(n, edges_act, m=m)

    # transpile once
    if not simulation:
        initial_layout = Layout(
            {qc_param.qubits[i]: phys_nodes[i] for i in range(n)}
        )
        qc_t = transpile(
            qc_param,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=0,
            seed_transpiler=SEED,
            routing_method="none",
        )
    else:
        pm = generate_preset_pass_manager(
            optimization_level=0, backend=backend, seed_transpiler=SEED
        )
        qc_t = pm.run(qc_param)

    if simulation == False:
        test_circuit_t(qc_t, backend)

    # observables Z_i and Z_iZ_j
    # obs_list, pairs_2q = Z_observables_edges(n, edges_act)
    obs_list, pairs_2q = Z_observables(n)
    obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]

    # It is necessary to add an extra dimension to the observables
    obs_isa_broadcast = [[obs] for obs in obs_isa]  # shape (n_obs, 1)    

    # parameter matrix for ALL rows
    theta_values_all = make_theta_matrix_all(
        X_q_reordered, J_dict, edges_act, tau, m
    )


    # after qc_t and theta_values_all are ready, it is necessary to reorder the columns of theta_values_all
    # so that it matches the order of the parameters of the transpiled circuit qc_t

    param_t = list(qc_t.parameters)          # REAL order used by the Estimator
    pos = {p: k for k, p in enumerate(param_order)}  # order used in the parameter matrix

    idx = [pos[p] for p in param_t]          # how to permute columns
    theta_values_all = theta_values_all[:, idx]


    # single job 
    job = estimator.run([(qc_t, obs_isa_broadcast, theta_values_all)])

    #result = job.result()[0]
    #evs = np.asarray(result.data.evs, dtype=float)   # should be something like (n_obs, N_samples) 

    # features per row (N_samples, n_obs)
    #Xq_all = evs.T
    
    job_id = job.job_id()
    return job_id, pairs_2q

import numpy as np

def build_quantum_features_all_ideal(
    X_q_all,
    J_dict,
    tau,
    m,
    estimator,
    backend=None,
):
    """
    Generates Xq_all (N x n_obs) in A SINGLE BATCH, assuming ideal simulation (logical qubits).

    Premises:
      - X_q_all: (N, n) is already in the order of logical qubits (does not use perm).
      - J_dict contains 2-way correlations between features; since we are in ideal simulation,
        we treat the graph as complete (all edges (i,j) with i<j).
      - phys_nodes/layout are not used.
      - Does not apply layout to observables.

    Returns:
      - Xq_all: matrix (N, n_obs) with expected values (quantum features)
      - pairs_2q: list of pairs (i,j) corresponding to 2-qubit features (if Z_observables returns this)
    """
    N, n = X_q_all.shape

    # In ideal simulation, we assume a complete graph (all 2-way correlations)
    edges_act = sorted({(min(i,j), max(i,j)) for (i,j) in J_dict.keys()})

    # Parameterized circuit (logical)
    qc_param, param_order = build_quench_circuit_param(n, edges_act, m=m)

    # Observables in the logical space (without apply_layout)
    obs_list, pairs_2q = Z_observables(n)
    obs_broadcast = [[obs] for obs in obs_list]  # shape (n_obs, 1)

    # Parameter matrix (N_samples x n_params), in the "param_order" order (from the builder)
    theta_values_all = make_theta_matrix_all(
        X_q_all, J_dict, edges_act, tau, m
    )

    # Reorders columns to match the REAL order of circuit parameters (qc_param)
    # (Usually, qc_param.parameters is the effective order used by the Estimator.)
    param_t = list(qc_param.parameters)
    pos = {p: k for k, p in enumerate(param_order)}
    idx = [pos[p] for p in param_t]
    theta_values_all = theta_values_all[:, idx]

    # Single job (batch)
    # Note: some backends/estimators accept 'backend' here, others do not. Kept optional.
    if backend is None:
        job = estimator.run([(qc_param, obs_broadcast, theta_values_all)])
    else:
        job = estimator.run([(qc_param, obs_broadcast, theta_values_all)], backend=backend)

    result = job.result()[0]
    evs = np.asarray(result.data.evs, dtype=float)   # (n_obs, N)

    # Features per row: (N, n_obs)
    Xq_all = evs.T
    return Xq_all, pairs_2q


def make_theta_matrix_all(X_q_reordered, J_dict, edges_act, tau, m):
    """
    Generates parameter matrix (all rows) in the same order as build_quench_circuit_param:
      [thy (m*n), the (m*|edges_act|)].
    """
    N, n = X_q_reordered.shape
    n_edges = len(edges_act)
    dt = tau / m

    theta_y_vals = np.zeros((N, m * n), dtype=np.float64)
    theta_e_vals = np.zeros((N, m * n_edges), dtype=np.float64)

    def J_val(i, j):
        return float(J_dict[(i, j)]) if (i, j) in J_dict else float(
            J_dict[(j, i)]
        )

    Jij_vec = np.array([J_val(i, j) for (i, j) in edges_act], dtype=np.float64)

    for step in range(m):
        t_mid = (step + 0.25) * dt
        s = s_curve(t_mid / tau)
        ds = ds_curve(t_mid, tau)

        for r in range(N):
            h_vec = X_q_reordered[r]
            alpha1 = makeAlpha1(h_vec, J_dict, s)
            coeff_global = -2 * ds * dt * alpha1

            theta_y_vals[r, step * n : (step + 1) * n] = coeff_global * h_vec
            theta_e_vals[
                r, step * n_edges : (step + 1) * n_edges
            ] = coeff_global * Jij_vec

    return np.hstack([theta_y_vals, theta_e_vals])


import numpy as np

def make_theta_matrix_cluster_using_global_alpha(
    X_q_full_reordered,     # (N, n_b)  -> X_q_all[:, perm]
    cluster_nodes_sorted,   # list of logical nodes of the cluster (in the space 0..n_b-1)
    J_dict_global,          # J_dict of the global run (in the space 0..n_b-1, active physical edges)
    old2new,                # maps global logical node -> local index 0..k-1
    edges_act_local,        # list of local edges (i,j) in indices 0..k-1
    tau, m,
    makeAlpha1_func,        # sua makeAlpha1 (do qfm_lib)
    s_curve_func,           # sua s_curve
    ds_curve_func,          # sua ds_curve
):
    """
    Generates theta_values_all (N, m*k + m*|E_local|), but using the SAME alpha1 from the global.
    """
    N, n_b = X_q_full_reordered.shape
    k = len(cluster_nodes_sorted)
    n_edges = len(edges_act_local)
    dt = tau / m

    # h_local por linha
    X_local = X_q_full_reordered[:, cluster_nodes_sorted]  # (N, k)

    # Jij_local vector (in edges_act_local order), extracted from J_dict_global
    # We need to map (i_local, j_local) -> (i_global, j_global)
    new2old = {v: u for (u, v) in old2new.items()}

    def J_global_val(a_global, b_global):
        if (a_global, b_global) in J_dict_global:
            return float(J_dict_global[(a_global, b_global)])
        if (b_global, a_global) in J_dict_global:
            return float(J_dict_global[(b_global, a_global)])
        return 0.0

    Jij_vec = np.zeros(n_edges, dtype=np.float64)
    for e_idx, (iL, jL) in enumerate(edges_act_local):
        iG = new2old[iL]
        jG = new2old[jL]
        Jij_vec[e_idx] = J_global_val(iG, jG)

    theta_y_vals = np.zeros((N, m * k), dtype=np.float64)
    theta_e_vals = np.zeros((N, m * n_edges), dtype=np.float64)

    for step in range(m):
        t_mid = (step + 0.25) * dt
        s = s_curve_func(t_mid / tau)
        ds = ds_curve_func(t_mid, tau)

        for r in range(N):
            h_full = X_q_full_reordered[r]                 # (n_b,)
            alpha1 = makeAlpha1_func(h_full, J_dict_global, s)
            coeff_global = -2 * ds * dt * alpha1

            h_loc = X_local[r]                              # (k,)
            theta_y_vals[r, step*k:(step+1)*k] = coeff_global * h_loc
            theta_e_vals[r, step*n_edges:(step+1)*n_edges] = coeff_global * Jij_vec

    return np.hstack([theta_y_vals, theta_e_vals])



def Z_observables(n, k_max=2):
    obs = []
    # weight-1
    for i in range(n):
        obs.append(SparsePauliOp.from_list([( "I"*i + "Z" + "I"*(n-i-1), 1.0 )]))
    # weights 2..k_max
    uniq_edges = []
    for k in range(2, k_max+1):
        for idx in combinations(range(n), k):
            p = ["I"]*n
            for j in idx: p[j]="Z"
            obs.append(SparsePauliOp.from_list([("".join(p), 1.0)]))
            if k == 2:
                uniq_edges.append(idx)  # idx already comes sorted (i<j)
    return obs, uniq_edges

def Z_observables_edges(n, edges_act):
    """
    Observables:
      - Z_i (1-local) for all i
      - Z_i Z_j for each unique edge (i<j) in edges_act
    """
    obs = []
    # 1-local
    for i in range(n):
        obs.append(
            SparsePauliOp.from_list(
                [("I" * i + "Z" + "I" * (n - i - 1), 1.0)]
            )
        )

    # 2-local on active edges (unique)
    uniq_edges = []
    seen = set()
    for (i, j) in edges_act:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        uniq_edges.append(key)

    for (i, j) in uniq_edges:
        p = ["I"] * n
        p[i] = "Z"
        p[j] = "Z"
        obs.append(
            SparsePauliOp.from_list([("".join(p), 1.0)])
        )
    return obs, uniq_edges  # uniq_edges = 2-local pairs (for names)

def test_circuit_t(qc_t, backend):
    print("Depth:", qc_t.depth())
    print("dt (s):", backend.target.dt)

    ops = qc_t.count_ops()
    print("SWAP =", ops.get("swap", 0))

    d2 = qc_t.depth(
        filter_function=lambda inst: inst.operation.num_qubits == 2
    )
    d1 = qc_t.depth(
        filter_function=lambda inst: inst.operation.num_qubits == 1
    )
    print("2q-depth =", d2)
    print("1q-depth =", d1)

    cm_edges = set(map(tuple, backend.coupling_map.get_edges()))
    bad = []
    for ci in qc_t.data:
        op = ci.operation
        if op.num_qubits == 2:
            a = qc_t.find_bit(ci.qubits[0]).index
            b = qc_t.find_bit(ci.qubits[1]).index
            if (a, b) not in cm_edges and (b, a) not in cm_edges:
                bad.append((op.name, a, b))
    print("2q pairs outside the coupling map =", len(bad))

    per_q = Counter()
    twoq = Counter()
    for ci in qc_t.data:
        op = ci.operation
        if op.num_qubits == 2:
            twoq[op.name] += 1
            qs = [qc_t.find_bit(q).index for q in ci.qubits]
            for q in qs:
                per_q[q] += 1

    total_2q = sum(twoq.values())
    involved = len(per_q)
    nq = qc_t.num_qubits

    print("2q gate types:", twoq)
    print("Total 2q gates =", total_2q)
    print("Qubits involved in 2q =", involved, f"(of {nq})")
    if involved > 0:
        print(
            "Average 2q per qubit (involved) =",
            sum(per_q.values()) / involved,
        )
    print(
        "Average 2q per qubit (all) =",
        (sum(per_q.values()) / nq) if nq else 0,
    )
    print("Top loaded:", per_q.most_common(10))

# 4) HAMILTONIAN AND CIRCUIT FUNCTIONS
def ds_curve(t, T):
    s = t / T
    D = -((np.pi**2) / (4 * T)) * np.sin(np.pi * s) * np.sin(
        np.pi * (np.sin((np.pi / 2) * s) ** 2)
    )
    return D


def s_curve(t_norm: float) -> float:
    return np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * t_norm) ** 2) ** 2


def makeRt(h_x, J_dict, s, sum_hi2, sum_Jij2):
    sum_hi4 = 0
    sum_Jij4 = 0
    n = len(h_x)

    for hi in h_x:
        sum_hi4 += hi**4

    for (i, j), Jij in J_dict.items():
        sum_Jij4 += Jij**4

    sum_hi2Jii2 = 0
    for (i, j), Jij in J_dict.items():
        hi = h_x[i]
        hj = h_x[j]
        sum_hi2Jii2 += (hi**2 + hj**2) * (Jij**2)

    contrib_duplet = 2 * 6 * sum_hi2Jii2  # i<>j

    E = {(min(i, j), max(i, j)) for (i, j) in J_dict.keys()}

    def J(i, j):
        return J_dict[(i, j)] if (i, j) in J_dict else J_dict[(j, i)]

    sum_triplet = 0.0
    for i, j, k in combinations(range(n), 3):
        e1 = (min(i, j), max(i, j))
        e2 = (min(i, k), max(i, k))
        e3 = (min(j, k), max(j, k))

        if e1 in E and e2 in E and e3 in E:
            Jij = J(i, j)
            Jik = J(i, k)
            Jjk = J(j, k)
            sum_triplet += (
                Jij**2 * Jik**2 + Jij**2 * Jjk**2 + Jik**2 * Jjk**2
            )

    contrib_triplet = 6.0 * sum_triplet  # i<j<k

    Rt = ((1 - s) ** 2) * (sum_hi2 + 4 * sum_Jij2) + (s**2) * (
        sum_hi4 + 2 * sum_Jij4 + contrib_duplet + contrib_triplet
    )
    return Rt


def makeAlpha1(h_x, J_dict, s):
    sum_hi2 = 0
    sum_Jij2 = 0

    for hi in h_x:
        sum_hi2 += hi**2

    for (i, j), Jij in J_dict.items():
        sum_Jij2 += Jij**2

    alpha1 = -(1 / 4) * (sum_hi2 + 2 * sum_Jij2)
    Rt = makeRt(h_x, J_dict, s, sum_hi2, 2 * sum_Jij2)
    alpha1 = alpha1 / Rt
    return alpha1


def add_exp_minus_i_theta_Y(qc, i, theta):
    qc.rx(+np.pi / 2, i)
    qc.rz(2 * theta, i)
    qc.rx(-np.pi / 2, i)


def add_exp_minus_i_theta_YZ(qc, i, j, theta):
    qc.rx(+np.pi / 2, i)
    qc.append(RZZGate(2 * theta), [i, j])
    qc.rx(-np.pi / 2, i)


def add_exp_minus_i_theta_ZY(qc, i, j, theta):
    qc.rx(+np.pi / 2, j)
    qc.append(RZZGate(2 * theta), [i, j])
    qc.rx(-np.pi / 2, j)