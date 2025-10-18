# shallow_qfm.py  — versão rasa (QAOA-like) do seu feature map, sem Trotter

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ======================
# Seeds & dados
# ======================
SEED = 42
NUM_LINES = 300
N_REPS = 5
FOLDS = 10

BASE = 'Toxicity13'
# BASE = 'credit'

if BASE == 'credit':
    df_13 = pd.read_csv('./data/credit_card_default_UCI.csv', encoding='utf8')
else:
    df_13 = pd.read_csv('./data/Toxicity-13F.csv', encoding='utf8')
    # Troca NonToxic por 1 e Toxic por 0
    df_13['Class'] = (df_13['Class'] == 'NonToxic').astype(int)


def remove_low_mi(X, y, mi_lowerbound):
    mi = mutual_info_classif(
        X, y, n_neighbors=5, discrete_features=False, random_state=SEED
    )
    return [j for j in range(X.shape[1]) if mi[j] > mi_lowerbound]

if df_13.shape[0] > NUM_LINES and BASE == 'credit':
    sss = StratifiedShuffleSplit(n_splits=1, train_size=NUM_LINES, random_state=SEED)
    idx_sub, _ = next(sss.split(df_13, df_13['default_next_month'].values))
    df_13 = df_13.iloc[idx_sub].reset_index(drop=True)
else:
    NUM_LINES = df_13.shape[0]

if BASE == 'credit':
    y = df_13['default_next_month'].values
else:
    y = df_13['Class'].values


X = df_13.iloc[:, :-1].values


if BASE == 'credit':
    col_sel = remove_low_mi(X, y, 0.008)
    print('Colunas selecionadas: ', col_sel)
    X = X[:, col_sel]

def preprocess_train_test(X_train, X_test):
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_train))
    Xte = sc.transform(imp.transform(X_test))
    return Xtr, Xte

# ======================
# Backend & Estimator
# ======================
backend = AerSimulator(method="statevector")
np.random.seed(SEED)    
backend.options.seed_simulator = SEED
backend.options.seed_transpiler = SEED
backend.set_options(seed_simulator=SEED)
estimator = Estimator(backend)

# ======================
# Schedule e Hamiltonianos
# ======================
A_of_s = 2*np.pi * 15e9  # rad/s
B_of_s = 2*np.pi * 11e9  # rad/s

# A_of_s = 2*np.pi * 1e9  # rad/s
# B_of_s = 2*np.pi * 1e9  # rad/s

# tau    = 35e-9           # s
k_max  = 1               # usa só <Z_i> (k<=1)

# --- parâmetros do circuito raso ---
p_periods = 15           # 2–3 costuma bastar
use_cd    = False       # True ativa termo CD local (Ry)
alphas_cd = None        # ex.: [0.08,0.05,0.03] se use_cd=True

def s_curve(t_norm: float) -> float:
    """t_norm in [0,1]  →  lambda(t) in [0,1]"""
    return np.sin(0.5*np.pi * np.sin(0.5*np.pi * t_norm)**2)**2

def make_HD_HP(h_x, J_dict):
    n = len(h_x)
    HD = SparsePauliOp.from_list([("I"*i + "X" + "I"*(n-i-1), -1.0) for i in range(n)])
    terms_HP = []
    for i, hi in enumerate(h_x):
        if hi != 0:
            terms_HP.append(("I"*i + "Z" + "I"*(n-i-1), float(hi)))
    for (i, j), Jij in J_dict.items():
        z = ["I"] * n
        z[i] = z[j] = "Z"
        terms_HP.append(("".join(z), float(Jij)))
    HP = SparsePauliOp.from_list(terms_HP)
    return HD, HP

# -------- β,γ via integração do schedule em p janelas --------
def betas_gammas_from_schedule(tau, p, A_amp, B_amp, ngrid=4000):
    t = np.linspace(0.0, tau, ngrid, endpoint=True)
    s = np.array([s_curve(tt / tau) for tt in t])
    a = A_amp * (1.0 - s)   # rad/s
    b = B_amp * s           # rad/s
    edges = np.linspace(0, ngrid - 1, p + 1, dtype=int)
    betas, gammas = [], []
    for k in range(p):
        lo, hi = edges[k], edges[k + 1]
        betas.append(np.trapezoid(a[lo:hi+1], t[lo:hi+1]))    # rad
        gammas.append(np.trapezoid(b[lo:hi+1], t[lo:hi+1]))  # rad
    return np.array(betas), np.array(gammas)

# -------- circuito raso (QAOA-like) --------
def build_shallow_quench_circuit(h_x, J_dict, tau, p, A_amp, B_amp,
                                 add_cd=False, alphas=None):
    """
    U = ∏_k exp(-i beta_k HD) exp(-i gamma_k HP), com (beta_k,gamma_k)
    integrados do schedule. Opcional: termo CD local (Ry) por período.
    """
    n = len(h_x)
    HD, HP = make_HD_HP(h_x, J_dict)
    betas, gammas = betas_gammas_from_schedule(tau, p, A_amp, B_amp)

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for k in range(p):
        qc.append(PauliEvolutionGate(HD, time=betas[k]), range(n))
        if add_cd:
            alpha = 0.0 if (alphas is None) else float(alphas[k])
            if abs(alpha) > 0:
                qc.ry(2*alpha, range(n))
        qc.append(PauliEvolutionGate(HP, time=gammas[k]), range(n))
        qc.barrier()
    return qc

# -------- observáveis Z de peso ≤ k_max --------
def Z_observables(n, k_max=2):
    obs = []
    for i in range(n):  # peso-1
        obs.append(SparsePauliOp.from_list([("I"*i + "Z" + "I"*(n-i-1), 1.0)]))
    for k in range(2, k_max+1):  # pesos >1, se desejar
        for idx in combinations(range(n), k):
            p = ["I"]*n
            for j in idx: p[j] = "Z"
            obs.append(SparsePauliOp.from_list([("".join(p), 1.0)]))
    return obs

# -------- feature vector (só versão shallow) --------
def quantum_feature_vector_shallow(x_row, rho_ff, pairs, tau, p, A_amp, B_amp,
                                   k_max=1, backend=None, estimator=None,
                                   add_cd=False, alphas=None):
    n = len(x_row)
    h_vec = np.asarray(x_row, float)                           # h_i = x_i
    J_dict = {(i, j): float(rho_ff[i, j]) for (i, j) in pairs} # J_ij = rho_ij
    qc = build_shallow_quench_circuit(h_vec, J_dict, tau, p, A_amp, B_amp,
                                      add_cd=add_cd, alphas=alphas)
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend, seed_transpiler=SEED)
    qc_t = pm.run(qc)

    obs_list = Z_observables(n, k_max=k_max)
    obs_isa  = [obs.apply_layout(qc_t.layout) for obs in obs_list]

    job = estimator.run([(qc_t, obs_isa)])
    expvals = job.result()[0].data.evs
    return np.asarray(expvals, dtype=np.float64)

def q_feature_matrix_shallow(X, rho_ff, pairs, tau, p, A_amp, B_amp,
                             k_max=1, backend=None, estimator=None, desc="qmap",
                             add_cd=False, alphas=None):
    feats = []
    for row in tqdm(X, desc=desc, leave=True):
        feats.append(
            quantum_feature_vector_shallow(row, rho_ff, pairs, tau, p, A_amp, B_amp,
                                           k_max=k_max, backend=backend, estimator=estimator,
                                           add_cd=add_cd, alphas=alphas)
        )
    return np.vstack(feats)

# ======================
# CV + treinamento
# ======================

tau = 20e-9

# for k in range(1, 41):      
#     tau = k * 1e-9  
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
skf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=N_REPS, random_state=SEED)
metrics = {k: [] for k in [
    'AUC','F1-Score Overall','Balanced Accuracy',
    'Precision Class 0','Precision Class 1','Recall Class 0','Recall Class 1'
]}
qc_metrics = {k: [] for k in metrics.keys()}

rho_thr = 0
gamma_q = 1.0
out_dir = Path(f".\{BASE}_reps{N_REPS}_folds{FOLDS}_lines{NUM_LINES}_p{p_periods}\out_csv_shallow_tau{tau}_{BASE}_{NUM_LINES}")
out_dir.mkdir(parents=True, exist_ok=True)

for fold, (train_id, test_id) in enumerate(skf.split(X, y)):
    print(f'Fold: {fold}')
    X_tr_raw, X_te_raw = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]

    X_train, X_test = preprocess_train_test(X_tr_raw, X_te_raw)

    # baseline (clássico)
    from sklearn.ensemble import GradientBoostingClassifier
    base = GradientBoostingClassifier(random_state=SEED)
    base.fit(X_train, y_train)
    y_pred = base.predict(X_test)
    y_proba = base.predict_proba(X_test)[:, 1]
    metrics['AUC'].append(roc_auc_score(y_test, y_proba))
    metrics['F1-Score Overall'].append(f1_score(y_test, y_pred))
    metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_test, y_pred))
    metrics['Precision Class 0'].append(precision_score(y_test, y_pred, pos_label=0))
    metrics['Precision Class 1'].append(precision_score(y_test, y_pred, pos_label=1))
    metrics['Recall Class 0'].append(recall_score(y_test, y_pred, pos_label=0))
    metrics['Recall Class 1'].append(recall_score(y_test, y_pred, pos_label=1))

    # correlação entre features (só do treino)
    rho_ff = np.corrcoef(X_train, rowvar=False)
    np.fill_diagonal(rho_ff, 0)
    n_class = X_train.shape[1]
    pairs = [(i, j) for i in range(n_class) for j in range(i+1, n_class)
            if abs(rho_ff[i, j]) >= rho_thr]

    # features quânticas SHALLOW
    Xq_train = q_feature_matrix_shallow(
        X_train, rho_ff, pairs, tau, p_periods, A_of_s, B_of_s,
        k_max=k_max, backend=backend, estimator=estimator,
        desc="  train(shallow)", add_cd=use_cd, alphas=alphas_cd
    )
    Xq_test = q_feature_matrix_shallow(
        X_test, rho_ff, pairs, tau, p_periods, A_of_s, B_of_s,
        k_max=k_max, backend=backend, estimator=estimator,
        desc="  test(shallow)", add_cd=use_cd, alphas=alphas_cd
    )

    # concatenação: [clássico | quântico]
    X_aug_train = np.hstack([X_train, gamma_q * Xq_train]).astype(np.float32)
    X_aug_test  = np.hstack([X_test , gamma_q * Xq_test ]).astype(np.float32)

    qc_model = GradientBoostingClassifier(random_state=SEED)
    qc_model.fit(X_aug_train, y_train)
    y_pred  = qc_model.predict(X_aug_test)
    y_proba = qc_model.predict_proba(X_aug_test)[:, 1]

    qc_metrics['AUC'].append(roc_auc_score(y_test, y_proba))
    qc_metrics['F1-Score Overall'].append(f1_score(y_test, y_pred))
    qc_metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_test, y_pred))
    qc_metrics['Precision Class 0'].append(precision_score(y_test, y_pred, pos_label=0))
    qc_metrics['Precision Class 1'].append(precision_score(y_test, y_pred, pos_label=1))
    qc_metrics['Recall Class 0'].append(recall_score(y_test, y_pred, pos_label=0))
    qc_metrics['Recall Class 1'].append(recall_score(y_test, y_pred, pos_label=1))

    # --- salvamentos (iguais aos seus, ajustados para 'shallow') ---
    n_q = Xq_train.shape[1]
    cols_class = [f"class_{i}" for i in range(n_class)]
    cols_quant = [f"qf_{i}" for i in range(n_q)]
    all_cols   = cols_class + cols_quant

    df_train = pd.DataFrame(X_aug_train, columns=all_cols).astype(np.float32)
    df_train["y"] = y_train.astype(int); df_train["set"] = "train"
    df_train["fold"] = fold; df_train["row_id"] = train_id
    df_test = pd.DataFrame(X_aug_test, columns=all_cols).astype(np.float32)
    df_test["y"] = y_test.astype(int); df_test["set"] = "test"
    df_test["fold"] = fold; df_test["row_id"] = test_id
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    csv_features_path = out_dir / f"features_y_fold{fold}.csv"
    df_all.to_csv(csv_features_path, index=False)
    print(f"Salvo: {csv_features_path}")

    df_rho = pd.DataFrame(rho_ff, index=cols_class, columns=cols_class)
    csv_rho_path = out_dir / f"rho_ff_fold{fold}.csv"
    df_rho.to_csv(csv_rho_path, float_format="%.8g")
    np.save(out_dir / f"rho_ff_fold{fold}.npy", rho_ff)
    np.save(out_dir / f"train_idx_fold{fold}.npy", train_id)
    np.save(out_dir / f"test_idx_fold{fold}.npy", test_id)

    meta = {
        "fold": fold,
        "n_class": int(n_class),
        "n_q": int(n_q),
        "gamma_q": float(gamma_q),
        "tau": float(tau),
        "p_periods": int(p_periods),
        "use_cd": bool(use_cd),
        "alphas_cd": alphas_cd if alphas_cd is not None else [],
        "rho_thr": float(rho_thr),
        "pairs_count": int(len(pairs)),
    }
    pd.Series(meta).to_json(out_dir / f"meta_fold{fold}.json", indent=2)

# ======================
# Agregação de métricas
# ======================
COLS = [
    "AUC","F1-Score Overall","Balanced Accuracy",
    "Precision Class 0","Precision Class 1",
    "Recall Class 0","Recall Class 1",
]

def metrics_to_df(metrics_dict, qc_flag, **extra):
    n_by_col = {c: len(metrics_dict[c]) for c in COLS}
    if len(set(n_by_col.values())) != 1:
        raise ValueError(f"Tamanhos diferentes nas listas: {n_by_col}")
    df = pd.DataFrame({c: metrics_dict[c] for c in COLS})
    df["qc"] = qc_flag
    for k, v in extra.items():
        df[k] = v
    return df

df_qc = metrics_to_df(qc_metrics, "QC-shallow", tau=tau, p=p_periods)
df_cl = metrics_to_df(metrics,   "Classical")

df_all = pd.concat([df_qc, df_cl], ignore_index=True)
out_csv = f".\{BASE}_reps{N_REPS}_folds{FOLDS}_lines{NUM_LINES}_p{p_periods}\\new_note_results_qc_vs_classical_shallow_tau{tau}_{BASE}_{NUM_LINES}.csv"

df_all.to_csv(out_csv, index=False)
print(f"Salvo em {out_csv}")
