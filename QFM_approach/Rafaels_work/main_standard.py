import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

from itertools import combinations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pathlib import Path
from datetime import datetime

df_13 = pd.read_csv('./data/Toxicity-13F.csv', encoding='utf8')

# Troca NonToxic por 1 e Toxic por 0
df_13['Class'] = (df_13['Class'] == 'NonToxic').astype(int)

X = df_13.iloc[:,:-1].values
y = df_13['Class'].values

def preprocess_train_test(X_train, X_test):
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_train))
    Xte = sc.transform(imp.transform(X_test))
    return Xtr, Xte

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Armazenar as métricas
metrics = {
    'AUC': [],
    'F1-Score Overall': [],
    'Balanced Accuracy': [],
    'Precision Class 0': [],
    'Precision Class 1': [],
    'Recall Class 0': [],
    'Recall Class 1': []
}

for fold, (train_id, test_id) in enumerate(skf.split(X,y)):   
    X_tr_raw, X_te_raw = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]

    # 1) preprocess: mediana + StandardScaler (fit só no treino)
    X_train, X_test = preprocess_train_test(X_tr_raw, X_te_raw)     

    # 1) MI global
    mi = mutual_info_classif(
        X_train, y_train,
        discrete_features=False,
        random_state=42
    )    

    print(mi)
    
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)    

    # Previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    metrics['AUC'].append(roc_auc_score(y_test, y_proba))
    metrics['F1-Score Overall'].append(f1_score(y_test, y_pred))
    metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_test, y_pred))
    metrics['Precision Class 0'].append(precision_score(y_test, y_pred, pos_label=0))
    metrics['Precision Class 1'].append(precision_score(y_test, y_pred, pos_label=1))
    metrics['Recall Class 0'].append(recall_score(y_test, y_pred, pos_label=0))
    metrics['Recall Class 1'].append(recall_score(y_test, y_pred, pos_label=1)) 

# Calcular médias
mean_metrics = {k: np.mean(v) for k, v in metrics.items()}

# Exibir resultados
print("Métricas médias nos 5 folds:")
for k, v in mean_metrics.items():
    print(f"{k}: {v:.4f}")


# Exibir resultados
median_metrics = {k: np.median(v) for k, v in metrics.items()}

print("Métricas medianas nos 5 folds:")
for k, v in median_metrics.items():
    print(f"{k}: {v:.4f}")


backend = AerSimulator(method="statevector")

SEED = 42
np.random.seed(SEED)

backend.options.seed_simulator = SEED
backend.options.seed_transpiler = SEED

backend.set_options(seed_simulator=SEED)
estimator = Estimator(backend)

n=13
A_of_s = 2*np.pi * 15e9
B_of_s = 2*np.pi * 11e9
order, reps = 2, 64
m = int(np.ceil(2000 / reps))   # P ≈ 2k
tau = 30e-9
k_max = 1    


# Simula o schedule D-Wave usando função seno → funções A_of_s, B_of_s (rad/s) --------
def s_curve(t_norm: float) -> float:
    """t_norm in [0,1]  →  lambda(t) in [0,1]"""
    return np.sin(0.5*np.pi * np.sin(0.5*np.pi * t_norm)**2)**2

# -------- 2) H_d e H_p  --------
def make_HD_HP(h_x, J_dict):
    n = len(h_x)
    HD = SparsePauliOp.from_list([( "I"*i + "X" + "I"*(n-i-1), -1.0) for i in range(n)])
    terms_HP = []
    for i, hi in enumerate(h_x):
        if hi != 0: terms_HP.append(("I"*i + "Z" + "I"*(n-i-1), float(hi)))
    for (i,j), Jij in J_dict.items():
        z = ["I"]*n; z[i]=z[j]="Z"
        terms_HP.append(("".join(z), float(Jij)))
    HP = SparsePauliOp.from_list(terms_HP)
    return HD, HP

# -------- 3) circuito do quench --------
def build_quench_circuit(h_x, J_dict, tau, m, A_of_s, B_of_s,
                         order=2, reps=12, insert_barriers=True, preserve_order=True):
    n  = len(h_x)
    dt = tau/m
    HD, HP = make_HD_HP(h_x, J_dict)

    qc = QuantumCircuit(n)
    qc.h(range(n))

    synth = SuzukiTrotter(order=order, reps=reps,
                          insert_barriers=insert_barriers,
                          preserve_order=preserve_order)   # wrap=False por padrão

    for k in range(m):
        t_mid = (k + 0.5)*dt
        s = s_curve(t_mid / tau)                
        
        Hk = A_of_s*HD*(1-s) + B_of_s*HP*s
        qc.append(PauliEvolutionGate(Hk, time=dt, synthesis=synth), range(n))
        if insert_barriers: qc.barrier()

    return qc

# -------- 4) observáveis Z de peso ≤ k_max --------
def Z_observables(n, k_max=2):
    obs = []
    # peso-1
    for i in range(n):
        obs.append(SparsePauliOp.from_list([( "I"*i + "Z" + "I"*(n-i-1), 1.0 )]))
    # pesos 2..k_max
    for k in range(2, k_max+1):
        for idx in combinations(range(n), k):
            p = ["I"]*n
            for j in idx: p[j]="Z"
            obs.append(SparsePauliOp.from_list([("".join(p), 1.0)]))
    return obs

# -------- 5) vetor de features (multi-Z opcional) --------
def quantum_feature_vector(x_row, rho_ff, pairs, tau, m, A_of_s, B_of_s,
                           order=2, reps=12, k_max=2,
                           backend=None, estimator=None):
    n = len(x_row)
    h_vec = np.asarray(x_row, float)                           # h_i = x_i  (paper sem s_vec)
    J_dict = {(i,j): float(rho_ff[i,j]) for (i,j) in pairs}    # J_ij = rho_ij

    qc = build_quench_circuit(h_vec, J_dict, tau, m, A_of_s, B_of_s,
                              order=order, reps=reps)
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend, seed_transpiler=42)
    qc_t = pm.run(qc)

    obs_list = Z_observables(n, k_max=k_max)
    obs_isa  = [obs.apply_layout(qc_t.layout) for obs in obs_list]

    job = estimator.run([(qc_t, obs_isa)])
    expvals = job.result()[0].data.evs
    return np.asarray(expvals, dtype=np.float64)

def q_feature_matrix(X, rho_ff, pairs, tau, m, A_of_s, B_of_s,
                     order=2, reps=12, k_max=2,
                     backend=None, estimator=None, desc="qmap"):
    from tqdm import tqdm
    feats = []
    for row in tqdm(X, desc=desc, leave=True):
        feats.append(
            quantum_feature_vector(row, rho_ff, pairs, tau, m, A_of_s, B_of_s,
                                   order=order, reps=reps, k_max=k_max,
                                   backend=backend, estimator=estimator)
        )
    return np.vstack(feats)


# Definindo o modelo de estratificação e cross-validation
qc_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Armazenar as métricas
qc_metrics = {
    'AUC': [],
    'F1-Score Overall': [],
    'Balanced Accuracy': [],
    'Precision Class 0': [],
    'Precision Class 1': [],
    'Recall Class 0': [],
    'Recall Class 1': []
}


# Rodando o circuito quântico como feature map para a StratifiedKFold
rho_thr = 0
gamma_q = 1.0 
out_dir = Path("out_csv")
out_dir.mkdir(parents=True, exist_ok=True)
for fold, (train_id, test_id) in enumerate(qc_skf.split(X, y)):
    print(f'Fold: {fold}')

    # dados do fold
    X_tr_raw, X_te_raw = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]

    # preprocess
    X_train, X_test = preprocess_train_test(X_tr_raw, X_te_raw)

    # correlação entre features clássicas (apenas do treino)
    rho_ff = np.corrcoef(X_train, rowvar=False)
    np.fill_diagonal(rho_ff, 0)

    n_class = X_train.shape[1]
    pairs = [(i, j) for i in range(n_class) for j in range(i+1, n_class)
            if abs(rho_ff[i, j]) >= rho_thr]

    # features quânticas
    Xq_train = q_feature_matrix(
        X_train, rho_ff, pairs, tau, m, A_of_s, B_of_s,
        order=order, reps=reps, k_max=k_max,
        backend=backend, estimator=estimator, desc="  train"
    )
    Xq_test = q_feature_matrix(
        X_test, rho_ff, pairs, tau, m, A_of_s, B_of_s,
        order=order, reps=reps, k_max=k_max,
        backend=backend, estimator=estimator, desc="  test"
    )

    # concatenação: [clássico | quântico]
    X_aug_train = np.hstack([X_train, gamma_q * Xq_train]).astype(np.float32)
    X_aug_test  = np.hstack([X_test , gamma_q * Xq_test ]).astype(np.float32)

    # modelo
    qc_model = GradientBoostingClassifier(random_state=42)
    qc_model.fit(X_aug_train, y_train)

    # métricas
    y_pred  = qc_model.predict(X_aug_test)
    y_proba = qc_model.predict_proba(X_aug_test)[:, 1]

    qc_metrics['AUC'].append(roc_auc_score(y_test, y_proba))
    qc_metrics['F1-Score Overall'].append(f1_score(y_test, y_pred))
    qc_metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_test, y_pred))
    qc_metrics['Precision Class 0'].append(precision_score(y_test, y_pred, pos_label=0))
    qc_metrics['Precision Class 1'].append(precision_score(y_test, y_pred, pos_label=1))
    qc_metrics['Recall Class 0'].append(recall_score(y_test, y_pred, pos_label=0))
    qc_metrics['Recall Class 1'].append(recall_score(y_test, y_pred, pos_label=1))

    # ---------- salvamento por fold ----------
    # nomes de colunas
    n_q = Xq_train.shape[1]
    cols_class = [f"class_{i}" for i in range(n_class)]
    cols_quant = [f"qf_{i}"    for i in range(n_q)]
    all_cols   = cols_class + cols_quant

    # DataFrames com marcadores
    df_train = pd.DataFrame(X_aug_train, columns=all_cols).astype(np.float32)
    df_train["y"] = y_train.astype(int)
    df_train["set"] = "train"
    df_train["fold"] = fold
    df_train["row_id"] = train_id  # preserva índice original

    df_test = pd.DataFrame(X_aug_test, columns=all_cols).astype(np.float32)
    df_test["y"] = y_test.astype(int)
    df_test["set"] = "test"
    df_test["fold"] = fold
    df_test["row_id"] = test_id

    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # arquivos
    csv_features_path = out_dir / f"features_y_fold{fold}.csv"
    df_all.to_csv(csv_features_path, index=False)
    print(f"Salvo: {csv_features_path}")

    # matriz de correlação (clássicas)
    df_rho = pd.DataFrame(rho_ff, index=cols_class, columns=cols_class)
    csv_rho_path = out_dir / f"rho_ff_fold{fold}.csv"
    df_rho.to_csv(csv_rho_path, float_format="%.8g")
    print(f"Salvo: {csv_rho_path}")

    # binário sem perda
    npy_rho_path = out_dir / f"rho_ff_fold{fold}.npy"
    np.save(npy_rho_path, rho_ff)
    print(f"Salvo: {npy_rho_path}")

    # índices do split (útil)
    np.save(out_dir / f"train_idx_fold{fold}.npy", train_id)
    np.save(out_dir / f"test_idx_fold{fold}.npy", test_id)

    # metadados do fold
    meta = {
        "fold": fold,
        "n_class": int(n_class),
        "n_q": int(n_q),
        "gamma_q": float(gamma_q),
        "tau": float(tau),
        "m": int(m),
        "reps": int(reps),
        "k_max": int(k_max),
        "rho_thr": float(rho_thr),
        "pairs_count": int(len(pairs)),
    }
    pd.Series(meta).to_json(out_dir / f"meta_fold{fold}.json", indent=2)




COLS = [
    "AUC",
    "F1-Score Overall",
    "Balanced Accuracy",
    "Precision Class 0",
    "Precision Class 1",
    "Recall Class 0",
    "Recall Class 1",
]

def metrics_to_df(metrics_dict, qc_flag, A=None, B=None, kmax=None, m=None, tau=None, **extra):
    # sanity-check: todas as listas com o mesmo tamanho
    n_by_col = {c: len(metrics_dict[c]) for c in COLS}
    if len(set(n_by_col.values())) != 1:
        raise ValueError(f"Tamanhos diferentes nas listas: {n_by_col}")

    df = pd.DataFrame({c: metrics_dict[c] for c in COLS})
    df["qc"] = qc_flag            # "QC" ou "Classical"
    df["A"] = A
    df["B"] = B
    df["kmax"] = kmax
    df["m"] = m
    df["tau"] = tau               # ex.: 30 (ns). Para clássico, pode ser None/NaN.
    for k, v in extra.items():    # permite adicionar split_id, rep_id etc.
        df[k] = v
    return df

A_val   = A_of_s          
B_val   = B_of_s          
kmax_val= k_max
m_val   = m
tau_val = tau         

# DataFrames
df_qc = metrics_to_df(qc_metrics, qc_flag="QC",
                      A=A_val, B=B_val, kmax=kmax_val, m=m_val, tau=tau_val)

df_cl = metrics_to_df(metrics, qc_flag="Classical",
                      A=None, B=None, kmax=None, m=None, tau=None)

# Concatenar e salvar
df_all = pd.concat([df_qc, df_cl], ignore_index=True)
df_all.to_csv("new_note_results_qc_vs_classical.csv", index=False)

print(f"Salvo em new_note_results_qc_vs_classical_tau{tau}_{m}.csv")

