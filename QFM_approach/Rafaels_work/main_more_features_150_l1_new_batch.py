import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import comb

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
import shap

from itertools import combinations
from collections import Counter

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator

import config
import GA
from qfm_lib import *



config = config.Config()

SEED = 42
np.random.seed(SEED)
num_features = 25

# =============================================================================
# 1) DADOS E PRÉ-PROCESSAMENTO BÁSICO
# =============================================================================
# df_full = pd.read_csv("./data/data.csv")     # Toxicity
# df_full = pd.read_csv("./data/framingham_clear_100.csv")
# df_full = pd.read_csv("./data/diabetes.csv")
# df_full = pd.read_csv("./data/santander_costumer_transaction_100.csv")
# df_full = pd.read_csv("./data/dataset_covid_clean_100.csv")
# df_full = pd.read_csv("./data/MI_afib_preprocessed_median_imputed_scaled_25f_100.csv")
df_full = pd.read_csv("./data/MI_afib_preprocessed_median_imputed_scaled.csv")
# df_full = pd.read_csv("./data/MI_afib_preprocessed_median_imputed_scaled_84.csv")

# df_full["Class"] = df_full["Class"].map({"NonToxic": 1, "Toxic": 0})

# Toxicity and diabetes and covid and MI
X = df_full.iloc[:, :-1].values
y = df_full.iloc[:, -1].values

# Santander
# X = df_full.iloc[:, 1:].values
# y = df_full.iloc[:, 0].values


def top_n_mi_features(X, y, n_features):
    mi = mutual_info_classif(
        X, y, n_neighbors=5, discrete_features=False, random_state=SEED
    )
    top_indices = np.argsort(mi)[-n_features:][::-1]
    return top_indices.tolist()


col_sel = top_n_mi_features(X, y, num_features)
print("Colunas selecionadas: ", col_sel)

if len(col_sel) < num_features:
    num_features = len(col_sel)

# Mantemos duas cópias:
# - X_cl_all: para o pipeline clássico (será reescalado por fold)
# - X_q_all: para o mapa quântico global (escalonado uma vez só)
X = X[:, col_sel]
X_cl_all = X.copy()

sc_q_global = StandardScaler()
X_q_all = sc_q_global.fit_transform(X_cl_all)   # usado em Jij + QFM  (h_i)

# Normalização dos dados (por fold) – usado no baseline clássico e nas features de entrada dos modelos
def preprosseging_data(X_train, X_test):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    return Xtr, Xte


# 2) BASELINE CLÁSSICO 5x5 
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
metrics = {
    "AUC": [],
    "F1 Macro": [],
    "Precision Macro": [],
    "Recall Macro": [],
    "Accuracy": [],
}

for fold, (train_id, test_id) in enumerate(rskf.split(X_cl_all, y)):
    print(f"Fold clássico nº {fold}")

    x_train = X_cl_all[train_id]
    x_test = X_cl_all[test_id]
    y_train = y[train_id]
    y_test = y[test_id]

    x_train, x_test = preprosseging_data(x_train, x_test)
    classifier = GradientBoostingClassifier(n_estimators=1000, random_state=SEED)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)[:, 1]

    metrics["AUC"].append(roc_auc_score(y_test, y_proba))
    metrics["F1 Macro"].append(f1_score(y_test, y_pred, average="macro"))
    metrics["Precision Macro"].append(
        precision_score(y_test, y_pred, average="macro")
    )
    metrics["Recall Macro"].append(
        recall_score(y_test, y_pred, average="macro")
    )
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))

median_metrics = {k: np.median(v) for k, v in metrics.items()}
mean_metrics = {k: np.mean(v) for k, v in metrics.items()}

print("Métricas medianas nos 5x5 folds (baseline clássico):")
for k, v in median_metrics.items():
    print(f"{k}: {v:.4f}")

print("Métricas médias nos 5x5 folds (baseline clássico):")
for k, v in mean_metrics.items():
    print(f"{k}: {v:.4f}")    

# 3) CONFIGURAÇÃO DO BACKEND E ESTIMATOR
SIMULATION = True
FAKEBACKEND = False

USE_FIXED_PERM = True
USE_FIXED_PHYS_NODES = True

m = 1
k_max = 2
q_ks_for_aug = [1, 2]  # quais ks entram no X_aug
tau = 0.005
l = 1  # ordem de aproximação dos termos CD
k_top = 50

service = QiskitRuntimeService(channel="ibm_cloud", token=config.QXToken)
real_backend = service.backend("ibm_fez")


if SIMULATION:
    if FAKEBACKEND:
        # “fakebackend”: Aer configurado como o ibm_fez (target/coupling/basis)
        backend = AerSimulator.from_backend(real_backend)
        backend.set_options(
            method="density_matrix",
            device="GPU",
            seed_simulator=SEED,
        )
    else:
        backend = AerSimulator(
            method="statevector",
            device="GPU",
        )
        # backend.set_options(seed_simulator=SEED, seed_transpiler=SEED)
        backend.options.seed_simulator = SEED
        backend.options.seed_transpiler = SEED        
else:
    backend = real_backend    
    backend.options.seed_transpiler = SEED

# Você pode trocar resilience_level para 1 depois para testar mitigação
estimator = Estimator(
    backend,
    options={
        "default_shots": 4096,
        "resilience_level": 0,
        # "resilience": {"measure_mitigation": True},
        # "twirling": {"enable_gates": True, "num_randomizations": 4},
    },
)


# 5) JIJ GLOBAL + GA (uma vez só, usando X_q_all)
rho_thr = 0.0
gamma_q = 1.0


# Cria a pasta onde seram salvos todos os arquivos referentes a esse run
base_folder = f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}"
os.makedirs(base_folder, exist_ok=True)

# MI em TODO o dataset (já escalonado)
n_b = X_q_all.shape[1]
MI_b = np.zeros((n_b, n_b))
for i in range(n_b):
    for j in range(i + 1, n_b):
        MI_b[i, j] = mutual_info_regression(
            X_q_all[:, [i]], X_q_all[:, j], random_state=SEED
        )[0]
        MI_b[j, i] = MI_b[i, j]

MI_norm_b = MI_b / (MI_b.max() + 1e-12)

J = MI_norm_b.copy()
J[J < rho_thr] = 0.0
np.fill_diagonal(J, 0.0)

if (not SIMULATION) or FAKEBACKEND:
    cm = real_backend.coupling_map
    coupling_edges = cm.get_edges()
    target = real_backend.target

    for gname in ["ecr", "cz", "cx"]:
        if gname in target.operation_names:
            twoq_gate = gname
            break
    else:
        raise RuntimeError(
            "Não achei ecr/cz/cx no backend.target; veja target.operation_names"
        )

    edge_cost = {}
    for (u, v) in coupling_edges:
        props = target[twoq_gate].get((u, v)) or target[twoq_gate].get((v, u))
        if props is not None and props.error is not None:
            edge_cost[(u, v)] = float(props.error)
        else:
            edge_cost[(u, v)] = 1.0

    # escolhe subgrafo físico de tamanho n_b de forma "gulosa" por erro menor
    if USE_FIXED_PHYS_NODES:
        with open("phys_nodes.json", "r", encoding="utf-8") as f:
            phys_nodes = json.load(f)
    else:
        phys_nodes = GA.pick_connected_subset_greedy(
            n_b, coupling_edges, edge_cost
        )

        with open(f"{base_folder}/phys_nodes.json", "w", encoding="utf-8") as f:
            json.dump(phys_nodes, f)

    phys_set = set(phys_nodes)
    phys_to_log = {p: i for i, p in enumerate(phys_nodes)}

    edges_log = []
    edge_weight = {}
    for (u, v) in coupling_edges:
        if u in phys_set and v in phys_set:
            i, j = phys_to_log[u], phys_to_log[v]
            edges_log.append((i, j))
            err = edge_cost.get((u, v), edge_cost.get((v, u), 1.0))
            edge_weight[(i, j)] = max(1e-6, 1.0 - err)

    # GA para mapear features -> qubits lógicos: 
    # perm: a feature clássica original perm[i] vai para o qubit lógico i -> 
    # serve para otimizar o embedding: qual feature clássica deve ir para qual nó (qubit) 
    # de forma a priorizar como vizinhas as que tiverem Jij maior e menor erro de aresta.

    if USE_FIXED_PERM:
        with open("perm.json", "r", encoding="utf-8") as f:
            perm = json.load(f)
    else:
        perm, best_fit = GA.ga_assignment(
            J,
            edges_log,
            edge_weight=edge_weight,
            pop_size=80,
            ngen=300,
            k_tourn=3,
            elite_k=5,
            seed=SEED,
        )

        with open(f"{base_folder}/perm.json", "w", encoding="utf-8") as f:
            json.dump(perm, f)

    print("Mapeamento final de qubits lógicos em físicos:", phys_nodes)
    print("Ordem das features no encoding:", perm)


    J_dict = {}
    for (i, j) in edges_log:
        Jij = J[perm[i], perm[j]]
        if Jij > 0:
            J_dict[(i, j)] = Jij

    print("Número de interações 2-local ativas:", len(J_dict))

    # 6) RODAR O MAPA QUÂNTICO UMA VEZ (TODAS AS 171 LINHAS)
    Xq_all_raw, pairs_2q = build_quantum_features_all(
        X_q_all,
        J_dict,
        perm,
        phys_nodes,
        tau,
        m,
        backend,
        estimator,
        simulation=SIMULATION,
    )
else:
    n = J.shape[0]
    J_dict = {}

    # só parte superior (i<j)
    for i in range(n):
        for j in range(i + 1, n):
            Jij = float(J[i, j])
            if Jij > 0.0:
                J_dict[(i, j)] = Jij

    print("Número de interações 2-local ativas:", len(J_dict))

    Xq_all_raw, pairs_2q = build_quantum_features_all_ideal(
        X_q_all,
        J_dict,
        tau,
        m,
        estimator,
        backend=None,
    )

print('Features quânticas geradas com sucesso!')

# SALVANDO AS FEATURES QUÂNTICAS GLOBAIS
n_samples, n_q_feats = Xq_all_raw.shape
n1 = n_b                      # 1-local = número de qubits/features clássicas
n2 = n_q_feats - n1           # 2-local = nº de pares (i,j) medidos

# nomes das features quânticas globais
q1_names = [f"q1_z_{i}" for i in range(n1)]
q2_names = [f"q2_z_{i}_{j}" for (i, j) in pairs_2q] if n2 > 0 else []
q_col_names = q1_names + q2_names

df_q_all = pd.DataFrame(Xq_all_raw, columns=q_col_names)
df_q_all["y"] = y            # guarda também o alvo

base_name = f"T_{tau}_m_{m}_kmax{k_max}_rescaled_{num_features}"

df_q_all.to_csv(
    f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}/qfeatures_all_{base_name}.csv",
    index=False
)

# opcional: salvar também em .npy para carregar mais rápido depois
np.save(
    f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}/qfeatures_all_{base_name}.npy",
    Xq_all_raw
)

print("Features quânticas globais salvas em:",
      f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}/qfeatures_all_{base_name}.csv")


# 7) CROSS-VALIDATION 5x5 USANDO FEATURES QUÂNTICAS PRÉ-COMPUTADAS
def update_metrics(metrics_dict, y_test, y_pred, y_proba):
    metrics_dict["AUC"].append(roc_auc_score(y_test, y_proba))
    metrics_dict["F1 Macro"].append(
        f1_score(y_test, y_pred, average="macro")
    )
    metrics_dict["Precision Macro"].append(
        precision_score(y_test, y_pred, average="macro")
    )
    metrics_dict["Recall Macro"].append(
        recall_score(y_test, y_pred, average="macro")
    )
    metrics_dict["Accuracy"].append(accuracy_score(y_test, y_pred))


def init_metrics_dict():
    return {
        "AUC": [],
        "F1 Macro": [],
        "Precision Macro": [],
        "Recall Macro": [],
        "Accuracy": [],
    }


qc_rskf = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=SEED
)

qc_k_metrics = {k: init_metrics_dict() for k in range(1, k_max + 1)}
shap50_metrics = {
    k: [] for k in ["AUC", "F1 Macro", "Precision Macro", "Recall Macro", "Accuracy"]
}
shap_rankings = []

n_samples, n_cl = X_cl_all.shape
n_q_total = Xq_all_raw.shape[1]  # já é n1 + n2

cl_names = [f"cl_{j}" for j in range(n_cl)]

# reaproveita os nomes já definidos quando salvamos
q_names_by_k_global = {1: q1_names}
if k_max >= 2 and n2 > 0:
    q_names_by_k_global[2] = q2_names


block_sizes = {1: n1}
if k_max >= 2:
    block_sizes[2] = n2

for fold, (train_id, test_id) in enumerate(qc_rskf.split(X_cl_all, y)):
    print(f"\nFold QUÂNTICO: {fold}")

    X_tr_raw, X_te_raw = X_cl_all[train_id], X_cl_all[test_id]
    y_train, y_test = y[train_id], y[test_id]

    # clássico: escalar por fold
    X_train, X_test = preprosseging_data(X_tr_raw, X_te_raw)

    # quântico: seleciona linhas e reescala por fold
    Xq_train_full_raw = Xq_all_raw[train_id]
    Xq_test_full_raw = Xq_all_raw[test_id]

    sc_q_fold = StandardScaler()
    Xq_train_full = sc_q_fold.fit_transform(Xq_train_full_raw)
    Xq_test_full = sc_q_fold.transform(Xq_test_full_raw)

    # separa por k (1-body, 2-body)
    Xq_blocks_train = {}
    Xq_blocks_test = {}

    if k_max >= 1:
        Xq_blocks_train[1] = Xq_train_full[:, :n1]
        Xq_blocks_test[1] = Xq_test_full[:, :n1]
    if k_max >= 2 and n2 > 0:
        Xq_blocks_train[2] = Xq_train_full[:, n1 : n1 + n2]
        Xq_blocks_test[2] = Xq_test_full[:, n1 : n1 + n2]

    # quais k entram no X_aug efetivamente: serve só para garantir que 
    # os ks do q_ks_for_aug também estão em Xq_blocks_train
    q_ks_for_aug_eff = [k for k in q_ks_for_aug if k in Xq_blocks_train]

    # seleciona apenas os dados referentes aos ks que devem entrar
    Xq_train_aug = np.hstack([Xq_blocks_train[k] for k in q_ks_for_aug_eff])
    Xq_test_aug = np.hstack([Xq_blocks_test[k] for k in q_ks_for_aug_eff])

    # ----------------- MODELOS PURO QUÂNTICO (por k) -----------------
    for k in range(1, k_max + 1):
        if k not in Xq_blocks_train:
            continue
        Xqk_train = Xq_blocks_train[k]
        Xqk_test = Xq_blocks_test[k]

        model_qk = GradientBoostingClassifier(
            n_estimators=1000, random_state=SEED
        )
        model_qk.fit(Xqk_train, y_train)

        y_pred_qk = model_qk.predict(Xqk_test)
        y_proba_qk = model_qk.predict_proba(Xqk_test)[:, 1]

        update_metrics(qc_k_metrics[k], y_test, y_pred_qk, y_proba_qk)

    # ----------------- X_aug = clássico + quântico -------------------
    X_aug_train = np.hstack([X_train, gamma_q * Xq_train_aug]).astype(
        np.float32
    )
    X_aug_test = np.hstack([X_test, gamma_q * Xq_test_aug]).astype(
        np.float32
    )

    # nomes das features em X_aug (sempre iguais)
    all_q_names_for_aug = []
    for k in q_ks_for_aug_eff:
        all_q_names_for_aug.extend(q_names_by_k_global[k])

    feat_names_aug = cl_names + all_q_names_for_aug
    assert len(feat_names_aug) == X_aug_train.shape[1]

    # ----------------- SHAP em X_aug ---------------------------------
    model_shap = GradientBoostingClassifier(
        n_estimators=1000, random_state=SEED
    )
    model_shap.fit(X_aug_train, y_train)

    explainer = shap.TreeExplainer(model_shap)
    shap_values = explainer.shap_values(X_aug_train)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # SHAP das features clássicas
    offset = 0
    shap_cl = mean_abs_shap[offset : offset + n_cl]
    offset += n_cl

    # SHAP das features quânticas por k
    shap_by_k = {}
    for k in q_ks_for_aug_eff:
        size_k = block_sizes[k]
        shap_by_k[k] = mean_abs_shap[offset : offset + size_k]
        offset += size_k

    # reordenar features quânticas por importância SHAP
    Xq_blocks_train_sorted = {}
    Xq_blocks_test_sorted = {}
    q_names_sorted_by_k = {}

    for k in q_ks_for_aug_eff:
        shap_k = shap_by_k[k]
        names_k = q_names_by_k_global[k]
        idx_sort = np.argsort(shap_k)[::-1]

        Xq_blocks_train_sorted[k] = Xq_blocks_train[k][:, idx_sort]
        Xq_blocks_test_sorted[k] = Xq_blocks_test[k][:, idx_sort]
        q_names_sorted_by_k[k] = [names_k[i] for i in idx_sort]

    Xq_train_sorted = np.hstack(
        [Xq_blocks_train_sorted[k] for k in q_ks_for_aug_eff]
    )
    Xq_test_sorted = np.hstack(
        [Xq_blocks_test_sorted[k] for k in q_ks_for_aug_eff]
    )

    q_names_sorted_all = []
    for k in q_ks_for_aug_eff:
        q_names_sorted_all.extend(q_names_sorted_by_k[k])

    df_q_train = pd.DataFrame(Xq_train_sorted, columns=q_names_sorted_all)
    df_q_train["y"] = y_train
    df_q_train["split"] = fold
    df_q_train["set"] = "train"

    df_q_test = pd.DataFrame(Xq_test_sorted, columns=q_names_sorted_all)
    df_q_test["y"] = y_test
    df_q_test["split"] = fold
    df_q_test["set"] = "test"

    os.makedirs(
        f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}",
        exist_ok=True,
    )
    df_q_train.to_csv(
        f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}/qfeatures_fold{fold}_train.csv",
        index=False,
    )
    df_q_test.to_csv(
        f"features_T_{tau}_m_{m}_k_max{k_max}_l_{l}_top{k_top}_rescaled_{num_features}/qfeatures_fold{fold}_test.csv",
        index=False,
    )

    shap_df_fold = pd.DataFrame(
        {
            "split": fold,
            "feature": feat_names_aug,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_rankings.append(shap_df_fold)

    # ----------------- SHAP50 ----------------------------------------
    idx_top = np.argsort(mean_abs_shap)[-k_top:]

    X_train_50 = X_aug_train[:, idx_top]
    X_test_50 = X_aug_test[:, idx_top]

    model_50 = GradientBoostingClassifier(
        n_estimators=1000, random_state=SEED
    )
    model_50.fit(X_train_50, y_train)

    y_pred_50 = model_50.predict(X_test_50)
    y_proba_50 = model_50.predict_proba(X_test_50)[:, 1]

    update_metrics(shap50_metrics, y_test, y_pred_50, y_proba_50)


# 8) AGREGAÇÃO DE MÉTRICAS + GRÁFICOS 
def median_metrics(metrics_dict):
    return {k: np.median(v) for k, v in metrics_dict.items()}


# MÉDIA--------------------------------------------------------
print('=============MÉDIA================')
for k in range(1, k_max + 1):
    print(f"\n=== k = {k} ===")
    for metric, values in qc_k_metrics[k].items():
        print(metric, np.mean(values), "±", np.std(values))

print(f"\n=== SHAP = {k_top} ===")
for metric, values in shap50_metrics.items():
    print(metric, np.mean(values), "±", np.std(values))    
# -------------------------------------------------------------

# MEDIANA------------------------------------------------------
print('=============MEDIANA================')
for k in range(1, k_max + 1):
    print(f"\n=== k = {k} ===")
    for metric, values in qc_k_metrics[k].items():
        print(metric, np.median(values), "±", np.std(values))

print(f"\n=== SHAP = {k_top} ===")
for metric, values in shap50_metrics.items():
    print(metric, np.median(values), "±", np.std(values))
# ------------------------------------------------------------


shap_rankings_all = pd.concat(shap_rankings, ignore_index=True)
shap_rankings_all.to_csv(
    f"shap_rankings_k{k_max}_l{l}_T_{tau}_m_{m}_l_{l}_top{k_top}_rescaled_{num_features}.csv",
    index=False,
)

COLS = ["AUC", "F1 Macro", "Precision Macro", "Recall Macro", "Accuracy"]


def metrics_to_df(
    metrics_dict,
    qc_flag,
    model_type,
    num_steps=None,
    h_scale=None,
    theta_j=None,
    **extra,
):
    df = pd.DataFrame({c: metrics_dict[c] for c in COLS})
    df["qc"] = qc_flag
    df["model_type"] = model_type
    df["num_steps"] = num_steps
    df["h_scale"] = h_scale
    df["theta_j"] = theta_j
    for k, v in extra.items():
        df[k] = v
    return df


df_cl = metrics_to_df(
    metrics, qc_flag="Classical", model_type="Classical_orig"
)

df_q_list = []
for k in range(1, k_max + 1):
    df_qk = metrics_to_df(
        qc_k_metrics[k], qc_flag="QC", model_type=f"Q_{k}body"
    )
    df_q_list.append(df_qk)

df_shap50 = metrics_to_df(
    shap50_metrics, qc_flag="QC+CL", model_type="SHAP50"
)

df_all = pd.concat([df_cl, *df_q_list, df_shap50], ignore_index=True)

df_all.to_csv(
    f"cd_terms_qc_vs_classical_DIGITAL_k{k_max}_l{l}_T_{tau}_m_{m}_l_{l}_top{k_top}_rescaled_{num_features}.csv",
    index=False,
)
print(
    f"Salvo em cd_terms_qc_vs_classical_DIGITAL_k{k_max}_l{l}_T_{tau}_m_{m}_l_{l}_top{k_top}_rescaled_{num_features}.csv"
)

# -------------- Gráfico 1 -------- ----
metrics_order = ["F1 Macro", "Precision Macro", "Recall Macro", "AUC", "Accuracy"]
model_order = ["Classical_orig"] + [f"Q_{k}body" for k in range(1, k_max + 1)] + [
    "SHAP50"
]

label_map = {
    "Classical_orig": "X Original",
    "SHAP50": "Concat SHAP50",
}
for k in range(1, k_max + 1):
    if k == 1:
        label = "Local mag."
    elif k == 2:
        label = "2-body corr."
    else:
        label = f"{k}-body corr."
    label_map[f"Q_{k}body"] = label

agg = df_all.groupby("model_type")[metrics_order].agg(["mean", "std"])

means = agg.xs("mean", axis=1, level=1).reindex(model_order)
stds = agg.xs("std", axis=1, level=1).reindex(model_order)

x = np.arange(len(metrics_order))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 5))

for i, model in enumerate(model_order):
    ax.bar(
        x + (i - (len(model_order) - 1) / 2) * width,
        means.loc[model],
        width,
        yerr=stds.loc[model],
        label=label_map[model],
        capsize=3,
        alpha=0.9,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    ["F1\nMACRO", "PRECISION", "RECALL", "ROC\nAUC", "ACCURACY"]
)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.legend(loc="upper left")

for gx in np.arange(len(metrics_order) - 1) + 0.5:
    ax.axvspan(gx - 0.02, gx + 0.02, color="lightgray", alpha=0.6)

plt.tight_layout()
plt.savefig(
    f"performance_of_quantum_features_k{k_max}_l{l}_T_{tau}_m_{m}_l_{l}_top{k_top}_rescaled_{num_features}.png"
)

# ----------------- Gráfico 2 -----------------
# ---- 1) Ranking global de importância ----
feat_imp = (
    shap_rankings_all.groupby("feature")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
)

top_k = int(k_top)
imp_top = feat_imp.head(top_k) if top_k > 0 else feat_imp.iloc[0:0]
tot_top = float(imp_top.sum()) if len(imp_top) > 0 else 0.0

# ---- 2) Montar grupos (label, série) ----
groups = []

# X Original (cl_)
ser_cl = imp_top[imp_top.index.astype(str).str.startswith("cl_")]
if len(ser_cl) > 0:
    groups.append(("X Original", ser_cl))

# qk_
for k in range(1, int(k_max) + 1):
    prefix = f"q{k}_"
    ser_k = imp_top[imp_top.index.astype(str).str.startswith(prefix)]
    if len(ser_k) == 0:
        continue

    if k == 1:
        lab = "Local mag."
    elif k == 2:
        lab = "2-body corr."
    else:
        lab = f"{k}-body corr."

    groups.append((lab, ser_k))

# Se não houver nada para plotar, sai com mensagem
if len(groups) == 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(
        0.5, 0.5,
        "Nenhuma feature 'cl_' ou 'qk_' encontrada no top_k.\n"
        "Verifique os prefixos no nome das features.",
        transform=ax.transAxes,
        ha="center", va="center", fontsize=11
    )
    ax.axis("off")
    # plt.show()
else:
    group_labels = [lab for lab, _ in groups]
    group_series = [ser for _, ser in groups]

    # ---- 3) Construir blocos de x e separadores ----
    x_blocks = []
    separators = []
    current = 0

    for ser in group_series:
        n = len(ser)
        xs = np.arange(n) + current
        x_blocks.append(xs)
        current = int(xs[-1]) + 2          # +2 cria um gap entre blocos
        separators.append(current - 1)     # linha no "meio" do gap

    # ---- 4) Plot ----
    fig, ax = plt.subplots(figsize=(10, 4))

    for xs, ser, label in zip(x_blocks, group_series, group_labels):
        ax.bar(xs, ser.values, label=label)

    # Separadores (exceto o último)
    for s in separators[:-1]:
        ax.axvline(s, color="grey", linestyle="--", linewidth=1)

    ax.set_ylabel("SHAP importance")

    # ---- 5) X ticks no centro de cada bloco ----
    xticks = [float(xs.mean()) for xs in x_blocks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(group_labels)

    # ---- 6) Anotações acima ----
    y_max = float(imp_top.max()) if len(imp_top) > 0 else 0.0
    y_text = y_max * 1.05 if y_max > 0 else 0.01  # fallback se tudo zero

    for xs, ser, label in zip(x_blocks, group_series, group_labels):
        n_vars = len(ser)
        pct = 100.0 * float(ser.sum()) / tot_top if tot_top > 0 else 0.0
        x_center = float(xs.mean())

        ax.text(
            x_center,
            y_text,
            f"{label}\n{n_vars} vars ({pct:.1f}%)",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="lightgray"),
        )

    # ---- 7) Ajustes finais ----
    # Dá uma folga em cima para as anotações
    ax.set_ylim(0, y_text * 1.15 if y_text > 0 else 1.0)

    # Opcional: legenda (se quiser)
    # ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()

plt.savefig(
    f"shap_importance_of_quantum_features_k{k_max}_l{l}_T_{tau}_m_{m}_l_{l}_top{k_top}_rescaled_{num_features}.png"
)
