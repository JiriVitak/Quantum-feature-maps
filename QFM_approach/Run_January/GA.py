import numpy as np
import random
from typing import List, Tuple, Dict, Optional

Edge = Tuple[int, int]
# fitness é como a função de custo
def fitness_perm(perm: List[int],
                 edges: List[Edge],
                 J: np.ndarray,
                 edge_weight: Optional[Dict[Edge, float]] = None) -> float:
    s = 0.0
    for (i, j) in edges:
        w = 1.0 if edge_weight is None else edge_weight.get((i, j), edge_weight.get((j, i), 1.0))
        s += w * J[perm[i], perm[j]]
    return float(s)

def tournament_select(pop: List[List[int]], fit: List[float], k: int) -> List[int]:
    idxs = random.sample(range(len(pop)), k)
    best = max(idxs, key=lambda t: fit[t])
    return pop[best]

def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Order Crossover (OX) para permutações."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    fill = [x for x in p2 if x not in child]
    ptr = 0
    for i in list(range(0, a)) + list(range(b, n)):
        child[i] = fill[ptr]
        ptr += 1
    return child

def mutate_swap(p: List[int], pmut: float) -> None:
    if random.random() < pmut:
        i, j = random.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]

def mutate_scramble(p: List[int], pmut: float) -> None:
    """Mutação um pouco mais forte: embaralha um subsegmento."""
    if random.random() < pmut:
        n = len(p)
        a, b = sorted(random.sample(range(n), 2))
        seg = p[a:b]
        random.shuffle(seg)
        p[a:b] = seg

def ga_assignment(J: np.ndarray,
                            edges: List[Edge],
                            edge_weight: Optional[Dict[Edge, float]] = None,
                            pop_size: int = 80,
                            ngen: int = 300,
                            k_tourn: int = 3,
                            elite_k: int = 5,
                            pmut_init: float = 0.15,
                            pmut_min: float = 0.05,
                            pmut_max: float = 0.40,
                            patience: int = 25,
                            seed: int = 0):
    """
    GA para permutação (feature->qubit lógico), com:
    - top-K elitismo
    - 2 filhos por crossover
    - mutação adaptativa se estagnar
    - injeção de diversidade leve se necessário
    """
    random.seed(seed)
    np.random.seed(seed)

    n = J.shape[0]
    assert n > 1
    assert elite_k < pop_size

    # População inicial: permutações aleatórias
    pop = [random.sample(range(n), n) for _ in range(pop_size)]

    pmut = pmut_init
    best_fit = -1e18
    best_perm = None
    no_improve = 0

    for g in range(ngen):
        fit = [fitness_perm(p, edges, J, edge_weight) for p in pop]
        order = np.argsort(fit)[::-1]  # desc
        fit_sorted = [fit[i] for i in order]
        pop_sorted = [pop[i] for i in order]

        # Atualiza melhor global
        if fit_sorted[0] > best_fit + 1e-12:
            best_fit = fit_sorted[0]
            best_perm = pop_sorted[0][:]
            no_improve = 0
            # se melhorou, pode reduzir um pouco a mutação (explorar refinamento)
            pmut = max(pmut_min, pmut * 0.98)
        else:
            no_improve += 1

        # Se estagnou: aumenta mutação (exploração)
        if no_improve >= patience:
            pmut = min(pmut_max, pmut * 1.25)
            no_improve = 0

        # Top-K elitismo (cópia!)
        new_pop = [pop_sorted[i][:] for i in range(elite_k)]

        # Gera filhos até encher a população
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fit, k_tourn)
            p2 = tournament_select(pop, fit, k_tourn)

            # 2 filhos por crossover (troca papéis dos pais)
            c1 = ox_crossover(p1, p2)
            c2 = ox_crossover(p2, p1)

            # Mutação: misture swap + scramble (scramble com prob menor)
            mutate_swap(c1, pmut)
            mutate_swap(c2, pmut)
            mutate_scramble(c1, pmut * 0.25)
            mutate_scramble(c2, pmut * 0.25)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        # (Opcional) pequena injeção de diversidade a cada X gerações
        # substitui alguns piores por aleatórios
        if (g + 1) % 80 == 0:
            r = max(1, pop_size // 20)  # 5%
            for t in range(r):
                new_pop[-1 - t] = random.sample(range(n), n)

        pop = new_pop

    return best_perm, best_fit

def pick_connected_subset_greedy(n_target, coupling_edges, edge_cost):
    """
    coupling_edges: lista de arestas físicas (u,v) (não-direcionais)
    edge_cost[(u,v)]: custo (ex.: erro 2Q); quanto menor melhor


    Resumo: “entre os nós que posso adicionar sem quebrar conectividade, 
    adicione o que tem a conexão mais barata com o conjunto atual.” - barata significa menor erro
    """
    # monta adjacência
    adj = {}
    for u, v in coupling_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # escolhe seed: qubit com melhor "custo médio" nas arestas incidentes
    nodes = list(adj.keys())
    def node_score(u):
        cs = []
        for v in adj[u]:
            c = edge_cost.get((u,v), edge_cost.get((v,u), 1.0))
            cs.append(c)
        return np.mean(cs) if cs else 1e9
    seed = min(nodes, key=node_score)

    chosen = {seed}
    frontier = set(adj[seed])

    while len(chosen) < n_target:
        # escolhe o melhor candidato na fronteira
        best_cand = None
        best_score = 1e18

        for cand in list(frontier):
            # score do candidato = melhor aresta ligando cand ao conjunto escolhido
            scores = []
            for u in chosen:
                if cand in adj.get(u, []):
                    c = edge_cost.get((u,cand), edge_cost.get((cand,u), 1.0))
                    scores.append(c)
            if scores:
                s = min(scores)
                if s < best_score:
                    best_score = s
                    best_cand = cand

        if best_cand is None:
            raise RuntimeError("Não consegui crescer um subgrafo conectado com esse n_target.")

        # Adicionar o melhor candidato e atualizar a fronteira
        chosen.add(best_cand)
        frontier.remove(best_cand)
        for nb in adj[best_cand]:
            if nb not in chosen:
                frontier.add(nb)

    return sorted(chosen)
