import numpy as np
from core.graph import HNSWGraph
from baseline.brute_force import brute_force_knn
from bench.dataset import generate_random_dataset
from bench.recall import recall_at_k, evaluate_recall

# generate a modest dataset
dataset = generate_random_dataset(n=50000, dim=16, seed=42)

# build the graph
g = HNSWGraph(M=8, ef_construction=50)
for i, vec in enumerate(dataset):
    g.insert(i, vec)

# pick some queries (just reuse a few dataset points for now)
queries = dataset[:20]

# try a few different ef_search values and watch recall change
for ef in [5, 10, 20, 50, 100]:
    r = evaluate_recall(g, dataset, queries, k=10, ef_search=ef)
    print(f"ef_search={ef}: recall@10 = {r:.3f}")