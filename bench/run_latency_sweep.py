
import numpy as np
from core.graph import HNSWGraph
from baseline.brute_force import brute_force_knn
from bench.dataset import generate_random_dataset
from bench.latency import measure_latency

dataset = generate_random_dataset(n=50000, dim=16, seed=42)

g = HNSWGraph(M=8, ef_construction=50)
for i, vec in enumerate(dataset):
    g.insert(i, vec)

queries = dataset[:20]

# time brute-force
brute_latency = measure_latency(
    lambda q: brute_force_knn(q, dataset, 10),
    queries
)
print(f"brute-force avg latency: {brute_latency*1000:.3f} ms")

# time HNSW at a few different ef_search values
for ef in [5, 10, 20, 50, 100]:
    hnsw_latency = measure_latency(
        lambda q: g.search(q, 10, ef),
        queries
    )
    print(f"HNSW ef_search={ef}: avg latency = {hnsw_latency*1000:.3f} ms")