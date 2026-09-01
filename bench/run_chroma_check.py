from bench.chroma_baseline import build_chroma_collection, query_chroma
from bench.dataset import generate_random_dataset
from baseline.brute_force import brute_force_knn

dataset = generate_random_dataset(n=50000, dim=16, seed=42)
collection = build_chroma_collection(dataset)

query = dataset[0]
result = query_chroma(collection, query, k=10)
print(result)

brute_result = brute_force_knn(query, dataset, k=10)
print("Brute-force:", brute_result)