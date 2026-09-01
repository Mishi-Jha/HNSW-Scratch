import os
import matplotlib.pyplot as plt

# Hardcoded from the n=50000, k=10 sweeps already run and recorded
ef_values = [5, 10, 20, 50, 100]
recall_values = [0.500, 0.940, 0.970, 1.000, 1.000]
hnsw_latency_ms = [2.376, 2.024, 2.668, 4.376, 5.821]

brute_force_latency_ms = 7.523
chroma_latency_ms = 6.785

os.makedirs("bench/results", exist_ok=True)

# --- Plot 1: Recall vs ef_search ---
plt.figure(figsize=(7, 5))
plt.plot(ef_values, recall_values, marker='o', color='#2563eb', label='HNSW recall@10')
plt.xlabel("ef_search")
plt.ylabel("Recall@10")
plt.title("Recall vs ef_search (n=50,000)")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("bench/results/recall_vs_ef.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 2: Latency vs ef_search, with reference lines ---
plt.figure(figsize=(7, 5))
plt.plot(ef_values, hnsw_latency_ms, marker='o', color='#2563eb', label='HNSW (from scratch)')
plt.axhline(brute_force_latency_ms, color='#dc2626', linestyle='--', label=f'Brute-force ({brute_force_latency_ms} ms)')
plt.axhline(chroma_latency_ms, color='#16a34a', linestyle='--', label=f'ChromaDB ({chroma_latency_ms} ms)')
plt.xlabel("ef_search")
plt.ylabel("Latency (ms)")
plt.title("Latency vs ef_search (n=50,000)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("bench/results/latency_vs_ef.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved plots to bench/results/")