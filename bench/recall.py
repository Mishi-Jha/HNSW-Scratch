from baseline.brute_force import brute_force_knn
def recall_at_k(hnsw_results, brute_results) -> float:
    brute_ids=set(r[0] for r in brute_results)
    hnsw_ids=set(s[0] for s in hnsw_results)
    intersection=brute_ids & hnsw_ids
    if len(brute_ids)==0:
        return 1
    return len(intersection)/len(brute_ids)

def evaluate_recall(graph, dataset, queries, k, ef_search) -> float:
    total=0
    for query in queries:
        hnsw=graph.search(query, k, ef_search)
        brute=brute_force_knn(query, dataset, k)
        recall=recall_at_k(hnsw,brute)
        total+=recall
    return total/len(queries)    

