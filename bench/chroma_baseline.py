import numpy as np
from numpy import linalg as la
import chromadb
import tempfile
import os

def build_chroma_collection(dataset, collection_name="hnsw_benchmark"):
    client = chromadb.PersistentClient(path=os.path.join(tempfile.gettempdir(), "chroma_db"))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass    
    collection=client.get_or_create_collection(name=collection_name)
    batch_size=5000
    for i in range(0,len(dataset),batch_size):
        batch_dataset=dataset[i:i+batch_size]
        batch_ids=[str(j) for j in range(i,i+batch_size)]
        collection.add(
            embeddings=batch_dataset.tolist(),
            ids=batch_ids
        )    
    return collection

def query_chroma(collection, query, k) -> list[tuple[int, float]]:
    result=collection.query(query_embeddings=[query.tolist()],n_results=k)
    ids=result['ids'][0]
    distances=result['distances'][0]
    return list(zip([int(i) for i in ids],distances))

