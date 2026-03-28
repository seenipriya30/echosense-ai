# import chromadb
# from sentence_transformers import SentenceTransformer
# import pandas as pd

# # load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # connect vector database
# client = chromadb.Client()

# collection = client.get_or_create_collection(name="echosense_memory")

# def store_reflections(df):
#     reflections = df["Short reflection or description of how your day was?"].dropna()

#     for i, text in enumerate(reflections):
#         embedding = model.encode(text).tolist()

#         collection.add(
#             ids=[str(i)],
#             documents=[text],
#             embeddings=[embedding]
#         )

# def search_similar(query):

#     query_embedding = model.encode(query).tolist()

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=5
#     )

#     return results














# memory_engine.py
import chromadb
import hashlib
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="echosense_memory")


def store_reflections(df: pd.DataFrame):
    df_clean = df[["reflection", "mood", "stress_self", "timestamp"]].dropna(subset=["reflection"])
    added, skipped = 0, 0

    for _, row in df_clean.iterrows():
        text = str(row["reflection"])
        doc_id = hashlib.md5(text.encode()).hexdigest()

        try:
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                skipped += 1
                continue
        except Exception:
            pass

        embedding = model.encode(text).tolist()
        collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "mood":         float(row["mood"]) if pd.notna(row["mood"]) else 5.0,
                "stress_score": float(10 - row["mood"]) if pd.notna(row["mood"]) else 5.0,
                "timestamp":    str(row["timestamp"])
            }]
        )
        added += 1

    return {"added": added, "skipped": skipped}


def search_similar(query: str, n_results: int = 5) -> list:
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "text":       results["documents"][0][i],
            "mood":       results["metadatas"][0][i].get("mood"),
            "stress":     results["metadatas"][0][i].get("stress_score"),
            "timestamp":  results["metadatas"][0][i].get("timestamp"),
            "rank":       i + 1
        })
    return output


def get_collection_stats() -> dict:
    return {"total_stored": collection.count()}