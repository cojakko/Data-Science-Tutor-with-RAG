import json
import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics import ndcg_score

# ============================
# PATH ASSOLUTI DEL PROGETTO
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "evaluation", "ground_truth_dataset12.jsonl")

# ============================
# CARICAMENTO MODELLO ED EMBEDDINGS
# ============================
print("Loading embeddings and FAISS...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})  # TOP-K = 5

# ============================
# METRICHE
# ============================
def precision_at_k(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(retrieved) if retrieved else 0

def recall_at_k(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(relevant) if relevant else 0

def mrr_score(retrieved, relevant):
    for idx, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1 / idx
    return 0

# ============================
# EVALUATION
# ============================
def run_evaluation():
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            query = entry["query"]
            relevant_passages = entry["relevant_passages"]

            # Retrieve top-5
            docs = retriever.get_relevant_documents(query)
            retrieved_texts = [doc.page_content for doc in docs]

            # Compute metrics
            p = precision_at_k(retrieved_texts, relevant_passages)
            r = recall_at_k(retrieved_texts, relevant_passages)
            m = mrr_score(retrieved_texts, relevant_passages)

            # NDCG
            relevance_vector = np.array([1 if t in relevant_passages else 0 for t in retrieved_texts])
            ndcg = ndcg_score([relevance_vector], [np.arange(len(relevance_vector), 0, -1)]) if len(relevance_vector) > 0 else 0

            # Store
            precisions.append(p)
            recalls.append(r)
            mrrs.append(m)
            ndcgs.append(ndcg)

            print("\n======================")
            print(f"Query: {query}")
            print(f"Precision@5: {p:.3f}")
            print(f"Recall@5:    {r:.3f}")
            print(f"MRR:         {m:.3f}")
            print(f"NDCG:        {ndcg:.3f}")
            print("======================")

    # FINAL RESULTS
    print("\n===== FINAL METRICS =====")
    print(f"Average Precision@5: {np.mean(precisions):.3f}")
    print(f"Average Recall@5:    {np.mean(recalls):.3f}")
    print(f"Average MRR:         {np.mean(mrrs):.3f}")
    print(f"Average NDCG:        {np.mean(ndcgs):.3f}")

if __name__ == "__main__":
    run_evaluation()
