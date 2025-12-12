import json
import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Path assoluto della cartella in cui si trova questo file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path reale del vectorstore
DB_FAISS_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "vectorstore", "db_faiss")
)

print("FAISS PATH RISOLTO:", DB_FAISS_PATH)  # <-- debug
######


BASE_DIR1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_PATH = os.path.join(BASE_DIR1, "data", "queries.txt")

print("query_file RISOLTO:",QUERIES_PATH )  # <-- debug

#####
import os

BASE_DIR2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = os.path.join(BASE_DIR2, "data", "ground_truth_dataset1.jsonl")
##########

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embeddings + FAISS
print("Caricamento Embeddings e Vector Store...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})

print("Pronto.\n")


def ask_user_to_select_passages(query, docs):
    print("\n========================================")
    print(f"QUERY: {query}")
    print("========================================\n")

    for i, d in enumerate(docs):
        print(f"\n--- PASSAGE {i+1} ---")
        print(d.page_content[:700])  # mostra fino 700 caratteri
        print("-" * 40)

    print("\nScrivi i numeri dei passaggi rilevanti separati da virgola (es: 1,3,5)")
    print("Oppure premi invio per nessuno.")

    user_input = input("Rilevanti: ").strip()

    if not user_input:
        return []

    try:
        indices = [int(x) - 1 for x in user_input.split(",")]
        selected = [docs[i].page_content for i in indices if 0 <= i < len(docs)]
        return selected
    except:
        print("Input non valido. Nessun passaggio selezionato.")
        return []


def build_ground_truth():

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    output = []

    for query in queries:
        docs = retriever.get_relevant_documents(query)

        selected_passages = ask_user_to_select_passages(query, docs)

        entry = {
            "query": query,
            "relevant_passages": selected_passages
        }
        output.append(entry)

        print("\n✓ Salvato!\n")

    # Save jsonl
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n======================================")
    print("Dataset Ground Truth Creato!")
    print(f"File salvato in: {OUTPUT_PATH}")
    print("======================================\n")


if __name__ == "__main__":
    build_ground_truth()
