import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)

DATA_PATH = os.path.join(project_root, 'data')
DB_FAISS_PATH = os.path.join(project_root, 'vectorstore', 'db_faiss')
def load_documents(data_path):
    """
    Loads documents from the specified data path.
    Supports PDF, Markdown (.md, .txt), and Python (.py) files.
    """
    documents = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(('.md', '.txt', '.py')): 
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    """
    Splits the documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings():
    """
    Creates embeddings using a pre-trained model from Hugging Face.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

def create_vector_store(texts, embeddings):
    """
    Creates a FAISS vector store from the text chunks and embeddings.
    """
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def main():
    """
    Main function to run the data ingestion process.
    """
    print("Starting data ingestion process...")

    all_documents = []
    for root, dirs, files in os.walk(DATA_PATH):
        print(f"Loading documents from: {root}")
        all_documents.extend(load_documents(root))

    if not all_documents:
        print("No documents found. Please check your data directory.")
        return

    print(f"Loaded {len(all_documents)} documents.")

    texts = split_documents(all_documents)
    print(f"Split documents into {len(texts)} chunks.")

    embeddings = create_embeddings()
    print("Embeddings model loaded.")

    print("Creating and saving the vector store...")
    create_vector_store(texts, embeddings)
    print(f"Vector store created and saved at: {DB_FAISS_PATH}")
    print("Ingestion process complete!")

if __name__ == "__main__":
    main()