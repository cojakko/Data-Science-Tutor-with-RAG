## Project Overview

This project is an advanced, AI-powered study tutor for Data Science, built with a sophisticated, multi-featured architecture. The tutor is an interactive chatbot designed to help students learn complex topics, test their knowledge, get help with code, and visualize concepts.

The application is built using a **Retrieval-Augmented Generation (RAG)** architecture to ensure its answers are factually grounded in a curated knowledge base of academic notes and technical documentation. It leverages modern LLM techniques like agentic routing and structured JSON outputs to provide a robust and feature-rich user experience.


## Features

This tutor is equipped with a suite of advanced features to create a comprehensive and interactive learning environment:

*   **Conversational Q&A:** Ask questions about a wide range of data science topics. The tutor understands context and can answer follow-up questions.
*   **Proactive Code Generation:** Ask a conceptual question, and the tutor will proactively provide a clean, copy-pastable Python code example to demonstrate the concept.
*   **On-Demand Plotting:** Request a plot (e.g., "plot a normal distribution"), and the tutor will generate and display a Matplotlib graph directly in the chat.
*   **Interactive Code Explainer:** Paste a block of Python code, and the tutor will explain what it does step-by-step and show you its captured output.
*   **Practice Mode:** Enter a topic (e.g., "K-Means Clustering") to generate a set of practice questions and receive AI-powered feedback and a score on your answers.
*   **Honesty & Safety Guardrails:** The tutor is designed to recognize and refuse to answer questions outside its core domain of data science, preventing misinformation and knowledge contamination.

---

## Architecture & Technical Justification

The project is built on a modern RAG architecture using the LangChain framework, with several key design choices:

1.  **Knowledge Base & Ingestion:** A curated set of documents (`.pdf`, `.md`, `.py`) forms the knowledge base. An ingestion script (`src/ingest.py`) processes these documents, creates embeddings using a local `sentence-transformers` model, and stores them in a **FAISS** vector store for fast, local, and free retrieval.
2.  **Generative Model:** The application uses Google's powerful **`gemini-2.5-flash`** model via its API for all generation tasks, chosen for its strong reasoning capabilities and speed.
3.  **Frameworks:**
    *   **LangChain (LCEL):** The core logic is built with LangChain Expression Language (LCEL). This "LEGO brick" approach provides a robust, transparent, and highly customizable way to build complex chains, proving more stable than legacy chain methods.
    *   **Streamlit:** The entire user interface is built with Streamlit, chosen for its speed of development and native Python integration, allowing for a complex, multi-feature UI without requiring web development expertise.
4.  **Agentic Routing & Structured Output:**
    *   **Routing:** The application uses a simple but effective routing mechanism to distinguish between a general question and a code block, directing the user's input to the appropriate specialized chain (RAG Tutor vs. Code Explainer).
    *   **JSON Output:** The LLM is prompted to return structured JSON objects. This is a critical design choice that completely decouples the AI's "thinking" from the UI's "presentation," eliminating fragile string-parsing and enabling the reliable display of mixed content (text, code, and plots).

---

## Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

*   **Python 3.11** (This project is not compatible with Python 3.12+ due to dependency compilation issues).
*   A Google AI API Key.

### 2. Setup

First, clone the repository to your local machine and navigate into the project directory:
```bash
git clone [Your GitHub Repository URL]
cd [Your-Repository-Name]
```

Next, create and activate a Python 3.11 virtual environment:
```bash
# Create the virtual environment
python -m venv venv

# Activate it (on Windows)
venv\Scripts\activate
```

Then, install all the required dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a file named `.env` in the root of the project directory. Add your Google AI API key to this file:
```
GOOGLE_API_KEY="your_api_key_goes_here"
```

### 4. Build the Knowledge Base

Before running the app for the first time, you must run the ingestion script. This will build the local FAISS vector store from the documents located in the `/data` folder.
```bash
python src/ingest.py
```
This process is computationally intensive and may take several minutes to complete.

### 5. Run the Application

Once the ingestion is complete, you can launch the Streamlit application:
```bash
streamlit run src/main.py
```
The application should now be running and accessible in your web browser.

---

## Evaluation

The project includes a comprehensive, multi-faceted evaluation suite located in the `/evaluation` folder.

*   **Quantitative Retriever Analysis:**
    *   **`build_ground_truth.py`:** A script to manually create a labeled "answer key" dataset for testing retriever relevance.
    *   **`evaluate_retriever.py`:** A script that uses this ground truth to calculate key information retrieval metrics like **Precision@k, Recall@k, MRR, and NDCG**. This provides objective data on the performance of the RAG system's core.

*   **Qualitative & Feature-Based Analysis:**
    *   A detailed testing sheet was used to perform qualitative analysis on the generative performance of the tutor, covering **Accuracy, Clarity, Hallucination, Conversational Memory, and "Honesty"**.
    *   This analysis also included scenario-based testing for all advanced features, including the **Practice Mode's** question generation and feedback capabilities.

The results of these evaluations were used iteratively to improve the system's prompt engineering, leading to the final, robust version. Detailed findings can be found in the project's final report. 