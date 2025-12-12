import streamlit as st
import os
import matplotlib
import json
import io
import re
from contextlib import redirect_stdout
from operator import itemgetter
from dotenv import load_dotenv
import numpy as np # Added for practice mode

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
matplotlib.use("Agg")

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- 3. Chain Creation ---
def create_chains():
    """
    Creates and returns a dictionary containing two specialized chains:
    1. 'rag': The main RAG tutor for questions and plotting.
    2. 'explainer': A specialist chain for explaining code.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, convert_system_message_to_human=True)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Chain 1: The RAG Tutor (with Guardrails) ---
    rag_template = """
    Your primary function is to act as a JSON API. You MUST respond with a single, valid JSON object and nothing else.
    The JSON object must have two keys: "explanation" and "code".
    
    --- GUARDRAIL RULES (APPLY THESE FIRST) ---
    1.  **Analyze the user's QUESTION first.**
    2.  **If the QUESTION is about data science, statistics, machine learning, programming (Python/R), or a related technical topic, proceed to the INSTRUCTIONS FOR JSON CONTENT.**
    3.  **If the QUESTION is clearly outside of this domain (e.g., history, sports, geography, art), you MUST refuse to answer.** Your JSON response MUST have an "explanation" field containing only this message: "I apologize, but my knowledge is strictly limited to data science and related topics. I cannot answer questions about general knowledge." The "code" field must be empty. DO NOT use your general knowledge to answer.
    
    **Instructions for JSON content:**
    1.  The "explanation" value must be a clear, expert-level textual answer to the QUESTION.
    2.  If the QUESTION explicitly asks for a "plot", "graph", "chart", "visualization", or "diagram", you MUST generate complete, runnable Python code for it in the "code" value. The code must use Matplotlib and create a figure object named 'fig'.
    3.  If the QUESTION asks for a non-plotting code example, generate that code in the "code" value.
    4.  If the QUESTION is purely conceptual, the "code" value MUST be an empty string ("").
    
    CONTEXT:
    {context}
    
    CHAT HISTORY:
    {chat_history}
    
    QUESTION:
    {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # --- Chain 2: The Code Explainer ---
    code_explainer_template = """You are an expert Python code explainer.
    Your response MUST be a JSON object with a single key: "explanation".
    CODE: ```python\n{code_block}\n```
    EXECUTION OUTPUT: ```\n{code_output}\n```
    Explain the code's logic and what the final output means.
    """
    code_explainer_prompt = ChatPromptTemplate.from_template(code_explainer_template)
    code_explainer_chain = code_explainer_prompt | llm | StrOutputParser()

    return {"rag": rag_chain, "explainer": code_explainer_chain}

# --- 4. Helper Function for JSON Parsing ---
def find_and_parse_json(text: str):
    """Finds and parses the first valid JSON object in a string."""
    try:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_str = text[start_index:end_index]
            return json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return None

# ==================================
# --- 5. Main Chatbot UI Section ---
# ==================================
st.set_page_config(page_title="Data Science Tutor", layout="wide")
st.title("🎓 Data Science Study Tutor")
st.markdown("Ask a question, ask for a plot, or paste a block of Python code to have it explained!")

if "chains" not in st.session_state:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.chains = create_chains()
    st.success("Knowledge base ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, dict):
            if "explanation" in content and content["explanation"]:
                st.markdown("### 💡 Explanation")
                st.markdown(content["explanation"])
            if "code" in content and content["code"]:
                st.markdown("### 🐍 Generated Code")
                st.code(content["code"], language="python")
                st.divider()
            if "fig" in content:
                st.markdown("### 📊 Generated Plot")
                st.pyplot(content["fig"])
            if "code_block" in content:
                 st.markdown("### 🔬 Code Breakdown")
                 st.markdown(content.get("explanation", ""))
                 with st.expander("Show Executed Code and Output"):
                    st.code(content["code_block"], language="python")
                    st.text("Output:")
                    st.code(content["code_output"], language="text")
        else:
            st.markdown(content)

# React to new user input
if user_prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            is_code_block = bool(re.search(r"^\s*(import|def|for|while|if|#)", user_prompt.strip())) or len(user_prompt.strip().split('\n')) > 1

            if is_code_block:
                code_to_explain = user_prompt
                output_capture = io.StringIO()
                try:
                    with redirect_stdout(output_capture):
                        exec(code_to_explain)
                    code_output = output_capture.getvalue()
                except Exception as e:
                    code_output = f"An error occurred during execution: {e}"
                
                response_str = st.session_state.chains["explainer"].invoke({ "code_block": code_to_explain, "code_output": code_output })
                response_data = find_and_parse_json(response_str)

                if response_data and "explanation" in response_data:
                    explanation = response_data["explanation"]
                    st.markdown("### 🔬 Code Breakdown")
                    st.markdown(explanation)
                    with st.expander("Show Executed Code and Output"):
                        st.info("This is the code that was executed:")
                        st.code(code_to_explain, language="python")
                        st.text("Captured Output:")
                        st.code(code_output, language="text")
                    
                    st.session_state.messages.append({ "role": "assistant", "content": { "explanation": explanation, "code_block": code_to_explain, "code_output": code_output } })
                else:
                    st.error("I had trouble explaining that code. Here is the raw response:")
                    st.code(response_str, language="text")
                    st.session_state.messages.append({"role": "assistant", "content": response_str})

            else:
                history_string = ""
                for message in st.session_state.messages[-5:-1]:
                    content = message["content"]
                    if isinstance(content, str):
                        history_string += f"{message['role'].capitalize()}: {content}\n"
                    elif isinstance(content, dict) and "explanation" in content:
                        history_string += f"{message['role'].capitalize()}: {content['explanation']}\n"
                
                response_str = st.session_state.chains["rag"].invoke({ "question": user_prompt, "chat_history": history_string })
                response_data = find_and_parse_json(response_str)

                if response_data:
                    explanation = response_data.get("explanation", "")
                    generated_code = response_data.get("code", "")
                    response_content = {}
                    
                    if explanation:
                        st.markdown("### 💡 Explanation")
                        st.markdown(explanation)
                        response_content["explanation"] = explanation
                    
                    if generated_code:
                        st.divider()
                        if "fig" in generated_code or "plt.figure" in generated_code:
                            st.markdown("### 📊 Generated Plot")
                            try:
                                exec_globals = {}
                                exec(generated_code, exec_globals)
                                fig = exec_globals.get("fig")
                                if fig:
                                    st.pyplot(fig)
                                    response_content["fig"] = fig
                            except Exception as e:
                                st.error(f"An error occurred while generating the plot: {e}")
                        else:
                            st.markdown("### 🐍 Generated Code")
                            st.code(generated_code, language="python")
                            response_content["code"] = generated_code
                    
                    if response_content:
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                else:
                    st.error("I had trouble formatting my response. Here is the raw output:")
                    st.code(response_str, language="text")
                    st.session_state.messages.append({"role": "assistant", "content": response_str})

# --- START OF INTEGRATED PRACTICE MODE CODE ---
# =============================================
# 📚 PRACTICE MODE SECTION
# =============================================

def get_practice_models():
    """
    Builds (once per session) a retriever + LLM specifically for Practice Mode.
    """
    if "practice_retriever" not in st.session_state:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        st.session_state.practice_retriever = db.as_retriever(search_kwargs={"k": 5})

    if "practice_llm" not in st.session_state:
        st.session_state.practice_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, convert_system_message_to_human=True)

    return st.session_state.practice_retriever, st.session_state.practice_llm


def generate_practice_questions(topic, retriever, llm, n_questions=5):
    """
    Generates N practice questions on a given topic.
    """
    docs = retriever.get_relevant_documents(topic)
    context = "\n\n".join(d.page_content for d in docs[:5])

    prompt = f"""
You are an expert Data Science tutor. Based on the CONTEXT provided, generate {n_questions} practice questions about the topic: "{topic}".
- The questions should be general and self-contained.
- Do NOT reference "the text" or "the context".
- Start with easy questions and progress to harder ones.
- Output MUST be a numbered list.
CONTEXT:
\"\"\"{context}\"\"\"
Now generate the questions.
"""
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    questions = [q.split(".", 1)[1].strip() for q in text.split("\n") if q and q[0].isdigit()]
    return questions[:n_questions] if questions else [text.strip()]


def evaluate_practice_answer(question, student_answer, topic, retriever, llm):
    """
    Evaluates a student's answer and provides feedback.
    """
    docs = retriever.get_relevant_documents(topic + " " + question)
    context = "\n\n".join(d.page_content for d in docs[:5])

    prompt = f"""
You are a Data Science teaching assistant. Evaluate the STUDENT_ANSWER for the QUESTION based on the provided CONTEXT.
Your output MUST start with "Score: X/100" followed by 5-7 lines of friendly but rigorous feedback.
- DO NOT mention the context.
- Evaluate as if you are a professor who knows the subject.

CONTEXT: \"\"\"{context}\"\"\"
QUESTION: {question}
STUDENT_ANSWER: {student_answer}
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# --- Practice Mode UI (Streamlit) ---
st.divider()
st.subheader("📚 Practice Mode (self-assessment)")

# Initialize session state for practice mode
if "practice_topic" not in st.session_state: st.session_state.practice_topic = ""
if "practice_questions" not in st.session_state: st.session_state.practice_questions = []
if "practice_index" not in st.session_state: st.session_state.practice_index = 0
if "practice_feedback" not in st.session_state: st.session_state.practice_feedback = ""
if "practice_answer" not in st.session_state: st.session_state.practice_answer = ""

# Topic input
st.session_state.practice_topic = st.text_input(
    "Choose a topic you want to practice:",
    value=st.session_state.practice_topic,
)

col_gen, col_reset = st.columns([2, 1])

with col_gen:
    if st.button("Generate practice questions"):
        if st.session_state.practice_topic.strip():
            retriever, llm = get_practice_models()
            with st.spinner("Generating questions..."):
                st.session_state.practice_questions = generate_practice_questions(st.session_state.practice_topic, retriever, llm)
            st.session_state.practice_index = 0
            st.session_state.practice_feedback = ""
            st.session_state.practice_answer = ""
            st.success(f"Generated questions on: {st.session_state.practice_topic}")
        else:
            st.warning("Please enter a topic first.")

with col_reset:
    if st.button("Reset Practice Mode"):
        st.session_state.practice_topic = ""
        st.session_state.practice_questions = []
        # ... (reset other practice state variables) ...
        st.info("Practice Mode has been reset.")

# If questions exist, display the current one
if st.session_state.practice_questions:
    idx = st.session_state.practice_index
    current_question = st.session_state.practice_questions[idx]

    st.markdown(f"**Question {idx + 1}/{len(st.session_state.practice_questions)}:** {current_question}")
    
    st.session_state.practice_answer = st.text_area("Your answer:", value=st.session_state.practice_answer, key=f"practice_answer_{idx}")

    col_fb, col_next = st.columns([2, 1])
    with col_fb:
        if st.button("Get feedback"):
            if st.session_state.practice_answer.strip():
                retriever, llm = get_practice_models()
                with st.spinner("Tutor is evaluating..."):
                    feedback = evaluate_practice_answer(current_question, st.session_state.practice_answer, st.session_state.practice_topic, retriever, llm)
                st.session_state.practice_feedback = feedback
            else:
                st.warning("Please provide an answer.")

    with col_next:
        if st.button("Next question"):
            if idx < len(st.session_state.practice_questions) - 1:
                st.session_state.practice_index += 1
                st.session_state.practice_feedback = ""
                st.session_state.practice_answer = ""
                st.rerun()
            else:
                st.info("You've reached the last question.")

    if st.session_state.practice_feedback:
        st.markdown("### 🧠 Tutor feedback")
        st.markdown(st.session_state.practice_feedback)