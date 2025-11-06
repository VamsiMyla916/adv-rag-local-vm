# app.py

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="TinyLlama RAG Bot",
    page_icon="🤖",
    layout="wide" # Use "wide" layout for sidebar
)

# --- API Endpoint ---
# This is set to run on your LOCAL machine.
API_URL = "https://vamsimyla-adv-rag-chatbot-vm.hf.space"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm a chatbot that can answer questions about the project I'm built with. What would you like to know?"}
    ]
if "guard_score" not in st.session_state:
    st.session_state.guard_score = 0.0
if "source_found" not in st.session_state:
    st.session_state.source_found = True # Start in a "passing" state

# --- Sidebar for Guard Score ---
with st.sidebar:
    st.header("Behind the Scenes")
    st.subheader("Relevance Guardrail")
    
    score = st.session_state.guard_score
    if not st.session_state.source_found:
        st.metric(label="Top Relevance Score", value=f"{score:.4f}", delta="FAILED (Below 0.5)", delta_color="inverse")
    else:
        st.metric(label="Top Relevance Score", value=f"{score:.4f}", delta="PASSED (>= 0.5)", delta_color="normal")
    
    st.write("This is the score from the `BAAI/bge-reranker-base` model.")
    st.write("If the score for the most relevant document is below **0.5**, the 'Relevance Guard' stops the LLM from answering to prevent making things up.")
    st.markdown("---")
    st.subheader("Project Architecture")
    st.text("Frontend: Streamlit")
    st.text("Backend: FastAPI (Docker)")
    st.text("Vector DB: FAISS")
    st.text("LLM: TinyLlama 1.1B (Transformers)")

# --- Main Chat Interface ---
st.title("🤖 TinyLlama RAG Chatbot")
st.caption("An advanced RAG app with re-ranking and relevance guardrails.")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Box ---
if prompt := st.chat_input("Ask a question about this RAG project..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            payload = {"text": prompt}
            response = requests.post(API_URL, json=payload, timeout=300) 
            
            if response.status_code == 200:
                api_response = response.json()
                
                answer = api_response.get("answer", "Error: No answer key found.")
                st.session_state.source_found = api_response.get("source_found", False)
                st.session_state.guard_score = api_response.get("guard_score", 0.0)
                
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun() # Update the sidebar
                
            else:
                error_detail = response.json().get("detail", "Unknown error")
                message_placeholder.error(f"Error from API: {response.status_code} - {error_detail}")
                st.session_state.messages.append({"role": "assistant", "content": f"API Error: {response.status_code}"})

        except requests.exceptions.RequestException as e:
            message_config = f"Error connecting to the API at {API_URL}. Is the backend running? (Error: {e})"
            message_placeholder.error(message_config)
            st.session_state.messages.append({"role": "assistant", "content": f"Connection Error: {e}"})