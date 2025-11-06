import os
import json
import numpy as np
import faiss
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. SETTINGS & CONSTANTS ---
DOCUMENTS_FILE = "data.txt"
FAISS_INDEX_FILE = "faiss_index.index"
CHUNKS_FILE = "chunks.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- 2. LOAD ALL MODELS ON STARTUP ---
print("Loading models... This may take a moment.")
device = "cpu"
print(f"Using device: {device}")

# A) Load Embedding Model (for searching)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

# B) Load Re-ranker Model (for scoring)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# C) Load LLM (for generating answers)
print(f"Loading full-precision model: {LLM_MODEL_ID}...")
print("This will use ~5-6GB of RAM.")
hf_token = os.environ.get("HF_TOKEN")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=hf_token)

# --- This is the fix for the 'torch_dtype' warning ---
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    token=hf_token,
    dtype=torch.bfloat16, # Use 'dtype' instead of 'torch_dtype'
    device_map="auto"
)
# ---------------------------------------------------

print("All models loaded successfully.")

# --- 3. CREATE OR LOAD FAISS INDEX ---
faiss_index = None
text_chunks = []
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print(f"Loading existing FAISS index from {FAISS_INDEX_FILE}")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)
    print("Successfully loaded index and chunks.")
else:
    print(f"No index found. Creating new index from {DOCUMENTS_FILE}...")
    if not os.path.exists(DOCUMENTS_FILE):
        print(f"Error: {DOCUMENTS_FILE} not found! Please create this file and add your text data.")
        text_chunks = ["This is a demo chunk. Please add a data.txt file."]
    else:
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            data = f.read()
        text_chunks = [chunk.strip() for chunk in data.split('\n\n') if chunk.strip()]
        print(f"Created {len(text_chunks)} text chunks.")

    print("Embedding text chunks... This may take time.")
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    
    print("Creating FAISS index...")
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    faiss_index.add(embeddings.astype(np.float32))
    
    print(f"Saving index to {FAISS_INDEX_FILE} and chunks to {CHUNKS_FILE}")
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(text_chunks, f)
    
    print("Index creation complete.")

# --- 4. FASTAPI APP SETUP ---
app = FastAPI()

class Query(BaseModel):
    text: str
class Response(BaseModel):
    answer: str
    source_found: bool
    guard_score: float

# --- Health check endpoint (Fixes HF "Starting" loop) ---
@app.get("/")
async def health_check():
    """
    This endpoint is used by Hugging Face to check if the app is alive.
    """
    return {"status": "ok"}

@app.post("/query", response_model=Response)
async def handle_query(query: Query):
    if not query.text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # === STEP 1: RETRIEVE ===
    query_embedding = embedding_model.encode([query.text], convert_to_numpy=True)
    k_retrieve = 10
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), k_retrieve)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]

    # === STEP 2: RE-RANK ===
    query_pairs = [[query.text, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(query_pairs)
    scored_chunks = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)

    # === STEP 3: RELEVANCE GUARD ===
    RELEVANCE_THRESHOLD = 0.5
    top_score = scored_chunks[0][0]
    
    if top_score < RELEVANCE_THRESHOLD:
        return Response(
            answer="I'm sorry, I don't have enough relevant information to answer that question.",
            source_found=False,
            guard_score=float(top_score)
        )

    # === STEP 4: GENERATE ===
    top_k_generate = 3
    final_context = "\n\n".join([chunk for score, chunk in scored_chunks[:top_k_generate]])
    
    prompt = f"""<|system|>
You are a helpful assistant. Use the following context to answer the user's question. If the answer is not in the context, say 'I don't know'.<|end|>
<|user|>
Context:
{final_context}
Question:
{query.text}<|end|>
<|assistant|>
"""
    
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=250)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("<|assistant|>")[-1].strip()

    return Response(
        answer=answer,
        source_found=True,
        guard_score=float(top_score)
    )

print("--- Application startup complete. Uvicorn running. ---")