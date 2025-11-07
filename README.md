Here is a complete, professional `README.md` file for your project, written in Markdown.

You can copy and paste this text directly into your `README.md` file on GitHub.

-----

# Advanced RAG Chatbot with Relevance Guardrails

This project is a sophisticated, decoupled (frontend/backend) Retrieval-Augmented Generation (RAG) application. It is designed to answer questions about a custom knowledge base (`data.txt`) by using the lightweight `TinyLlama-1.1B` model.

The key feature is an **advanced RAG pipeline** that includes a re-ranking step and a "Relevance Guardrail" to prevent LLM hallucination and ensure answers are sourced directly from the provided context.

The frontend is a Streamlit app that includes a "Behind the Scenes" sidebar to visualize the relevance score of the guardrail in real-time.

### Key Features

  * **Decoupled Architecture:** A robust FastAPI backend (running in a Docker container) handles all ML/LLM logic, while a clean Streamlit frontend handles the user interface.
  * **Advanced RAG Pipeline:** Instead of just "Retrieve -\> Generate," this app uses a more accurate "Retrieve -\> **Re-rank** -\> Generate" pipeline.
  * **Relevance Guardrail:** Uses a `BAAI/bge-reranker-base` model to score the relevance of retrieved documents. If the top-scoring document is below a **0.5 threshold**, the bot will state that it doesn't have enough information, preventing it from answering with irrelevant or hallucinated content.
  * **Interactive UI:** The Streamlit frontend provides a clean chat interface and a live-updating sidebar that displays the re-ranker's **Guard Score** for each query, making the guardrail visible.
  * **Optimized for CPU:** Uses the `TinyLlama-1.1B` model (a "normal" `transformers` model, not GGUF) which is small enough (\~5-6GB RAM) to run on a standard CPU machine (or a free Hugging Face Space).

-----

### How to Run (Local)

This project requires **two terminals** running at the same time.

#### Prerequisites

  * [Git](https://www.google.com/search?q=https://git-scm.com/downloads)
  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) (must be running)
  * [Python 3.12+](https://www.python.org/downloads/)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

#### Step 2: Run the Backend (Terminal 1)

1.  Navigate to the `backend` folder:

    ```bash
    cd backend
    ```

2.  **Build the Docker image.** This will take a few minutes as it installs Python 3.12 and all the required libraries.

    ```bash
    docker build -t rag-api-backend .
    ```

3.  **Run the Docker container.**

      * *(Note: You'll need a [Hugging Face User Access Token](https://huggingface.co/settings/tokens) with "read" permissions. This is required for the libraries to download the models without being rate-limited.)*

    <!-- end list -->

    ```bash
    docker run --rm -p 8000:8000 -e HF_TOKEN=YOUR_HF_TOKEN_HERE --name rag-api rag-api-backend
    ```

4.  Wait for the models to load. The backend is ready when you see this log:
    `INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

#### Step 3: Run the Frontend (Terminal 2)

1.  **Open a new, separate terminal.**

2.  Navigate to the `frontend` folder from the root of your project:

    ```bash
    cd ../frontend
    ```

    *(Or, from a new terminal: `cd C:\...YOUR_REPOSITORY_NAME\frontend`)*

3.  Install the Streamlit requirements:

    ```bash
    pip install -r requirements.txt
    ```

4.  **(Verification)** Make sure your `frontend/app.py` is pointing to your local server:
    `API_URL = "http://127.0.0.1:8000/query"`

5.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

Your browser will automatically open to `http://localhost:8501`. You can now chat with your locally hosted, advanced RAG application\!

-----

### Deployment to Hugging Face (Next Step)

The backend is ready to be deployed to the cloud.

1.  **Deploy Backend:** Upload all 4 files from the `backend` folder (`Dockerfile`, `requirements.txt`, `main.py`, `data.txt`) to a new **Docker SDK** Hugging Face Space.
2.  **Add Secret:** In the HF Space "Settings" tab, add your `HF_TOKEN` to the "Repository secrets".
      * *Troubleshooting:* If you get a `401 Unauthorized` error in your logs, your token is invalid. Delete it, generate a new one, and add it. If that still fails, deleting the secret entirely will allow anonymous downloading, as all models are public.
3.  **Connect Frontend:** Once the Space is "Running", copy its public URL (e.g., `https://your-space-name.hf.space`) and paste it into the `API_URL` variable in your **local** `frontend/app.py` file.
4.  **Relaunch** your local `streamlit run app.py`. It will now be powered by your cloud-hosted backend.

### Technologies Used

  * **Backend:** FastAPI, Docker, `transformers`, `torch`, `sentence-transformers`, `faiss-cpu`, `accelerate`
  * **Frontend:** Streamlit, `requests`
  * **LLM:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  * **Embedding:** `all-MiniLM-L6-v2`
  * **Re-Ranker:** `BAAI/bge-reranker-base`
