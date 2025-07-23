# 🧠 Codebase Q\&A Bot

A local semantic search and question-answering tool for your codebase powered by LangChain, HuggingFace embeddings, Chroma vector store, and OpenRouter/Kimi (or Ollama).

---

## ✨ Features

* 🔍 **Semantic Search** over JavaScript/TypeScript/Markdown or Python files.
* 🤖 **LLM-Powered Answers** from relevant context using `moonshotai/kimi-k2:free` via OpenRouter.
* 🧱 **Embeddings** with local HuggingFace transformer models.
* 💾 **Vector Store** using [ChromaDB](https://www.trychroma.com/).
* 🗂️ Supports `Next.js` and `Python` repositories out-of-the-box.

---

## 🗂️ Project Structure

```
repo/                    # Your codebase directory (e.g., Next.js project)
portfolio_chroma/        # Persistent ChromaDB vector store
main.py                  # Builds the vector store and runs the QA loop
```

---

## ⚙️ Setup

### 1. Clone your target codebase

Place your Next.js or Python project under the `repo/` folder (or update `REPO_DIR` in the script).

### 2. Install dependencies

```bash
pip install langchain langchain-community langchain-huggingface langchain-openai chromadb tiktoken python-dotenv
```

> You may also need additional packages depending on your environment (e.g., `sentence-transformers`, `transformers`, `ollama` if using it locally).

### 3. Set your API key

Create a `.env` file in the root directory:

```
OPEN_ROUTER_API_KEY=your_openrouter_api_key
```

> Don't have one? Get a free key at [https://openrouter.ai](https://openrouter.ai)

---

## 🚀 Usage

### 1. Build the Vector Store

Run once to index the code:

```bash
python ingest.py
```

It will:

* Search for all relevant files in the `repo/` directory
* Split documents into overlapping chunks
* Generate embeddings
* Store them in ChromaDB

### 2. Ask Questions

After building the index, you can ask questions interactively:

```bash
python query.py
```

Example prompts:

```
What does the login handler do?
Where is the API route for submitting forms?
Explain the database connection setup.
```

---

## 🧠 LLM Options

The current setup uses:

* `moonshotai/kimi-k2:free` from OpenRouter for QA.

To use a local model like [Ollama](https://ollama.com/):

Uncomment and configure this section in `answer()`:

```python
# llm = ChatOllama(
#     model="gemma3:latest",
#     temperature=0,
# )
```

---

## 🛠️ Customize

* **Repo Location:** Set via `REPO_DIR = "repo"`
* **Supported Extensions:** Modify `list_nextjs_files()` for other file types.
* **Chunking Strategy:** Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` as needed.
* **Embeddings Model:** You can specify a local SentenceTransformer name:

  ```python
  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  ```

---

## 🧪 Example

```
Ask anything about the repo (empty to quit): Where is the auth middleware?
```

Will return a generated answer along with matching source files.

---

## 📄 License

MIT - Use freely, improve locally.

---
