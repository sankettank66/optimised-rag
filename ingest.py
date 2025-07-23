import os, glob, tiktoken
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

REPO_DIR = "repo"
CHROMA_PATH = "./portfolio_chroma"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def list_python_files(root: str) -> List[str]:
    """All *.py files under root (including sub-dirs)."""
    print(f"Searching for Python files in: {root}")
    files = glob.glob(os.path.join(root, "**/*.py"), recursive=True)
    print(f"Found {len(files)} Python files.")
    return files


def list_nextjs_files(root: str) -> List[str]:
    """All relevant Next.js files under root (including sub-dirs)."""
    exts = ["js", "ts", "jsx", "tsx", "md"]
    patterns = [os.path.join(root, f"**/*.{ext}") for ext in exts]
    files = []
    print(f"Searching for Next.js files in: {root}")
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        print(f"Found {len(found)} files for pattern {pattern}")
        files.extend(found)
    print(f"Total files found: {len(files)}")
    return files


def build_vector_store():
    # 1. Load every relevant file as plain text
    print("Loading Next.js files as plain text...")
    docs = []
    for file in list_nextjs_files(REPO_DIR):
        print(f"Loading file: {file}")
        loader = TextLoader(file, encoding="utf-8")
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} documents.")

    # 2. Split into overlapping chunks
    print("Splitting documents into overlapping chunks...")
    encoding = tiktoken.get_encoding("cl100k_base")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda x: len(encoding.encode(x)),
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks.")

    # 3. Embed with local Sentence-Transformers model
    print("Embedding chunks with Sentence-Transformers model...")
    embeddings = HuggingFaceEmbeddings()

    # 4. Store in Chroma
    print("Storing embeddings in Chroma vector database...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Indexed {len(splits)} code chunks from {len(docs)} files.")


if __name__ == "__main__":
    print("Starting vector store build process...")
    build_vector_store()
