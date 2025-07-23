from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
CHROMA_PATH = "./portfolio_chroma"
OLLAMA_MODEL = "gemma3:latest" 

def answer(question: str):
    
    # 1. Vector store
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Retriever – top-k most relevant snippets
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    # # 3. Local generative model (Ollama)
    # llm = ChatOllama(
    #     model=OLLAMA_MODEL,
    #     temperature=0,
    # )
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPEN_ROUTER_API_KEY,  # Replace with your OpenRouter API key
        model="moonshotai/kimi-k2:free",
        temperature=0,
        # max_tokens=4096,
    )

    # 4. Chain: retrieve → inject context → generate
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa.invoke({"query":question})
    print("\n--- ANSWER ---")
    print(result["result"])
    print("\n--- SOURCES ---")
    for doc in result["source_documents"]:
        print(
            doc.metadata["source"],
            doc.metadata.get("line", ""),
            "\n",
            # doc.page_content[:300],
            # "...\n",
        )


if __name__ == "__main__":
    while True:
        q = input("\nAsk anything about the repo (empty to quit): ").strip()
        if not q:
            break
        answer(q)
