import nest_asyncio
nest_asyncio.apply()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import os
import shutil
import sys
import tiktoken


def load_documents(path: str) -> list:
    """Load documents from a given path, supporting directories and various file types."""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    # If directory → process all files recursively
    if p.is_dir():
        docs = []
        for f in p.rglob("*"):
            if f.is_file():  # avoid re-calling on dirs
                docs.extend(load_documents(str(f)))
        return docs

    # File type detection
    suffix = p.suffix.lower()
    loader_map = {
        ".txt": lambda: TextLoader(str(p), encoding="utf-8").load(),
        ".md":  lambda: TextLoader(str(p), encoding="utf-8").load(),
        ".pdf": lambda: PyPDFLoader(str(p)).load(),
        ".docx": lambda: Docx2txtLoader(str(p)).load()
    }

    # Pick loader or fallback
    if suffix in loader_map:
        return loader_map[suffix]()
    return TextLoader(str(p), encoding="utf-8").load()


def token_len(text: str) -> int:
    """Token length function using OpenAI's cl100k_base tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def chunk_documents(docs, chunk_size=1500, chunk_overlap=0):
    """Chunk documents by tokens, prioritizing whole Q-A pairs for Malay FAQ."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # max tokens per chunk
        chunk_overlap=chunk_overlap, # token overlap
        separators=[
            "\nQuestion:",  # main FAQ cue
            "\n\n",         # paragraph break
            "\n",           # single line break
            " "             # space
        ],
        length_function=token_len,   # token-based splitting
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# Load .env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")


def build_faiss(chunks, persist_dir="index"):
    """
    Build a FAISS index from document chunks and save locally.
    Automatically deletes existing index folder to start fresh.
    """
    # Delete existing folder if it exists
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        async_client=False
    )

    # Build FAISS index
    vs = FAISS.from_documents(chunks, embeddings)

    # Save index locally
    vs.save_local(persist_dir)
    return persist_dir

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data"
    docs = load_documents(src)
    chunks = chunk_documents(docs)
    build_faiss(chunks)
    print(f"Indexed {len(chunks)} chunks → ./index")
