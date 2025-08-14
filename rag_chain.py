from typing import Tuple
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# rag_chain.py (simple precheck)
DISALLOWED = ("cara membuat bom", "bahan letupan", "kebencian", "self-harm")

def is_allowed(question: str) -> bool:
    ql = question.lower()
    return not any(term in ql for term in DISALLOWED)


def load_retriever(persist_dir="index"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
        async_client=False
    )
    vs = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 4})

SYSTEM_PROMPT = """You are a helpful RAG assistant.
- Answer strictly based on the provided context.
- If the answer is not in context, say you don't know.
- Reply in the SAME language as the user's question.
- Cite brief sources at the end as [S1], [S2], ... using metadata if present.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

def format_docs(docs):
    # Join content and attach minimal citation markers.
    out = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "doc")
        page = meta.get("page", "")
        label = f"[S{i} {src}{' p.'+str(page) if page!='' else ''}]"
        out.append(d.page_content + f"\n{label}")
    return "\n\n".join(out)

def build_chain() -> Tuple:
    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
