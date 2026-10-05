import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_vectorstore(pdf_bytes: bytes, filename: str, persist_dir: str):
    """Read a PDF, chunk it, embed it with Mistral, and store it in Chroma."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    text_segments = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            text_segments.append(text)

    os.unlink(tmp_path)

    combined_text = "\n".join(text_segments)
    if not combined_text.strip():
        raise ValueError("No extractable text found in this PDF (it may be scanned/image-only).")

    docs = [Document(page_content=combined_text, metadata={"source": filename})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings_model = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )
    return vectorstore, len(chunks)


def get_retriever(vectorstore, k: int):
    fetch_k = max(k * 2, k + 1)  # fetch_k must be >= k, both ints
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# ---------------------------------------------------------------------------
# Sidebar: upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    k_value = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=3)

    if uploaded_file is not None:
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Reading, chunking, and embedding your PDF..."):
                try:
                    persist_dir = os.path.join(
                        tempfile.gettempdir(), f"chroma_db_{uploaded_file.name}"
                    )
                    vectorstore, n_chunks = build_vectorstore(
                        uploaded_file.getvalue(), uploaded_file.name, persist_dir
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Indexed '{uploaded_file.name}' into {n_chunks} chunks.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

    if st.session_state.processed_file:
        st.info(f"Active document: **{st.session_state.processed_file}**")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Chat with your PDF")

if st.session_state.vectorstore is None:
    st.warning("Upload a PDF from the sidebar to get started.")
    st.stop()

llm = ChatMistralAI(model="mistral-small-2506")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = get_retriever(st.session_state.vectorstore, k_value)
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            final_prompt = PROMPT.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

            st.markdown(response.content)

            with st.expander("Retrieved context"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

    st.session_state.messages.append({"role": "assistant", "content": response.content})