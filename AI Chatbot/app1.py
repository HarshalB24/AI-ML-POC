# import os
# import tempfile
# import time
# from typing import List, Union
# import chromadb
# import ollama
# import streamlit as st
# from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import CrossEncoder
# from streamlit.runtime.uploaded_file_manager import UploadedFile

# # 🚀 SPEED FIX 1: Much shorter system prompt
# system_prompt = """
# Answer using only the context provided. Use clear paragraphs and bullets.
# """

# def process_document(uploaded_file: Union[UploadedFile, "streamlit.runtime.uploaded_file_manager.UploadedFile"]) -> List[Document]:
#     temp_file_path = None
#     try:
#         with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_file_path = temp_file.name

#         loader = PyMuPDFLoader(temp_file_path)
#         docs = loader.load()

#         # 🚀 SPEED FIX 2: Larger chunks = 50% fewer embeddings
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,  # Was 400
#             chunk_overlap=50,  # Was 100
#             separators=["\n\n", "\n", ".", "?", "!", " ", ""],
#         )
#         return text_splitter.split_documents(docs)

#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except PermissionError:
#                 time.sleep(0.1)
#                 if os.path.exists(temp_file_path):
#                     os.unlink(temp_file_path)

# def get_vector_collection() -> chromadb.Collection:
#     ollama_ef = OllamaEmbeddingFunction(
#         url="http://localhost:11434/api/embeddings",
#         model_name="nomic-embed-text:latest",
#     )
#     chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
#     return chroma_client.get_or_create_collection(
#         name="rag_app",
#         embedding_function=ollama_ef,
#         metadata={"hnsw:space": "cosine"},
#     )

# def add_to_vector_collection(all_splits: list[Document], file_name: str):
#     collection = get_vector_collection()
#     documents, metadatas, ids = [], [], []
    
#     for idx, split in enumerate(all_splits):
#         documents.append(split.page_content)
#         metadatas.append(split.metadata)
#         ids.append(f"{file_name}_{idx}")
    
#     collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
#     st.success("Data added to the vector store!")

# def query_collection(prompt: str, n_results: int = 4):  # 🚀 SPEED FIX 3: Reduced from 10
#     collection = get_vector_collection()
#     results = collection.query(query_texts=[prompt], n_results=n_results)
#     return results

# # 🚀 SPEED FIX 4: Skip cross-encoder entirely (biggest speedup)
# def get_top_context(results) -> str:
#     """Just take top 3 docs - no slow cross-encoder"""
#     if not results["documents"][0]:
#         return ""
#     return "\n\n".join(results["documents"][0][:3])

# def call_llm(context: str, prompt: str):
#     """🚀 SPEED FIX 5: Smaller model + no streaming"""
#     response = ollama.chat(
#         model="moondream:latest",  # ⚠️ MUST RUN: ollama pull llama3.2:1b
#         stream=False,  # Instant response
#         options={"num_predict": 400},  # Limit tokens
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context: {context[:2500]}\nQuestion: {prompt}"},
#         ],
#     )
#     return response['message']['content']

# # Your original UI - unchanged
# if __name__ == "__main__":
#     with st.sidebar:
#         st.set_page_config(page_title="RAG Question Answer")
#         uploaded_file = st.file_uploader(
#             "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
#         )
#         process = st.button("⚡️ Process")
#         if uploaded_file and process:
#             normalize_uploaded_file_name = uploaded_file.name.translate(
#                 str.maketrans({"-": "_", ".": "_", " ": "_"})
#             )
#             all_splits = process_document(uploaded_file)
#             add_to_vector_collection(all_splits, normalize_uploaded_file_name)

#     st.header("🗣️ Document Chatbot")
#     prompt = st.text_area("**Ask a question related to your document:**")
#     ask = st.button("🔥 Ask")

#     if ask and prompt:
#         with st.spinner("Answering..."):
#             results = query_collection(prompt)
#             context = get_top_context(results)
            
#             if context:
#                 answer = call_llm(context, prompt)
#                 st.markdown(answer)
                
#                 with st.expander("See retrieved documents"):
#                     st.write(results["documents"][0][:3])
#             else:
#                 st.warning("No relevant documents found.")


# import os
# from io import BytesIO
# from typing import List, Union

# import chromadb
# import ollama
# import streamlit as st
# from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import CrossEncoder

# # ----------------------------
# # Global Configs
# # ----------------------------
# system_prompt = """
# You are an AI assistant tasked with providing detailed answers based solely on the given context. 
# Format your response clearly and thoroughly.
# """

# # Load CrossEncoder globally (once)
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # ----------------------------
# # Document Processing
# # ----------------------------
# def process_document(uploaded_file: Union[BytesIO, "streamlit.runtime.uploaded_file_manager.UploadedFile"]) -> List[Document]:
#     """Load PDF from memory and split into chunks."""
#     pdf_bytes = uploaded_file.read()
#     loader = PyMuPDFLoader(BytesIO(pdf_bytes))
#     docs = loader.load()
    
#     # Use larger chunks to reduce embeddings
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", ".", "?", "!", " ", ""],
#     )
#     return splitter.split_documents(docs)

# # ----------------------------
# # Vector Storage
# # ----------------------------
# def get_vector_collection() -> chromadb.Collection:
#     """Return Chroma collection with Ollama embeddings."""
#     embedding_fn = OllamaEmbeddingFunction(
#         url="http://localhost:11434/api/embeddings",
#         model_name="nomic-embed-text:latest",
#     )
#     client = chromadb.PersistentClient(path="./demo-rag-chroma")
#     return client.get_or_create_collection(
#         name="rag_app",
#         embedding_function=embedding_fn,
#         metadata={"hnsw:space": "cosine"},
#     )

# def add_to_vector_collection(all_splits: List[Document], file_name: str):
#     """Batch upload document splits to Chroma."""
#     collection = get_vector_collection()
#     docs = [d.page_content for d in all_splits]
#     metadatas = [d.metadata for d in all_splits]
#     ids = [f"{file_name}_{i}" for i in range(len(all_splits))]
#     collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
#     st.success("✅ Document added to vector store!")

# # ----------------------------
# # Query & Re-ranking
# # ----------------------------
# def query_collection(prompt: str, n_results: int = 10):
#     collection = get_vector_collection()
#     return collection.query(query_texts=[prompt], n_results=n_results)

# def re_rank_cross_encoders(documents: List[str], prompt: str) -> tuple[str, List[int]]:
#     """Re-rank top documents using CrossEncoder."""
#     if not documents:
#         return "", []
    
#     # CrossEncoder expects (query, doc) pairs
#     queries = [prompt] * len(documents)
#     scores = cross_encoder.predict(list(zip(queries, documents)))
    
#     # Get top 3 docs
#     top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
#     top_text = "".join([documents[i] for i in top_indices])
#     return top_text, top_indices

# # ----------------------------
# # LLM Call
# # ----------------------------
# def call_llm(context: str, prompt: str):
#     """Call Ollama LLM with top context chunks."""
#     response = ollama.chat(
#         model="moondream:latest",
#         stream=True,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
#         ],
#     )
#     for chunk in response:
#         if not chunk["done"]:
#             yield chunk["message"]["content"]

# # ----------------------------
# # Streamlit App
# # ----------------------------
# st.set_page_config(page_title="📄 RAG QnA")

# # Sidebar: Upload PDF
# with st.sidebar:
#     uploaded_file = st.file_uploader("📑 Upload PDF", type=["pdf"], accept_multiple_files=False)
#     process_btn = st.button("⚡ Process PDF")
    
#     if uploaded_file and process_btn:
#         file_name = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
#         splits = process_document(uploaded_file)
#         add_to_vector_collection(splits, file_name)

# # Main: Ask Questions
# st.header("🗣️ Ask Questions about your PDF")
# prompt = st.text_area("Enter your question here:")
# ask_btn = st.button("🔥 Ask")

# if ask_btn and prompt:
#     results = query_collection(prompt)
#     retrieved_docs = results.get("documents")[0] if results.get("documents") else []
    
#     relevant_text, top_ids = re_rank_cross_encoders(retrieved_docs, prompt)
    
#     # Stream LLM response
#     st.subheader("Answer:")
#     response_generator = call_llm(relevant_text, prompt)
#     response_text = ""
#     for chunk in response_generator:
#         response_text += chunk
#     st.write(response_text)
    
#     # Optional: show retrieved docs and top doc indices
#     with st.expander("Retrieved Documents"):
#         st.write(retrieved_docs)
#     with st.expander("Top Document IDs"):
#         st.write(top_ids)



import os
import tempfile
from typing import List, Union

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Global Config
# ----------------------------
SYSTEM_PROMPT = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Format your response clearly and thoroughly. Maintain context from previous Q&A in the conversation.
"""

# ----------------------------
# Session State Initialization
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "splits" not in st.session_state:
    st.session_state.splits = []

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# ----------------------------
# Lazy-Load Heavy Resources
# ----------------------------
cross_encoder = None

def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return cross_encoder

@st.cache_resource
def get_vector_collection():
    embedding_fn = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return client.get_or_create_collection(
        name="rag_app",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

# ----------------------------
# Document Processing
# ----------------------------
def process_document(uploaded_file: Union[bytes, "streamlit.runtime.uploaded_file_manager.UploadedFile"], fast_mode=False) -> List[Document]:
    """Load PDF using a temporary file. Supports fast mode for small PDFs."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        if fast_mode:
            full_text = "".join([d.page_content for d in docs])
            return [Document(page_content=full_text, metadata={"source": uploaded_file.name})]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return splitter.split_documents(docs)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ----------------------------
# Vector Storage
# ----------------------------
def add_to_vector_collection(all_splits: List[Document], file_name: str):
    collection = get_vector_collection()
    docs = [d.page_content for d in all_splits]
    metadatas = [d.metadata for d in all_splits]
    ids = [f"{file_name}_{i}" for i in range(len(all_splits))]
    collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
    st.success("✅ Document added to vector store!")

# ----------------------------
# Query & Re-ranking
# ----------------------------
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)

def re_rank_cross_encoders(documents: List[str], prompt: str):
    encoder = get_cross_encoder()
    queries = [prompt] * len(documents)
    scores = encoder.predict(list(zip(queries, documents)))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    top_text = "".join([documents[i] for i in top_indices])
    return top_text, top_indices

# ----------------------------
# LLM Call with Chat History
# ----------------------------
def call_llm(context: str, user_prompt: str, max_history=5):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Only send last max_history conversation turns
    messages.extend(st.session_state.chat_history[-2*max_history:])
    messages.append({"role": "user", "content": f"Context: {context}\nQuestion: {user_prompt}"})

    response = ollama.chat(model="llama3:latest", stream=True, messages=messages)
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="📄 RAG QnA Optimized Instant UI")

# Sidebar: Upload PDF
with st.sidebar:
    uploaded_file = st.file_uploader("📑 Upload PDF", type=["pdf"], accept_multiple_files=False)
    fast_mode_checkbox = st.checkbox("⚡ Fast Mode (instant for small PDFs)")
    process_btn = st.button("⚡ Process PDF")

    if uploaded_file and process_btn:
        st.session_state.file_name = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
        st.session_state.splits = process_document(uploaded_file, fast_mode=fast_mode_checkbox)
        add_to_vector_collection(st.session_state.splits, st.session_state.file_name)

# Main: Chat
st.header("🗣️ Ask Questions about your PDF")
prompt = st.text_area("Enter your question here:")
ask_btn = st.button("🔥 Ask")

if ask_btn and prompt:
    if not st.session_state.splits:
        st.warning("Please upload and process a PDF first!")
        st.stop()

    # Fast mode: skip vector search & CrossEncoder
    if fast_mode_checkbox:
        context = "".join([d.page_content for d in st.session_state.splits])
    else:
        # Normal mode: vector search + re-rank
        results = query_collection(prompt)
        retrieved_docs = results.get("documents")[0] if results.get("documents") else []
        context, top_ids = re_rank_cross_encoders(retrieved_docs, prompt)

    # Stream LLM response
    st.subheader("Answer:")
    response_generator = call_llm(context, prompt)
    response_text = ""
    for chunk in response_generator:
        response_text += chunk

    # Save conversation in chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    # Show top document IDs for normal mode
    if not fast_mode_checkbox:
        with st.expander("Top Document IDs"):
            st.write(top_ids)


###https://www.youtube.com/watch?v=1y2TohQdNbo