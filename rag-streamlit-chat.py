import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def initialize_llm():
    Settings.llm = Ollama(model="llama3", request_timeout=30.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model


def process_document(file):
    with open("uploaded_document.txt", "wb") as f:
        f.write(file.getbuffer())
    return ["uploaded_document.txt"]


def create_index(document_paths):
    documents = SimpleDirectoryReader(input_files=document_paths).load_data()

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    nodes = node_parser.get_nodes_from_documents(documents)

    client = QdrantClient(url="http://localhost:6333")

    collection_name = "RAG"
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    return index


def query_index(index, query):
    query_engine = index.as_query_engine(
        similarity_top_k=6,
        llm=Settings.llm,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base"),
        ],
        vector_store_query_mode="hybrid",
        alpha=0.5,
    )

    response = query_engine.query(query)
    return response


class ChatEngine:
    def __init__(self, index):
        self.index = index
        self.conversation_history = []

    def chat(self, query):
        self.conversation_history.append({"role": "user", "content": query})
        response = query_index(self.index, query)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


# Streamlit app interface
st.title("Advanced RAG Chat Application")

st.sidebar.title("Upload and Index Document")
uploaded_file = st.sidebar.file_uploader("Choose a text file", type="txt")

initialize_llm()

if uploaded_file is not None:
    document_paths = process_document(uploaded_file)
    st.sidebar.write("Document uploaded successfully.")

    index = create_index(document_paths)
    st.sidebar.write("Document indexed successfully.")

    chat_engine = ChatEngine(index)

    st.write("### Chat with the document")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = chat_engine

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    if query := st.chat_input("Enter your message"):
        with st.chat_message("user"):
            st.markdown(query)

        response = st.session_state.chat_engine.chat(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.sidebar.write("Please upload a document to start.")
