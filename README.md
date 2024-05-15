# 📦 Advanced RAG Chat Application with Llama3, LlamaIndex, and Qdrant

Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

## Tools Used:
1. Ollama (a tool that allows you to run open-source large language models (LLMs) locally on your machine)
2. Llama3 (the most capable openly available LLM to date)
3. LlamaIndex (a framework for building context-augmented LLM applications)
4. Qdrant (vector similarity search engine and vector database)
5. Streamlit (framework to rapidly build machine learning and data science web apps)

## RAG Mechanism:
1. SimpleDirectoryReader (the simplest way to load data from local files into LlamaIndex)
2. SentenceWindowNodeParser (split a document into nodes, where each node represents a sentence. Each node also contains a window of surrounding sentences in its metadata)
3. Qdrant Vector Store
4. BAAI/bge-small-en-v1.5 Embedding
4. MetadataReplacementPostProcessor (replaces the text of each node with the value of the "window" metadata key)
5. SentenceTransformerRerank (Rerank can speed up an LLM query without sacrificing accuracy (and probably improving it). It does so by pruning away irrelevant nodes from the context.)

## Steps to Reproduce

1. Run Qdrant
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```

2. Run Ollama
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3
    ollama serve
    ```

3. Run Application
    ```
    streamlit run --server.port 8501 app.py
    ```
4. Enjoy !!!


## Demo
![](/media/advancerag.webp)
