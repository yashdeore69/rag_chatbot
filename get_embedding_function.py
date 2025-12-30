from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:137m-v1.5-fp16"
    )

    return embeddings

