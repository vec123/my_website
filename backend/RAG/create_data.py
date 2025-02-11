import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import glob

my_chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Stores the database persistently
collection = chroma_client.get_or_create_collection(name="blog_rag", embedding_function=OpenAIEmbeddingFunction(api_key="your_openai_api_key"))