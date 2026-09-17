from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()


class OpenAIEmbeddingProvider:
    def __init__(self):
        pass

    def get_embedding_provider(self):
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )

    def embed_text(self, texts: List[str], batch_size: int = 100):
        client = self.get_embedding_provider()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = client.embed_documents(batch)  
            all_embeddings.extend(batch_embeddings)
            print(f"Embedded {i + len(batch)}/{len(texts)}")

        return all_embeddings

    def get_embeddings_for_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        return self.embed_text(texts)

    def embed_query(self, query: str):
        client = self.get_embedding_provider()
        return client.embed_query(query)  
    
    