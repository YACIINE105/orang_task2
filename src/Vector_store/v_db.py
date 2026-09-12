from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List



class VectorDataBase:
    def __init__(self, vector_size):
        
    
        self.qdrant = QdrantClient(host="localhost", port=6333)

        self.VECTOR_SIZE = vector_size  # match your embedding model's output dim


    def store_embeddings_for_asset(self, asset_id, chunks: List, embeddings: List[List[float]]):
        """
        chunks: list of Document objects (with .page_content, .metadata)
        embeddings: list of vectors, same order/length as chunks
        """
        qdrant_collection = f"asset_{asset_id}"

        self.qdrant.create_collection(
            collection_name=qdrant_collection,
            vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    **chunk.metadata,
                },
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        self.qdrant.upsert(collection_name=qdrant_collection, points=points)
        print(f"Stored {len(points)} embeddings for asset '{asset_id}' in Qdrant.")
        
        
        
    def search_embeddings(self, asset_id: str, query_vector, top_k: int = 5):
        collection_name = f"asset_{asset_id}"

        if not self.qdrant.collection_exists(collection_name):
            print(f"No collection found for asset '{asset_id}'.")
            return []

        results = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text"),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in results
        ]
            
                
                