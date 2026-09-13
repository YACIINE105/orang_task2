from Controllers.ProcessController import ProcessController
from mongo_store.mongo_db import DataBase
from Vector_store.v_db import VectorDataBase
from llm.OpenaiProvider import OpenAIGenerationProvider
from embedding.OpenAIProvider import OpenAIEmbeddingProvider
from dotenv import load_dotenv
import os

load_dotenv()

pro = ProcessController()
mongo = DataBase(db_name=os.getenv("DB_NAME"))
qdrant = VectorDataBase(vector_size=os.getenv("EMBEDDING_MODEL_DIM"))
embeddings_client = OpenAIEmbeddingProvider()
generation_client = OpenAIGenerationProvider()

file_path = "/home/yacine_105/orange_tasks/task2/Sherlock Internship Challenge.pdf"
chunks = pro.process_text(file_path=file_path)
print(f"Total chunks: {len(chunks)}")
print(f"Max chunk length: {max(len(c.page_content) for c in chunks)}")

asset_id = input("Enter asset id: ").strip()


inserted_ids = mongo.store_chunks_for_asset(asset_id=asset_id, chunks=chunks)

if inserted_ids is not None:
    print("Skipping embedding step (Mongo insert didn't happen).")
else:
    embeddings = embeddings_client.get_embeddings_for_chunks(chunks=chunks)
    qdrant.store_embeddings_for_asset(asset_id=asset_id, chunks=chunks, embeddings=embeddings)

while True:
    print("#######################    welcome to the minimal rag system    #######################")
    
    query = input("Enter a search query: ").strip()
    if query.lower() in ("exit", "q"):
        break
    query_vector = embeddings_client.embed_query(query=query)

    results = qdrant.search_embeddings(asset_id, query_vector, top_k=5)

    answer = generation_client.generate_text(query=query, search_results=results)
    print("\n\n", f"Assistant: {answer}")


