from Controllers.ProcessController import ProcessController
from mongo_store.mongo_db import DataBase
from Vector_store.v_db import VectorDataBase
from embedding.OpenAIProvider import OpenAIEmbeddingProvider
from graph.nodes import AgentNodes
from graph.build import build_graph
from dotenv import load_dotenv
import glob
import os

load_dotenv()

pro = ProcessController()
mongo = DataBase(db_name=os.getenv("DB_NAME"))
qdrant = VectorDataBase(vector_size=int(os.getenv("EMBEDDING_MODEL_DIM")))
embeddings_client = OpenAIEmbeddingProvider()

nodes = AgentNodes()
graph = build_graph(nodes)


# ---------- ingestion ----------
folder_path = "/home/yacine_105/orange_tasks/task2/pdfs"
pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
print(f"Found {len(pdf_files)} PDFs")

asset_id = input("Enter asset id: ").strip()

for file_path in pdf_files:
    print(f"\nProcessing: {file_path}")
    chunks = pro.process_text(file_path=file_path)
    for c in chunks:
        c.metadata["source"] = file_path
    print(f"Chunks: {len(chunks)}")

    inserted_ids = mongo.store_chunks_for_asset(asset_id=asset_id, chunks=chunks, source_path=file_path)
    if inserted_ids is None:
        print("Skipping embedding step (asset already exists).")
        continue

    embeddings = embeddings_client.get_embeddings_for_chunks(chunks=chunks)
    qdrant.store_embeddings_for_asset(asset_id=asset_id, chunks=chunks, embeddings=embeddings)

# ---------- resume history ----------
stored_history = mongo.get_chat_history(asset_id=asset_id)
if stored_history:
    nodes.provider.load_history_from_dicts(asset_id, stored_history)
    print(f"Resumed {len(stored_history)} prior messages for asset '{asset_id}'.")

# ---------- chat loop ----------
print("####### multi-agent RAG (type 'exit' or 'q' to quit) #######")
while True:
    query = input("\nEnter a request: ").strip()
    if query.lower() in ("exit", "q"):
        break
    if not query:
        continue

    result = graph.invoke({"asset_id": asset_id, "query": query})
    print(f"\nAssistant: {result['answer']}")

    mongo.save_chat_history(asset_id=asset_id, history=nodes.provider.export_history(asset_id))
    
    