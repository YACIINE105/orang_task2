from pymongo import MongoClient
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

def clear_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    client.drop_database(os.getenv("DB_NAME"))
    print(f"Dropped Mongo db: {os.getenv('DB_NAME')}")

def clear_qdrant():
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections().collections
    for c in collections:
        client.delete_collection(c.name)
        print(f"Deleted Qdrant collection: {c.name}")

if __name__ == "__main__":
    clear_mongo()
    clear_qdrant()