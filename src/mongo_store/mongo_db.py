from pymongo import MongoClient

from Controllers.ProcessController import ProcessController


class DataBase:

    def __init__(self, db_name):
        
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]

    def store_chunks_for_asset(self, asset_id, chunks):
        collection_name = f"asset_{asset_id}"

        # Check if this collection already exists in the DB
        if collection_name in self.db.list_collection_names():
            print(f"Asset '{asset_id}' already stored — skipping.")
            return self.db[collection_name]

        # Doesn't exist yet — create it by inserting the chunks
        collection = self.db[collection_name]

        chunk_docs = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]

        collection.insert_many(chunk_docs)
        print(f"Stored {len(chunk_docs)} chunks for asset '{asset_id}'.")
        return collection


if __name__ == "__main__":
    asset_id = input("Enter asset id: ").strip()
    # chunks = your_chunking_function(...)  # your existing chunking logic
    # store_chunks_for_asset(asset_id, chunks)