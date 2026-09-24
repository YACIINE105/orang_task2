from pymongo import MongoClient


class DataBase:

    def __init__(self, db_name):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    def store_chunks_for_asset(self, asset_id, chunks):
        collection_name = f"chunks_of_assets_{asset_id}"

        if collection_name in self.db.list_collection_names():
            print(f"Asset '{asset_id}' already stored — skipping.")
            return None  

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

        result = collection.insert_many(chunk_docs)
        print(f"Stored {len(chunk_docs)} chunks for asset '{asset_id}'.")
        return result.inserted_ids

if __name__ == "__main__":
    asset_id = input("Enter asset id: ").strip()

