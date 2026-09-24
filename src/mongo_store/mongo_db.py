from pymongo import MongoClient
from datetime import datetime, timezone


class DataBase:

    def __init__(self, db_name):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]

    def store_chunks_for_asset(self, asset_id, chunks, source_path=None):
        collection_name = f"chunks_of_assets_{asset_id}"
        collection = self.db[collection_name]

        if source_path and collection.find_one({"source": source_path}):
            print(f"Asset '{asset_id}' / file '{source_path}' already stored — skipping.")
            return None

        chunk_docs = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", source_path),
                "page": chunk.metadata.get("page"),
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]

        result = collection.insert_many(chunk_docs)
        print(f"Stored {len(chunk_docs)} chunks for asset '{asset_id}' / file '{source_path}'.")
        return result.inserted_ids

    def save_chat_history(self, asset_id: str, history: list[dict]):
        self.db["chat_history"].update_one(
            {"asset_id": asset_id},
            {"$set": {"history": history}},
            upsert=True,
        )

    def get_chat_history(self, asset_id: str) -> list[dict]:
        doc = self.db["chat_history"].find_one({"asset_id": asset_id})
        return doc["history"] if doc else []

    def save_session_chat(self, session_id: str, asset_id: str, history: list[dict]):
        if not session_id:
            session_id = asset_id or "default_session"
        self.db["chat_sessions"].update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "session_id": session_id,
                    "asset_id": asset_id,
                    "history": history,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    def get_session_chat(self, session_id: str, asset_id: str | None = None) -> list[dict]:
        if not session_id:
            return []
        query = {"session_id": session_id}
        if asset_id:
            query["asset_id"] = asset_id
        doc = self.db["chat_sessions"].find_one(query)
        return doc["history"] if doc else []


if __name__ == "__main__":
    asset_id = input("Enter asset id: ").strip()

