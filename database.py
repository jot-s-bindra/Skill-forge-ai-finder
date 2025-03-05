import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["skill-forge"]
users_collection = db["users"]

def test_connection():
    try:
        users_collection.insert_one({"test": "connection_check"})
        print("✅ MongoDB Connected Successfully")
    except Exception as e:
        print("❌ MongoDB Connection Failed:", str(e))

if __name__ == "__main__":
    test_connection()
