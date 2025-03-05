from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pymongo
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(MONGO_URI)
db = client["skill-forge"]
users_collection = db["users"]

print("🚀 Loading Similarity Model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model Loaded Successfully!")

def find_best_partners(user_profile, all_students):
    user_embedding = model.encode(user_profile, convert_to_tensor=True)
    similarities = []

    for student in all_students:
        student_profile = f"{student.get('tech_stacks', [])} {student.get('semesters', [])}"
        student_embedding = model.encode(student_profile, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(user_embedding, student_embedding).item()
        similarities.append((student["uid"], similarity_score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    if len(similarities) < 2:
        return None  
    
    return similarities[1]  


@app.route("/api/recommend-partner", methods=["POST"])
def recommend_partner():
    try:
        data = request.json
        uid = data.get("uid")

        user = users_collection.find_one({"uid": uid, "role": "student"})
        if not user:
            return jsonify({"error": "User not found or not a student"}), 404

        all_students = list(users_collection.find({"uid": {"$ne": uid}, "role": "student"}))

        user_profile = f"{user.get('tech_stacks', [])} {user.get('semesters', [])}"

        best_partner = find_best_partners(user_profile, all_students)

        if not best_partner:
            return jsonify({"message": "No suitable partner found"}), 404

        return jsonify({"recommended_partner": best_partner})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=6000)
