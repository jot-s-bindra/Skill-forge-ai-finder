from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pymongo
import os
from bson import ObjectId 
from dotenv import load_dotenv
from flask_cors import CORS  

app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(MONGO_URI)
db = client["skill-forge"]
users_collection = db["users"]
projects_collection = db["projects"]

print("🚀 Loading Similarity Model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model Loaded Successfully!")

def find_best_partner(user_uid, project, all_students):
    """Finds the most suitable partner based on project requirements, excluding the request sender."""
    project_profile = f"{project['description']} {' '.join(project.get('required_techstacks', []))}"
    project_embedding = model.encode(project_profile, convert_to_tensor=True)

    similarities = []

    for student in all_students:
        if student["uid"] == user_uid:  
            continue  

        student_profile = " ".join(student.get("tech_stacks", [])) + " " + " ".join(
            [str(course) for sem in student.get("semesters", []) for course in sem.get("courses", [])]
        )
        student_embedding = model.encode(student_profile, convert_to_tensor=True)

        similarity_score = util.pytorch_cos_sim(project_embedding, student_embedding).item()
        similarities.append((student["uid"], similarity_score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[0][0] if len(similarities) > 0 else None


@app.route("/api/recommend-partner", methods=["POST"])
def recommend_partner():
    try:
        data = request.json
        uid = data.get("uid")
        project_id = data.get("project_id")

        if not uid or not project_id:
            return jsonify({"error": "UID and Project ID are required"}), 400

        project = projects_collection.find_one({"_id": ObjectId(project_id)})  
        if not project:
            return jsonify({"error": "Project not found"}), 404

        assigned_students = {app["uid"] for app in project.get("applicants", [])}
        available_students = list(users_collection.find({
            "uid": {"$nin": list(assigned_students) + [uid]}, 
            "role": "student"
        }))

        if not available_students:
            return jsonify({"message": "No available students found"}), 404

        recommended_partner = find_best_partner(uid, project, available_students)

        if not recommended_partner:
            return jsonify({"message": "No suitable partner found"}), 404

        return jsonify({"recommended_partner": recommended_partner})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
