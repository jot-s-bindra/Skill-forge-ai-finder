from sentence_transformers import SentenceTransformer, util

print("🚀 Loading Hugging Face Model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model Loaded Successfully!")

def test_similarity():
    sentence1 = "I love deep learning and AI."
    sentence2 = "Machine learning and artificial intelligence are amazing."

    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    print(f"✅ Similarity Score: {similarity_score}")

if __name__ == "__main__":
    test_similarity()
