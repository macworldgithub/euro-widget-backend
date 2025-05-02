from backup import load_bike_data, get_embedding
import faiss
import numpy as np

bike_data, bike_descriptions = load_bike_data("euro_cars_only.json")

embeddings = np.array([get_embedding(desc) for desc in bike_descriptions], dtype=np.float32)
print(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
print(index)
index.add(embeddings)
faiss.write_index(index, "asian_cars_indexnew2.idx")

print("✅ FAISS index regenerated and saved as bike_faiss_index.idx")


