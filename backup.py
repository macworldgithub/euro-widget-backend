import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

client = OpenAI(
  api_key= "xai-pA9l7Tkegg7TN8mDJE6CBOyLXWRHFLgm9H23qg0ukmlo4QRTx1gUqhMkb66CSDQkdX3FFmgY9A6W2fpH",
  base_url= "https://api.x.ai/v1"
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Step 1: Load and Preprocess the Bike Data
def load_bike_data(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        bike_data = json.load(file)
    
    bikes = []
    descriptions = []
    
    for bike_info in bike_data:  # No longer inside ['hits']
        bike = {}
        
        if 'make' in bike_info and bike_info['make']:
            bike["brand"] = bike_info['make'][0]
        if 'model' in bike_info and bike_info['model']:
            bike["model"] = bike_info['model'][0]
        if 'year' in bike_info and bike_info['year']:
            bike["year"] = bike_info['year'][0]
        if 'field_engine_capacity' in bike_info and bike_info['field_engine_capacity']:
            bike["engine_cc"] = bike_info['field_engine_capacity'][0]
        if 'field_odometer' in bike_info and bike_info['field_odometer'] and 'field_odometer_qualifier' in bike_info and bike_info['field_odometer_qualifier']:
            bike["mileage"] = f"{bike_info['field_odometer'][0]} {bike_info['field_odometer_qualifier'][0]}"
        if 'field_price' in bike_info and bike_info['field_price']:
            bike["price"] = bike_info['field_price'][0]
        if 'description' in bike_info and bike_info['description']:
            bike["description"] = bike_info['description'][0]
        if 'title' in bike_info and bike_info['title']:
            bike["title"] = bike_info['title'][0]
        if 'field_is_wreckable' in bike_info:
            bike["is_wreckable"] = bike_info['field_is_wreckable'][0]
        if 'package_title' in bike_info and bike_info['package_title']:
            bike["package_title"] = bike_info['package_title'][0]
        if 'field_verticals' in bike_info and bike_info['field_verticals']:
            bike["verticals"] = bike_info['field_verticals'][0]
        if 'body_type' in bike_info and bike_info['body_type']:
            bike["body_type"] = bike_info['body_type'][0]
        if 'suburb' in bike_info and bike_info['suburb']:
            bike["suburb"] = bike_info['suburb'][0]
        if 'state' in bike_info and bike_info['state']:
            bike["state"] = bike_info['state'][0]
        if 'postcode' in bike_info and bike_info['postcode']:
            bike["postcode"] = bike_info['postcode'][0]
        if 'field_ad_number' in bike_info and bike_info['field_ad_number']:
            bike["ad_number"] = bike_info['field_ad_number'][0]
        if 'field_ad_source' in bike_info and bike_info['field_ad_source']:
            bike["ad_source"] = bike_info['field_ad_source'][0]
        if 'field_ad_type' in bike_info and bike_info['field_ad_type']:
            bike["ad_type"] = bike_info['field_ad_type'][0]
        if 'field_engine_size' in bike_info and bike_info['field_engine_size']:
            bike["engine_size"] = bike_info['field_engine_size'][0]
        
        bikes.append(bike)

        # Compose description
        description_parts = []
        if 'brand' in bike:
            description_parts.append(f"Brand: {bike['brand']}")
        if 'model' in bike:
            description_parts.append(f"Model: {bike['model']}")
        if 'year' in bike:
            description_parts.append(f"Year: {bike['year']}")
        if 'engine_cc' in bike:
            description_parts.append(f"Engine: {bike['engine_cc']}cc")
        if 'mileage' in bike:
            description_parts.append(f"Mileage: {bike['mileage']}")
        if 'price' in bike:
            description_parts.append(f"Price: ${bike['price']}")
        if 'description' in bike:
            description_parts.append(f"Description: {bike['description']}")
        if 'title' in bike:
            description_parts.append(f"Title: {bike['title']}")

        descriptions.append(", ".join(description_parts))
    
    return bikes, descriptions



bike_data, bike_descriptions = load_bike_data("euro_cars_only.json")

# 🔹 Step 2: Generate Embeddings using OpenAI
model = SentenceTransformer('all-MiniLM-L6-v2') 
# model = SentenceTransformer('multi-qa-mpnet-base-dot-v1') 
# model = SentenceTransformer('all-mpnet-base-v2') 

def get_embedding(text):
   
    return model.encode(text)
   

bike_embeddings = np.array([get_embedding(desc) for desc in bike_descriptions], dtype=np.float32)

# 🔹 Step 3: Store Embeddings in FAISS
# index = faiss.IndexFlatL2(bike_embeddings.shape[1])
# index.add(bike_embeddings)

# faiss.write_index(index, "boats_index.idx")

# index = faiss.read_index("bike_faiss_index2.idx")




if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=8084, reload=True)