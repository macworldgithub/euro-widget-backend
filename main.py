
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize OpenAI Grok client
client = OpenAI(
    api_key= "xai-pA9l7Tkegg7TN8mDJE6CBOyLXWRHFLgm9H23qg0ukmlo4QRTx1gUqhMkb66CSDQkdX3FFmgY9A6W2fpH",
    base_url="https://api.x.ai/v1",  # Update if using proxy
)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load car data
def load_car_data(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    cars = []
    descriptions = []

    for info in data:
        car = {}
        if 'make' in info and info['make']:
            car["brand"] = info['make'][0]
        if 'model' in info and info['model']:
            car["model"] = info['model'][0]
        if 'year' in info and info['year']:
            car["year"] = info['year'][0]
        if 'field_engine_capacity' in info and info['field_engine_capacity']:
            car["engine_cc"] = info['field_engine_capacity'][0]
        if 'field_odometer' in info and info['field_odometer'] and 'field_odometer_qualifier' in info and info['field_odometer_qualifier']:
            car["mileage"] = f"{info['field_odometer'][0]} {info['field_odometer_qualifier'][0]}"
        if 'field_price' in info and info['field_price']:
            car["price"] = info['field_price'][0]
        if 'description' in info and info['description']:
            car["description"] = info['description'][0]
            descriptions.append(info['description'][0])
        if 'title' in info and info['title']:
            car["title"] = info['title'][0]
        if 'field_ad_number' in info and info['field_ad_number']:
            car["ad_number"] = info['field_ad_number'][0]
            car["url"] = f"https://justeurocars.com.au//car-details/?ad_number={car['ad_number']}"
        cars.append(car)

    return cars, descriptions

# Sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('multi-qa-mpnet-base-dot-v1') 
# model = SentenceTransformer('all-mpnet-base-v2') 

def get_embedding(text):
    return model.encode(text)

# Load car data and index
car_data, _ = load_car_data("euro_cars_only.json")
index = faiss.read_index("euro_cars_faiss_index_new.idx")

# Search function
def search_similar_cars(query, top_k=50):
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [car_data[i] for i in indices[0]]

# Chat memory
conversation_history = []

def ask_grok(question, cars):
    global conversation_history

    formatted_cars = []
    for car in cars:
        parts = []
        if 'brand' in car:
            parts.append(f"Brand: {car['brand']}")
        if 'model' in car:
            parts.append(f"Model: {car['model']}")
        if 'year' in car:
            parts.append(f"Year: {car['year']}")
        if 'engine_cc' in car:
            parts.append(f"Engine: {car['engine_cc']}cc")
        if 'mileage' in car:
            parts.append(f"Mileage: {car['mileage']}")
        if 'price' in car:
            parts.append(f"Price: ${car['price']}")
        if 'description' in car:
            parts.append(f"Description: {car['description']}")
        if 'title' in car and 'price' in car and 'url' in car:
            parts.append(f"👉 [{car['title']} - ${car['price']}] - {car['url']}")
        formatted_cars.append(", ".join(parts))

    formatted_text = "\n\n".join(formatted_cars)

    conversation_history.append({"role": "user", "content": question})
    conversation_history = conversation_history[-5:]

    system_prompt = {
        "role": "system",
        "content": """
You're a helpful, human-sounding car expert. Only recommend cars if the user requests suggestions.
If they say "hi" or greet you, return a friendly reply and ask how you can help — don’t suggest anything unsolicited.

Keep answers brief, natural, and smart. When cars are mentioned, include a link to their page:
👉 [2022 Honda Accord - $22,500] - https://justeurocars.com.au//car-details/?ad_number=123456
⚠️ Do NOT use Markdown-style links like [title](url). Just use the format above — square brackets, then a hyphen, then the plain link.
Never give generic answers. If confused, ask clarifying questions.
"""
    }

    conversation_history.append({
        "role": "user",
        "content": f"""
Customer Question:
{question}

Top Matches from Inventory:
{formatted_text}

Instructions:
- Recommend only if the customer is asking about car suggestions or comparisons.
- Be short (2–3 lines), helpful, and conversational — like a friendly car expert.
- If appropriate, include direct links to the cars if the user is interested in purchasing.
- Don’t act like a salesperson; focus on fit and clarity.
- If unsure, ask for more info.
"""
    })

    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=[system_prompt] + conversation_history
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history = conversation_history[-5:]
    return reply

# API input model
class QueryRequest(BaseModel):
    query: str

# API endpoint
@app.post("/query")
async def query_cars(request: QueryRequest):
    results = search_similar_cars(request.query)
    answer = ask_grok(request.query, results)
    return {"message": answer}

# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import json
# import faiss
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from fastapi import FastAPI
# from openai import OpenAI
# from pydantic import BaseModel
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware

# # Initialize OpenAI Grok client
# client = OpenAI(  api_key= "xai-pA9l7Tkegg7TN8mDJE6CBOyLXWRHFLgm9H23qg0ukmlo4QRTx1gUqhMkb66CSDQkdX3FFmgY9A6W2fpH",

#     base_url="https://api.x.ai/v1"  # Replace if using proxy or wrapper

# )

# # FastAPI app setup
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load car data
# def load_car_data(json_file):
#     with open(json_file, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     cars = []
#     for info in data:
#         car = {}
#         if 'make' in info and info['make']:
#             car["brand"] = info['make'][0]
#         if 'model' in info and info['model']:
#             car["model"] = info['model'][0]
#         if 'year' in info and info['year']:
#             car["year"] = info['year'][0]
#         if 'field_engine_capacity' in info and info['field_engine_capacity']:
#             car["engine_cc"] = info['field_engine_capacity'][0]
#         if 'field_odometer' in info and info['field_odometer'] and 'field_odometer_qualifier' in info and info['field_odometer_qualifier']:
#             car["mileage"] = f"{info['field_odometer'][0]} {info['field_odometer_qualifier'][0]}"
#         if 'field_price' in info and info['field_price']:
#             car["price"] = info['field_price'][0]
#         if 'description' in info and info['description']:
#             car["description"] = info['description'][0]
#         if 'title' in info and info['title']:
#             car["title"] = info['title'][0]
#         if 'field_ad_number' in info and info['field_ad_number']:
#             car["ad_number"] = info['field_ad_number'][0]
#             car["url"] = f"https://justjdmcars.com.au/car-details/?ad_number={car['ad_number']}"

#         cars.append(car)
#     return cars

# # Load car data and embeddings
# car_data = load_car_data("asian_cars_only.json")
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_embedding(text):
#     return model.encode(text)

# # index = faiss.read_index("asian_cars_faiss_index_new.idx")

# # Search similar cars using FAISS
# def search_similar_cars(query, top_k=150):
#     query_embedding = np.array([get_embedding(query)], dtype=np.float32)
#     distances, indices = index.search(query_embedding, top_k)
#     return [car_data[i] for i in indices[0]]

# # Conversation memory
# conversation_history = []

# def ask_grok(question, cars):
#     global conversation_history

#     formatted_cars = []
#     for car in cars:
#         parts = []
#         if 'brand' in car:
#             parts.append(f"Brand: {car['brand']}")
#         if 'model' in car:
#             parts.append(f"Model: {car['model']}")
#         if 'year' in car:
#             parts.append(f"Year: {car['year']}")
#         if 'engine_cc' in car:
#             parts.append(f"Engine: {car['engine_cc']}cc")
#         if 'mileage' in car:
#             parts.append(f"Mileage: {car['mileage']}")
#         if 'price' in car:
#             parts.append(f"Price: ${car['price']}")
#         if 'description' in car:
#             parts.append(f"Description: {car['description']}")
#         # if 'title' in car and 'price' in car and 'url' in car:
#         #     parts.append(f"👉 [{car['title']} - ${car['price']}] - {car['url']}")
#         if 'title' in car and 'price' in car and 'url' in car:
#             parts.append(f"👉 [{car['title']} - ${car['price']}] - {car['url']}")

#         formatted_cars.append(", ".join(parts))

#     formatted_text = "\n\n".join(formatted_cars)

#     conversation_history.append({"role": "user", "content": question})
#     conversation_history = conversation_history[-5:]

#     system_prompt = {
#         "role": "system",
#         "content": """
# You're a helpful, human-sounding car expert. Only recommend cars if the user requests suggestions.
# If they say "hi" or greet you, return a friendly reply and ask how you can help — don’t suggest anything unsolicited.

# Keep answers brief, natural, and smart. When cars are mentioned, include a link to their page:
# 👉 [2022 Honda Accord - $22,500] - https://justjdmcars.com.au/car-details/?ad_number=123456
# ⚠️ Do NOT use Markdown-style links like [title](url). Just use the format above — square brackets, then a hyphen, then the plain link.
# Never give generic answers. If confused, ask clarifying questions.
# """
#     }

#     conversation_history.append({
#         "role": "user",
#         "content": f"""
# Customer Question:
# {question}

# Top Matches from Inventory:
# {formatted_text}

# Instructions:
# - Recommend only if the customer is asking about car suggestions or comparisons.
# - Be short (2–3 lines), helpful, and conversational — like a friendly car expert.
# - If appropriate, include direct links to the cars if the user is interested in purchasing.
# - Don’t act like a salesperson; focus on fit and clarity.
# - If unsure, ask for more info.
# """
#     })

#     response = client.chat.completions.create(
#         model="grok-2-latest",
#         messages=[system_prompt] + conversation_history
#     )

#     reply = response.choices[0].message.content
#     conversation_history.append({"role": "assistant", "content": reply})
#     conversation_history = conversation_history[-5:]
#     return reply

# # Request model
# class QueryRequest(BaseModel):
#     query: str

# @app.post("/query")
# async def query_cars(request: QueryRequest):
#     results = search_similar_cars(request.query)
#     answer = ask_grok(request.query, results)
#     return {"message": answer}

# @app.get("/anas")
# def health_check():
#     return {"message": "Server is Running ...."}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8084, reload=True)
