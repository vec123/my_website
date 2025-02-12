import requests
import chromadb
from flask import Flask, request, jsonify
from serverless_wsgi import handle_request

app = Flask(__name__)

# Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="/chroma_db")
chroma_client = chromadb.Client(path="/chroma_db")
collection = chroma_client.get_collection(name="my_markdown_data")

TOGETHER_API_KEY = "your_together_ai_api_key"

def generate_llm_response(prompt):
    """Queries Together AI API to generate a response"""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistralai/Mistral-7B-Instruct",  # Use Llama-2, Mixtral, etc.
        "messages": [{"role": "system", "content": "You are a helpful chatbot."},
                     {"role": "user", "content": prompt}]
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

@app.route('/query', methods=['POST', 'OPTIONS'])
def query():
    if request.method == "OPTIONS":
        return '', 204

    user_query = request.json.get('question', '').strip()

    search_results = collection.query(query_texts=[user_query], n_results=3)
    retrieved_docs = search_results["documents"][0] if search_results["documents"] else []
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    You are an AI assistant. Answer the user's question using the provided context.
    If the context does not have the answer, reply with "I don't know."
    
    Context:
    {context}

    User Question: {user_query}

    Answer:
    """
    
    generated_response = generate_llm_response(prompt)

    return jsonify({'response': generated_response})

def lambda_handler(event, context):
    return handle_request(app, event, context)
