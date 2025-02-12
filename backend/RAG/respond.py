from flask import Flask, request, jsonify
from serverless_wsgi import handle_request  # Converts Flask requests for API Gateway

# Initialize Flask
app = Flask(__name__)

# Query Processing Logic
@app.route('/query', methods=['POST'])
def query():
    """Handles user queries and returns predefined responses"""
    user_query = request.json.get('question', '').strip().lower() if request.json else ""

    response = "hello world" if user_query == "test" else "I do not understand"

    return jsonify({'response': response})

# Function to handle CORS in preflight (OPTIONS) requests
def _build_cors_preflight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

# Function to attach CORS headers to all responses
def _build_cors_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# AWS Lambda Handler
def lambda_handler(event, context):
    return handle_request(app, event, context)