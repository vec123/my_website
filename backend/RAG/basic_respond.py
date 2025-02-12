from flask import Flask, request, jsonify
from serverless_wsgi import handle_request  # Converts Flask requests for API Gateway

app = Flask(__name__)

# Attach CORS headers to all responses
@app.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Query Processing Logic
@app.route('/query', methods=['POST', 'OPTIONS'])
def query():
    """Handles user queries and returns predefined responses"""

    # Handle preflight (OPTIONS) request
    if request.method == "OPTIONS":
        return '', 204  # Empty response for OPTIONS request

    user_query = request.json.get('question', '').strip().lower() if request.json else ""
    response = "hello world" if user_query == "test" else "I do not understand"

    return jsonify({'response': response})

# AWS Lambda Handler
def lambda_handler(event, context):
    return handle_request(app, event, context)