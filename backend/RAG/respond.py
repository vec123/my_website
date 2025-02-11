from flask import Flask, request, jsonify
import chromadb

# Initialize Flask
app = Flask(__name__)

# Simple Query Processing Logic
@app.route('/query', methods=['POST'])
def query():
    """Handles user queries and returns predefined responses"""
    user_query = request.json.get('question', '').strip().lower()

    if user_query == "test":
        response = "hello world"
    else:
        response = "I do not understand"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
