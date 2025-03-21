import azure.functions as func
import logging
import json
from sentence_transformers import SentenceTransformer

# Create the FunctionApp with anonymous HTTP access
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Load the SentenceTransformer model once at startup
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

@app.route(route="embed", methods=["POST"])
def embed(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function for text embedding has been called.")

    try:
        # Parse JSON from the request body and extract the "input" field
        body = req.get_json()
        text = body.get("input", "")
    except ValueError:
        return func.HttpResponse("Invalid or missing JSON body.", status_code=400)

    if not text:
        return func.HttpResponse("Error: 'input' field missing or empty in JSON body.", status_code=400)

    try:
        # Generate embedding using the SentenceTransformer model
        embedding = model.encode(text)
    except Exception as e:
        return func.HttpResponse(f"Error generating embedding: {str(e)}", status_code=500)

    # Convert the embedding (likely a numpy array) to a list for JSON serialization
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    # Wrap the vector in the desired JSON format
    response_data = {
        "data": [
            {"embedding": embedding}
        ]
    }
    
    return func.HttpResponse(
        json.dumps(response_data),
        mimetype="application/json",
        status_code=200
    )
