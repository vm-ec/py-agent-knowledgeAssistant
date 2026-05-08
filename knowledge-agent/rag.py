from openai import OpenAI
from dotenv import load_dotenv
from pinecone_client import get_index
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve(query):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_vec = response.data[0].embedding
    
    index = get_index()

    results = index.query(
        vector=query_vec,
        top_k=5,
        include_metadata=True
    )

    docs = [match["metadata"]["text"] for match in results["matches"]]

    return "\n".join(docs)
