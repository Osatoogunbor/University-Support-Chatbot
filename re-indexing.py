
import json
import os
import time
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# Retrieve secrets from Streamlit
OPENAI_API_KEY = st.secrets["openai_api_key"]
PINECONE_API_KEY = st.secrets["pinecone_api_key"]
# If you need Pinecone ENV, add it:
PINECONE_ENV = st.secrets["pinecone_env"]

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in st.secrets.")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in st.secrets.")
if not PINECONE_ENV:
    raise ValueError("❌ PINECONE_ENV not found in st.secrets.")

# Initialize OpenAI & Pinecone
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index("ai-powered-chatbot")


# Path to merged knowledge base
MERGED_JSON_PATH = "/mnt/c/Users/osato/openai_setup/merged_knowledge_base.json"

if not os.path.isfile(MERGED_JSON_PATH):
    raise FileNotFoundError("❌ Merged knowledge base file not found.")

# Load merged knowledge base
with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
    knowledgebase = json.load(f)

qa_pairs = knowledgebase.get("qa_pairs", [])
print(f"✅ Loaded {len(qa_pairs)} QA pairs from merged file.")

# Re-index the knowledge base
def index_qa_pairs(pairs):
    for idx, qa in enumerate(pairs):
        question = qa.get("question", "").strip()

        # Ensure "main_points" is a list and join its elements into a single string
        answer_list = qa.get("answer", {}).get("main_points", [])
        if isinstance(answer_list, list):
            answer = " ".join(answer_list).strip()
        else:
            answer = str(answer_list).strip()  # Handle unexpected cases

        full_text = f"Q: {question}\nA: {answer}"

        # Generate embeddings
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=full_text
        )
        vector = response.data[0].embedding

        # Upsert into Pinecone
        index.upsert([
            {
                "id": f"qa_{idx}",
                "values": vector,
                "metadata": {
                    "question": question,
                    "answer": answer
                }
            }
        ], )

        time.sleep(0.1)  # Avoid hitting rate limits

if __name__ == "__main__":
    print("🔄 Re-indexing QA pairs...")
    index_qa_pairs(qa_pairs)
    print("✅ Pinecone index updated.")
