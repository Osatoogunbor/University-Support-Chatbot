import json
import os
import time
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

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
