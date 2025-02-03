#!/usr/bin/env python
# coding: utf-8

# In[19]:


import json
import streamlit as st
import openai
from openai import AsyncOpenAI
# ✅ Access API keys securely
OPENAI_API_KEY = st.secrets["openai_api_key"]
# ✅ Check if API keys are loaded correctly
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found! Check your Streamlit secrets.")

# ✅ Initialize OpenAI & Pinecone
openai.api_key = OPENAI_API_KEY

# ✅ Define aclient for AsyncOpenAI usage
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

print("✅ API key loaded successfully!")
print("✅ OpenAI client initialized!")
# ------------------------------------------------------------------------
# Minimal helper to handle 'answer' which may be a dict or a string
# ------------------------------------------------------------------------
def convert_answer_to_string(ans):
    """Convert the 'answer' field to a string if it's a dict."""
    if isinstance(ans, dict):
        # Minimal approach: just turn the dict into a string
        # (If you need a more advanced approach, build a textual summary.)
        return str(ans)
    else:
        return ans if ans else ""

# ------------------------------------------------------------------------
# Load knowledge base
# ------------------------------------------------------------------------
knowledgebase_file = "knowledge_base.json"
output_embedding_file = "knowledgebase_embeddings.json"

try:
    with open(knowledgebase_file, "r", encoding="utf-8") as f:
        knowledgebase = json.load(f)
    print(f"📄 Successfully loaded {knowledgebase_file}.")
except Exception as e:
    print(f"❌ Error loading JSON file: {e}")
    exit()

# Ensure correct structure
if "qa_pairs" not in knowledgebase:
    print("❌ Error: 'qa_pairs' key is missing from JSON. Check file structure.")
    exit()

qa_pairs = knowledgebase["qa_pairs"]
embeddings_data = []

# ------------------------------------------------------------------------
# Generate embeddings
# ------------------------------------------------------------------------
for idx, entry in enumerate(qa_pairs):
    try:
        question = entry.get("question", "").strip()
        # answer might be a dict or string; convert to string consistently
        answer_raw = entry.get("answer", "")
        answer = convert_answer_to_string(answer_raw).strip()

        if not question or not answer:
            print(f"⚠️ Skipping entry {idx} due to missing text.")
            continue

        combined_text = f"Q: {question}\nA: {answer}"

        # Call embeddings endpoint via our OpenAI client
        response = aclient.embeddings.create(
            input=combined_text,
            model="text-embedding-ada-002"
        )
        # NOTE: Depending on your library version, data access may vary
        embedding = response.data[0].embedding

        # Store embedding
        embeddings_data.append({
            "id": entry.get("id", idx),
            "question": question,
            "answer": answer,
            "embedding": embedding
        })

        print(f"✅ Embedded entry {idx + 1}/{len(qa_pairs)}")

    except Exception as e:
        print(f"❌ Error embedding entry {idx}: {e}")

# ------------------------------------------------------------------------
# Save embeddings
# ------------------------------------------------------------------------
try:
    with open(output_embedding_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=4)
    print(f"✅ Embeddings successfully saved to {output_embedding_file}")
except Exception as e:
    print(f"❌ Error saving embeddings: {e}")





