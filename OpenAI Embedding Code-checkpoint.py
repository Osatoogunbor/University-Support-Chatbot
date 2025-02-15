#!/usr/bin/env python3
from typing import Optional, List
import asyncio
import streamlit as st
import openai
from openai import AsyncOpenAI
from pinecone import Pinecone
from transformers import pipeline

# No dotenv usage here
# from dotenv import load_dotenv
# load_dotenv()

# Grab secrets from st.secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]
PINECONE_API_KEY = st.secrets["pinecone_api_key"]
PINECONE_ENV = st.secrets["pinecone_env"]

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in st.secrets.")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in st.secrets.")
if not PINECONE_ENV:
    raise ValueError("❌ Missing PINECONE_ENV in st.secrets.")

openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index("ai-powered-chatbot")

# Paths
MERGED_JSON_PATH = "/mnt/c/Users/osato/openai_setup/merged_knowledge_base.json"
EMBEDDINGS_OUTPUT_PATH = "/mnt/c/Users/osato/openai_setup/knowledgebase_embeddings.json"

# --------------------------------------------------------------------------
# 2. LOAD MERGED KNOWLEDGE BASE
# --------------------------------------------------------------------------
if not os.path.isfile(MERGED_JSON_PATH):
    raise FileNotFoundError(f"❌ File not found: {MERGED_JSON_PATH}")

with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
    knowledgebase = json.load(f)

qa_pairs = knowledgebase.get("qa_pairs", [])
num_qas = len(qa_pairs)
print(f"✅ Loaded {num_qas} QA entries from: {MERGED_JSON_PATH}")

if num_qas == 0:
    print("⚠️ No QA pairs found. Nothing to embed. Exiting.")
    raise SystemExit

# --------------------------------------------------------------------------
# 3. TOKENIZER & CHUNKING UTILS
# --------------------------------------------------------------------------
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

def count_tokens(text: str) -> int:
    """Count tokens for text using the text-embedding-ada-002 tokenizer."""
    return len(tokenizer.encode(text))

def split_text_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
    """Naive word-split approach ensuring <= max_tokens in each chunk."""
    words = text.split()
    chunks = []
    current_words = []

    for word in words:
        current_words.append(word)
        if count_tokens(" ".join(current_words)) > max_tokens:
            # finalize current chunk
            current_words.pop()
            chunk_str = " ".join(current_words).strip()
            if chunk_str:
                chunks.append(chunk_str)
            current_words = [word]

    # leftover words
    if current_words:
        leftover_str = " ".join(current_words).strip()
        if leftover_str:
            chunks.append(leftover_str)

    return chunks

# --------------------------------------------------------------------------
# 4. EMBEDDING LOGIC (UPDATING PINECONE CORRECTLY)
# --------------------------------------------------------------------------
def embed_qa_pairs(pairs):
    """
    For each QA pair:
      - Embed the text.
      - Ensure answers are structured properly.
      - Store metadata in Pinecone.
    """
    total = len(pairs)
    print(f"🔵 Embedding {total} QA pairs...")

    for idx, qa in enumerate(pairs):
        doc_id = qa.get("id", f"qa_{idx}")
        category = qa.get("category_id", "unknown")
        source = qa.get("source", "unknown")  # "existing" or "extracted"
        is_emergency = qa.get("is_emergency", False)
        question = (qa.get("question") or "").strip()

        # Ensure answer is a structured dictionary
        ans_block = qa.get("answer", {})
        if not isinstance(ans_block, dict):
            ans_block = {
                "main_points": [str(ans_block).strip()],
                "examples": [],
                "tips": [],
                "related_topics": []
            }

        # Combine answer fields into a single string
        main_points = " ".join(ans_block.get("main_points", []))
        examples = " ".join(ans_block.get("examples", []))
        tips = " ".join(ans_block.get("tips", []))
        rel_topics = " ".join(ans_block.get("related_topics", []))
        combined_answer = f"{main_points}\n{examples}\n{tips}\n{rel_topics}".strip()

        if not question or not combined_answer:
            print(f"⚠️ Skipping doc_id='{doc_id}' due to empty question or answer.")
            continue

        full_text = f"Q: {question}\nA: {combined_answer}"
        full_len = count_tokens(full_text)

        # If the text is too long, split it into chunks
        if full_len > 8192:
            text_chunks = split_text_by_tokens(full_text, max_tokens=512)
        else:
            text_chunks = [full_text]

        print(f"   • {idx+1}/{total} => doc_id='{doc_id}', {len(text_chunks)} chunk(s), tokens={full_len}")

        for c_idx, chunk_str in enumerate(text_chunks):
            try:
                # Generate embedding
                resp = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk_str
                )
                vector = resp.data[0].embedding

                chunk_id = f"{doc_id}_chunk{c_idx}"
                index.upsert([
                    {
                        "id": chunk_id,
                        "values": vector,
                        "metadata": {
                            "doc_id": doc_id,
                            "category": category,
                            "source": source,
                            "is_emergency": is_emergency,
                            "text_chunk": chunk_str
                        }
                    }
                ])

                time.sleep(0.2)  # Avoid rate limits

            except Exception as e:
                print(f"❌ Error embedding doc_id={doc_id}, chunk={c_idx}: {e}")

# --------------------------------------------------------------------------
# 5. MAIN - EMBED & SAVE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    embed_qa_pairs(qa_pairs)
    print("✅ Embedding process completed.")
