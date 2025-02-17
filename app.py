#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py

Lightweight version of UniEase:
 - Removes the large BART summarization to prevent heavy memory usage
 - Uses a simple text truncation for long chunks
"""

from typing import Optional, List
import asyncio
import streamlit as st
import openai
from openai import AsyncOpenAI
from pinecone import Pinecone
from transformers import pipeline

# Grab secrets from st.secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]
PINECONE_API_KEY = st.secrets["pinecone_api_key"]

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in st.secrets.")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in st.secrets.")

openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ai-powered-chatbot")

# -------------------------------------------------------------------------
# 2. SET PAGE CONFIG FIRST
# -------------------------------------------------------------------------
st.set_page_config(page_title="UniEase Chatbot", layout="wide")

# -------------------------------------------------------------------------
# 3. OPTIONAL: ADD CSS STYLING FOR SIDEBAR
# -------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Make the sidebar title bigger and more noticeable */
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #333333;
        margin-bottom: 0.5em;
    }
    /* Larger font for sidebar text, paragraphs and list items */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] div {
        font-size: 1.15rem !important;
        line-height: 1.5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# 4. SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.title("UniEase: University 24/7 Assistant")
st.sidebar.markdown(
    """
Welcome to **UniEase**—your university wellbeing companion!

- Receive accurate, concise support tailored to help you navigate university life.
- Ask questions about enrollment, extensions, deadlines and university resources.
- Access mental health resources and strategies for managing academic stress.
""",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# 5. SENTIMENT ANALYSIS, ETC.
# -------------------------------------------------------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

def detect_sentiment(query: str) -> str:
    result = sentiment_analyzer(query)[0]
    return result['label'].lower()

def truncate_chunk(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"

GENERIC_INTENTS = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! How can I help?",
    "hey": "Hey! What do you need help with?",
    "good morning": "Good morning! How can I assist?",
    "bye": "Goodbye! Have a great day!",
    "exit": "Goodbye! Take care!",
    "quit": "Goodbye! See you next time!",
    "uniease": "Hello, I'm UniEase, your University Student Support Chatbot. How can I assist you today?"
}

def detect_generic_intent(query: str) -> Optional[str]:
    return GENERIC_INTENTS.get(query.strip().lower())

async def retrieve_chunks(query: str, top_k: int = 5) -> List[dict]:
    try:
        embedding_resp = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = embedding_resp.data[0].embedding

        pinecone_result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        if not pinecone_result.matches:
            return []

        retrieved_chunks = []
        for match in pinecone_result.matches:
            meta = match.metadata or {}
            chunk_text = meta.get("text_chunk", "").strip()
            category = meta.get("category", "unknown").strip()
            source = meta.get("source", "unknown")
            is_emergency = meta.get("is_emergency", False)

            if chunk_text:
                truncated_text = truncate_chunk(chunk_text, max_chars=600)
                retrieved_chunks.append({
                    "text": truncated_text,
                    "category": category,
                    "source": source,
                    "is_emergency": is_emergency,
                    "score": match.score
                })

        emergency_chunks = [c for c in retrieved_chunks if c["is_emergency"]]
        if emergency_chunks:
            return sorted(emergency_chunks, key=lambda x: -x["score"])

        sorted_chunks = sorted(retrieved_chunks, key=lambda x: -x["score"])
        return sorted_chunks

    except Exception as e:
        st.error(f"❌ Retrieval Error: {e}")
        return []

async def generate_response(user_query: str, top_k: int = 5) -> str:
    greeting_reply = detect_generic_intent(user_query)
    if greeting_reply:
        return greeting_reply

    context_chunks = await retrieve_chunks(user_query, top_k=top_k)
    if not context_chunks:
        return "I couldn't find relevant information. Could you rephrase your question or provide more details?"

    unique_chunks = list({c["text"]: c for c in context_chunks}.values())
    combined_context = "\n\n---\n\n".join([c["text"] for c in unique_chunks])

    # Use GPT-4 Turbo exclusively
    chosen_model = "gpt-4-turbo"

    system_message = (
        "You are UniEase, a University Student Support Chatbot responsible for providing accurate and concise answers to student inquiries. "
        "When a question is straightforward or requires only brief clarification, answer in one or two succinct sentences. "
        "If the question is complex or the context requires further explanation, provide a detailed, well-structured response using clear paragraphs or bullet points. "
        "Do not include unnecessary details; focus only on the key information required. "
        "If you lack sufficient detail, politely ask for clarification. "
        "If the query is urgent (e.g., mental health or emergencies), acknowledge the urgency and prioritize your response. "
        "Remember, your primary audience is University of Wolverhampton students."
    )

    user_prompt = (
        f"User's question:\n{user_query}\n\n"
        "Relevant context from your knowledge base:\n"
        f"{combined_context}\n\n"
        "Please provide an answer that is concise if the question is simple, but detailed and well-structured if the question is complex."
    )

    try:
        chat_response = await client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=350,   # Reduced token limit for faster responses
            temperature=0.7,  # Lower randomness
            top_p=0.5
        )
        final_answer = chat_response.choices[0].message.content.strip()
        return final_answer

    except Exception as e:
        st.error(f"❌ Error generating GPT response: {e}")
        return "Oops, something went wrong."

# -------------------------------------------------------------------------
# MAIN CHATBOT FUNCTION
# -------------------------------------------------------------------------
def main():
    st.title("🎓 UniEase: Your University Study & Wellbeing Companion")
    st.markdown("Ask me about university life, deadlines, university resources, or any academic stress concerns!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(generate_response(user_input))
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        st.rerun()

    st.divider()

if __name__ == "__main__":
    main()
