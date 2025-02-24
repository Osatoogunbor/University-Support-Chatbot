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
from pinecone import Pinecone  # ✅ Corrected import statement
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
pc = Pinecone(api_key=PINECONE_API_KEY)  # ✅ Corrected initialization
index = pc.Index("ai-powered-chatbot")  # ✅ Ensures proper index usage

# -------------------------------------------------------------------------
# 2. SET PAGE CONFIG FIRST
# -------------------------------------------------------------------------
st.set_page_config(page_title="UniEase Chatbot", layout="wide")

# -------------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION
# -------------------------------------------------------------------------
st.sidebar.title("UniEase: University 24/7 Assistant")
st.sidebar.markdown(
    """
Welcome to **UniEase**—your university wellbeing companion!

- Receive accurate, concise support tailored to help you navigate university life.
- Ask questions about enrollment, extensions, deadlines, and university resources.
- Access mental health resources and strategies for managing academic stress.
""",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# 4. SENTIMENT ANALYSIS
# -------------------------------------------------------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

def detect_sentiment(query: str) -> str:
    result = sentiment_analyzer(query)[0]
    return result['label'].lower()

# -------------------------------------------------------------------------
# 5. FIXED FUNCTION ORDER
# -------------------------------------------------------------------------

GENERIC_INTENTS = {
    # Basic greetings (Mixing plain and emoji responses)
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! How can I help? 👋",
    "hey": "Hey! What do you need help with?",
    "good morning": "Good morning! How can I assist?",
    "good afternoon": "Good afternoon! What do you need help with?",
    "good evening": "Good evening! How can I assist? 🌙",
    "sup": "Hey! What's up?",
    "yo": "Yo! How can I help? 😎",
    "hiya": "Hiya! What can I do for you?",
    "heyy": "Heyy! What’s up?",
    "hello there": "Hello there!",
    "what's up": "Hey! How’s it going?",

    # Ways people might call UniEase
    "UniEase": "Hello, I'm UniEase, your University Student Support Chatbot. How can I assist you today?",
    "UniEase bot": "Yes! I’m UniEase, your AI assistant. How can I help?",
    "UniEase assistant": "I’m here to help! How can I assist you today?",
    "hey UniEase": "Hey! What do you need help with?",
    "hello UniEase": "Hello! I’m listening. How can I assist?",

    # Common farewells (Mixing plain and emoji responses)
    "bye": "Goodbye! Have a great day! 👋",
    "goodbye": "Goodbye! Take care!",
    "see you": "See you next time!",
    "later": "Catch you later! ✌️",
    "peace": "Peace out! ✌️",
    "quit": "Goodbye! See you next time!",
    "exit": "Goodbye! Take care!",

    # Appreciation & Thanks (Keeping these plain)
    "thank you": "You're welcome! I'm always here to help!",
    "thanks": "No problem! Let me know if you need anything else.",
    "thx": "You're welcome!",
    "appreciate it": "Glad I could help!",

    # Emoji-based responses
    "👋": "Hello! How can I assist you today?",
    "🤗": "Aww, sending you a virtual hug! 🤗 How can I help?",
    "😊": "You seem happy! How can I assist you today?",
    "😃": "Great energy! What do you need help with?",
    "😢": "Oh no! What’s wrong? I’m here to help.",
    "😞": "I hear you. Tell me what's bothering you.",
    "😔": "I’m here for you. What can I do to help?",
    "😡": "I sense some frustration. Want to talk about it?",
    "🤬": "Yikes! What happened? Maybe I can help?",
    "❤️": "Aww, thank you! ❤️ How else can I assist you?",
    "💕": "Sending good vibes your way! 💕 How can I help?",
    "🤍": "I appreciate your kindness! 🤍 How can I support you?"
}


# Define detect_generic_intent BEFORE retrieve_chunks()
def detect_generic_intent(query: str) -> Optional[str]:
    """Detects common greetings, farewells, appreciation, and emoji-based intents."""
    return GENERIC_INTENTS.get(query.strip().lower())  # Matches case-insensitive text & emoji

# Define truncate_chunk BEFORE retrieve_chunks()
def truncate_chunk(text: str, max_chars: int = 600) -> str:
    """Truncates text if it exceeds a character limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"

# -------------------------------------------------------------------------
# 6. RETRIEVING RELEVANT CHUNKS FROM PINECONE
# -------------------------------------------------------------------------
async def retrieve_chunks(query: str, top_k: int = 5) -> List[dict]:
    try:
        embedding_resp = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = embedding_resp.data[0].embedding

        pinecone_result = index.query(  # ✅ Updated to use the correct Pinecone package
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

# -------------------------------------------------------------------------
# 7. GENERATING GPT RESPONSE
# -------------------------------------------------------------------------
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
# 8. MAIN CHATBOT FUNCTION
# -------------------------------------------------------------------------
def main():
    st.title("🎓 UniEase: Your University Study & Wellbeing Companion")
    st.markdown("Ask me about university life, deadlines, university resources, or any academic stress concerns!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(generate_response(user_input))
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        st.rerun()

if __name__ == "__main__":
    main()
