from transformers import pipeline
import asyncio
import streamlit as st
import openai
from openai import AsyncOpenAI
from pinecone import Pinecone



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

print("✅ Pinecone index connected successfully!\n")

# ✅ Load Sentiment Analysis Model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# --------------------------------------------------------------------------
# 3. DETECT GENERIC GREETINGS (INTENTS)
# --------------------------------------------------------------------------
GENERIC_INTENTS = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! How can I help?",
    "hey": "Hey! What do you need help with?",
    "good morning": "Good morning! How can I assist?",
    "bye": "Goodbye! Have a great day!",
    "exit": "Goodbye! Take care!",
    "quit": "Goodbye! See you next time!"
}

def detect_generic_intent(query: str) -> str | None:
    return GENERIC_INTENTS.get(query.strip().lower())

# ✅ Function to Detect Sentiment
def detect_sentiment(query):
    result = sentiment_analyzer(query)[0]
    return result['label'].lower()

# --------------------------------------------------------------------------
# 4. IMPROVED RETRIEVAL PROCESS WITH RERANKING
# --------------------------------------------------------------------------
async def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    1. Embed the user's query.
    2. Query Pinecone for top_k matches.
    3. Extract relevant text and emergency status.
    4. Apply reranking for better retrieval.
    """
    try:
        # Embed the query
        embedding_response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = embedding_response.data[0].embedding

        # Query Pinecone
        result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        if not result.matches:
            print("⚠️ No relevant matches found in Pinecone.")
            return []

        retrieved_chunks = []
        for match in result.matches:
            meta = match.metadata or {}
            chunk_text = meta.get("text_chunk", "").strip()
            category = meta.get("category", "unknown").strip()
            source = meta.get("source", "unknown")
            is_emergency = meta.get("is_emergency", False)

            if chunk_text:
                retrieved_chunks.append({
                    "text": chunk_text,
                    "category": category,
                    "source": source,
                    "is_emergency": is_emergency,
                    "score": match.score  # Use similarity score for reranking
                })

        # Prioritize emergency responses if found
        emergency_chunks = [c for c in retrieved_chunks if c["is_emergency"]]
        if emergency_chunks:
            print("⚠️ Emergency-related query detected! Prioritizing emergency responses.")
            return sorted(emergency_chunks, key=lambda x: -x["score"])

        # Reranking: Sort by score and source balance
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: -x["score"])
        print(f"✅ Retrieved {len(sorted_chunks)} chunks from Pinecone after reranking.")
        return sorted_chunks

    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        return []

# --------------------------------------------------------------------------
# 5. GENERATE A CHATBOT RESPONSE WITH IMPROVED CONTEXT
# --------------------------------------------------------------------------
async def generate_response(user_query: str, top_k: int = 5) -> str:
    """
    - Checks for generic greetings first.
    - Retrieves and reranks context.
    - Uses improved GPT prompting.
    """

    # A. Handle generic greetings
    greeting_reply = detect_generic_intent(user_query)
    if greeting_reply:
        return greeting_reply

    # B. Retrieve and rerank context
    context_chunks = await retrieve_chunks(user_query, top_k=top_k)
    if not context_chunks:
        return "I couldn't find relevant info. Could you try rephrasing your question?"

    # Remove duplicate or overly generic responses
    unique_chunks = list({c["text"]: c for c in context_chunks}.values())

    # Combine for GPT prompt
    combined_context = "\n\n---\n\n".join([c["text"] for c in unique_chunks])

    # C. Construct a more effective prompt for GPT
    system_message = (
        "You are a University Student Support Chatbot. Use the retrieved information below "
        "to answer the question accurately and concisely. Do NOT make up answers. "
        "Prioritize emergency responses when necessary. Avoid repetition."
    )

    user_prompt = (
        f"User's question:\n\n{user_query}\n\n"
        f"Relevant context:\n\n{combined_context}\n\n"
        "Please provide a clear, concise, and relevant response."
    )

    try:
        chat_response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.8,  # Slightly higher temperature to encourage more natural responses
            top_p=0.5  # Adjust top-p to allow for slight randomness and avoid repetitive responses
        )
        final_answer = chat_response.choices[0].message.content.strip()
        return final_answer

    except Exception as e:
        print(f"❌ Error generating GPT response: {e}")
        return "Oops, something went wrong."

# --------------------------------------------------------------------------
# 6. MAIN CHATBOT LOOP
# --------------------------------------------------------------------------
async def main_chat_loop():
    print("\n🔵 Welcome to the University Student Support Chatbot!")
    print("🔹 Type your questions below. Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("\n🟢 Your message: ").strip()
        if not user_input:
            print("⚠️ No input detected. Please try again.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("🔴 Chatbot: Goodbye!")
            break

        # Generate an answer
        answer = await generate_response(user_input, top_k=5)
        print(f"\n🔵 Chatbot: {answer}")

# --------------------------------------------------------------------------
# 7. RUN (ASYNC)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main_chat_loop())
