#!/usr/bin/env python
# coding: utf-8


import asyncio
import os
import sys
import pyaudio
import nest_asyncio
import speech_recognition as sr
from gtts import gTTS
from transformers import pipeline
import streamlit as st
import openai
from openai import AsyncOpenAI
from pinecone import Pinecone

# ✅ Access API keys securely
OPENAI_API_KEY = st.secrets["openai_api_key"]
PINECONE_API_KEY = st.secrets["pinecone_api_key"]

# ✅ Check if API keys are loaded correctly
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found! Check your Streamlit secrets.")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found! Check your Streamlit secrets.")

# ✅ Initialize OpenAI & Pinecone
openai.api_key = OPENAI_API_KEY

# ✅ Define aclient for AsyncOpenAI usage
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("ai-powered-chatbot")

print("✅ API keys loaded successfully!")
print("✅ Pinecone and OpenAI clients initialized!")

# ✅ Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis")

def speak_response(text):
    try:
        audio = pyaudio.PyAudio()
        if audio.get_default_input_device_info():
            # Proceed with microphone input
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source)
                text = recognizer.recognize_google(audio_data)
        else:
            print("No default input device available. Please check your microphone.")
    except OSError as e:
        print(f"Error accessing the microphone: {e}")

# ✅ Generic Intent Responses
GENERIC_INTENTS = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! How can I help you?",
    "how are you": "I'm just a chatbot, but I'm here to help you! What can I do for you?",
    "bye": "Goodbye! Have a great day!",
    "exit": "Goodbye! Have a great day!",
    "quit": "Goodbye! Have a great day!",
}
def detect_generic_intent(query):
    query = query.lower().strip()
    for intent, response in GENERIC_INTENTS.items():
        if intent in query:
            return response
    return None

# ------------------------------------------------------------
# 2. SENTIMENT ANALYSIS
# ------------------------------------------------------------
def detect_sentiment(query):
    result = sentiment_analyzer(query)[0]
    return result['label'].lower()  # e.g. "positive", "negative", or "neutral"

# ------------------------------------------------------------
# 3. RETRIEVE CHUNKS FROM PINECONE
# ------------------------------------------------------------
async def retrieve_chunks(query, top_k=2):
    try:
        response = await aclient.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
        query_embedding = response.data[0].embedding

        # Query Pinecone
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract answers from matches
        return [match.metadata.get("answer", "") for match in result.matches]
    except Exception as e:
        print(f"❌ Error retrieving chunks: {e}")
        return []

# ------------------------------------------------------------
# 4. SPEECH-TO-TEXT
# ------------------------------------------------------------
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)  # Uses Google STT
        print(f"🟢 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("🔴 Sorry, I could not understand the speech.")
        return ""
    except sr.RequestError:
        print("🔴 Could not request results. Check your internet connection.")
        return ""

# ------------------------------------------------------------
# 5. TEXT-TO-SPEECH
# ------------------------------------------------------------
def speak_response(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("start response.mp3")  # Windows-specific approach
    
    # Alternative offline method using pyttsx3:
    # tts_engine.say(text)
    # tts_engine.runAndWait()

# ------------------------------------------------------------
# 6. GENERATE CHATBOT RESPONSE
# ------------------------------------------------------------
async def generate_response(query):
    # Step 1: Check for generic intents
    generic_response = detect_generic_intent(query)
    if generic_response:
        return generic_response  # No sentiment analysis needed

    # Step 2: Detect sentiment (optional usage)
    sentiment = detect_sentiment(query)
    # (You could do something with 'sentiment' if desired.)

    # Step 3: Retrieve relevant chunks
    retrieved_chunks = await retrieve_chunks(query)
    if not retrieved_chunks:
        return "Unfortunately, I couldn't find relevant information. Please try rephrasing your question."

    # Step 4: Create GPT prompt
    context = "\n".join(retrieved_chunks)
    prompt = f"User's question: {query}\n\nRelevant information:\n{context}\n\nAnswer:"

    try:
        gpt_response = await aclient.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.6
        )

        gpt_reply = gpt_response.choices[0].message.content.strip()

        # 🔹 Check for potential cut-off
        if gpt_reply.endswith(("I'm", "but", "and", "because", "These")):
            print("🔹 Response may be cut off. Generating continuation...")
            follow_up = await aclient.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Continue the previous response in a concise manner."},
                    {"role": "user", "content": "Continue from where you left off."}
                ],
                max_tokens=200,
                temperature=0.6
            )
            gpt_reply += " " + follow_up.choices[0].message.content.strip()

        return gpt_reply

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Unfortunately, I couldn't generate a response. Please try again."

# ------------------------------------------------------------
# 7. MAIN FUNCTION: VOICE & TEXT SUPPORT
# ------------------------------------------------------------
async def test_chatbot():
    print("\n🔵 Welcome to the University Student Support Service!")
    print("🔹 Speak or type your queries. Say 'exit' or 'quit' to end the conversation.\n")

    while True:
        use_voice = input("\n🟢 Press Enter to speak or type your message: ")
        if use_voice == "":
            query = recognize_speech()
            voice_mode = True
        else:
            query = use_voice
            voice_mode = False  # user typed manually
        
        if query.lower() in ["exit", "quit"]:
            print("🔴 Chatbot: Goodbye!")
            break

        response = await generate_response(query)
        print(f"\n🔵 Chatbot: {response}")

        # 🔹 Speak the response aloud **ONLY if the user spoke**
        if voice_mode:
            speak_response(response)

# ------------------------------------------------------------
# 8. ASYNC LOOP SETUP
# ------------------------------------------------------------
nest_asyncio.apply()  # Fixes async loop issues in some environments

if __name__ == "__main__":
    if sys.platform.startswith("win"):  # Windows fix
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.get_running_loop().run_until_complete(test_chatbot())
    except RuntimeError:
        asyncio.run(test_chatbot())


# In[ ]:




