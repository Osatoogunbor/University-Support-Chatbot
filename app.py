#!/usr/bin/env python
# coding: utf-8

import json
import asyncio
import streamlit as st
import speech_recognition as sr
import pyttsx3
from openai import AsyncOpenAI  # <-- IMPORTANT: Keep your AsyncOpenAI import
from pinecone import Pinecone
from transformers import pipeline
from gtts import gTTS
import os
import time
import streamlit.components.v1 as components

import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Access API keys securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Check if API keys are loaded correctly
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found! Check .env file.")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found! Check .env file.")

# ✅ Initialize OpenAI & Pinecone
openai.api_key = OPENAI_API_KEY

# <-- ADDED: define 'aclient' for embedding & chat calls
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)  # ✅ Pinecone initialization
index = pc.Index("ai-powered-chatbot")

st.write("✅ API keys loaded successfully!")

# ✅ Load Sentiment Analysis Model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# ✅ Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()

import sounddevice as sd
import soundfile as sf

def speak_response(text):
    try:
        filename = f"response_{int(time.time())}.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(filename)

        # ✅ Play audio inside Streamlit without opening a media player
        data, samplerate = sf.read(filename)
        sd.play(data, samplerate)
        sd.wait()  # Ensure the audio fully plays before continuing

    except Exception as e:
        st.error(f"❌ Error playing response: {e}")

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

# ✅ Function to Detect Sentiment
def detect_sentiment(query):
    result = sentiment_analyzer(query)[0]
    return result['label'].lower()

# ✅ Retrieve Relevant Chunks from Pinecone
async def retrieve_chunks(query, top_k=3):
    try:
        if not query or not isinstance(query, str):
            return []

        response = await aclient.embeddings.create(
            model="text-embedding-ada-002",
            input=[query.strip()]
        )
        query_embedding = response.data[0].embedding

        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return [match.metadata.get("answer", "") for match in result.matches]
    except Exception as e:
        st.error(f"❌ Error retrieving chunks: {e}")
        return []

# ✅ Function to Convert Speech to Text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🟢 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("🔴 Sorry, I could not understand the speech.")
        return ""
    except sr.RequestError:
        st.error("🔴 Could not request results. Check your internet connection.")
        return ""

# ✅ Generate Response
async def generate_response(query):
    generic_response = detect_generic_intent(query)
    if generic_response:
        return generic_response

    sentiment_task = asyncio.to_thread(detect_sentiment, query)
    retrieval_task = retrieve_chunks(query)

    sentiment, retrieved_chunks = await asyncio.gather(sentiment_task, retrieval_task)

    if not retrieved_chunks:
        return "Unfortunately, I couldn't find relevant information. Please try rephrasing your question."

    context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are a university support chatbot. Answer user queries using the provided relevant information. 
    Only use the retrieved information and do not add extra knowledge unless necessary.

    User Question: {query}

    Relevant Information from Knowledge Base:
    {context}

    Provide a detailed but concise response based on the above information.
    """

    try:
        response_text = ""
        # Using GPT-3.5-turbo as an example
        gpt_response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                    "You are a university support chatbot. Your responses should be well-structured, informative, and engaging. "
                    "Expand on key points, avoid generic responses, and ensure clarity. "
                    "If discussing study techniques, provide examples or step-by-step guidance. "
                    "Use full sentences rather than short bullet points unless specifically requested."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
            stream=True
        )

        async for chunk in gpt_response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        return response_text

    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Function to display link cards properly
def display_link_card(title, description, image_url, link):
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 12px; padding: 15px; text-align: center; background-color: #f9f9f9; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); width: 100%; max-width: 300px;">
            <a href="{link}" target="_blank" style="text-decoration: none;">
                <img src="{image_url}" width="100%" style="border-radius: 8px; margin-bottom: 10px;">
                <h4 style="margin-bottom: 5px; color: #1A5276; font-size: 16px;">{title}</h4>
                <p style="margin: 0px; font-size: 13px; color: #555;">{description}</p>
                <button style="background-color: #1E3A8A; color: white; padding: 6px 10px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; margin-top: 8px;">
                    Visit
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ✅ Main Streamlit UI
def main():
    st.markdown(
        """
        <style>
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #103c84;
                text-align: center;
                margin-bottom: 20px;
                text-transform: uppercase;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="title">🎓 University Student Support Chatbot</div>', unsafe_allow_html=True)
    st.write("🔹 Speak or type your queries below. Click **'Start Voice Input'** to speak.")

    # ✅ Chat history container
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.container():
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.divider()

    # ✅ Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input.strip()})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(generate_response(user_input.strip()))

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.rerun()

    st.divider()
    if st.button("🎤 Start Voice Input", key="voice_button"):
        voice_query = recognize_speech()

        if voice_query:
            st.session_state["messages"].append({"role": "user", "content": voice_query})

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(generate_response(voice_query))

            st.session_state["messages"].append({"role": "assistant", "content": response})
            speak_response(response)

            st.rerun()

    st.divider()
    st.subheader("🔗 Useful Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        display_link_card(
            title="Mind - Mental Health Support",
            description="Get expert advice and resources on managing mental health.",
            image_url="https://1000logos.net/wp-content/uploads/2021/12/Mind-Logo.png",
            link="https://www.mind.org.uk/"
        )

    with col2:
        display_link_card(
            title="Pomodoro Study Technique (YouTube)",
            description="Learn how to use the Pomodoro technique for effective studying.",
            image_url="https://img.youtube.com/vi/mNBmG24djoY/0.jpg",
            link="https://www.youtube.com/watch?v=mNBmG24djoY"
        )

    with col3:
        display_link_card(
            title="University of Wolverhampton Support",
            description="Explore student support services at your university.",
            image_url="https://www.wlv.ac.uk/media/departments/womenx27s-staff-network/Accessibility,-Disability-and-Inclusion-Content-Block-Third.jpg",
            link="https://www.wlv.ac.uk/university-life/student-life/"
        )

if __name__ == "__main__":
    main()
