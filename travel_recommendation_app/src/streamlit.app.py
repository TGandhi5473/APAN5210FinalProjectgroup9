import streamlit as st
from shared_memory import SharedMemory
from agents.agents import WebSearchAgent, RecommenderAgent, SummarizerAgent, TranslationAgent
from dotenv import load_dotenv
import os

load_dotenv()  # This will read .env file

# Now you can access your key
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize shared memory
shared_memory = SharedMemory()

# Initialize agents
web_agent = WebSearchAgent(shared_memory)
recommender_agent = RecommenderAgent(shared_memory)
summarizer_agent = SummarizerAgent(shared_memory)
translation_agent = TranslationAgent(shared_memory)

st.title("🌍 Travel Recommendation App")

# User Inputs
user_input = st.text_input("Enter your travel preferences or questions:")
target_language = st.selectbox("Select output language:", ["English", "French", "Spanish", "German", "Chinese"])

if st.button("Get Recommendations"):
    if not user_input:
        st.error("Please enter your travel preferences.")
    else:
        # Store inputs in shared memory
        shared_memory.set("user_input", user_input)
        shared_memory.set("target_language", target_language)

        # Execute agents in sequence
        web_agent.execute()
        recommender_agent.execute()
        summarizer_agent.execute()
        translation_agent.execute()

        # Retrieve and display results
        summary = shared_memory.get("summary")
        translated_summary = shared_memory.get("translated_summary")

        st.subheader("Travel Summary (English)")
        if summary:
            st.write(summary.content)
        else:
            st.write("No summary available.")

        if target_language != "English":
            st.subheader(f"Travel Summary ({target_language})")
            if translated_summary:
                st.write(translated_summary.content)
            else:
                st.write("No translated summary available.")
        else:
            st.info("You selected English, so no translation was applied.")
