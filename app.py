import logging
import os
import time
import warnings
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from common_functions import create_chain, process_chat, retrieving_data_from_db

warnings.filterwarnings("ignore")

load_dotenv()

# creating logs folder if it doesn't exits
if not os.path.exists(f"{os.getcwd()}/logs/chatbot_logs"):
    os.makedirs(f"{os.getcwd()}/logs/chatbot_logs")

logging.basicConfig(
    filename=f"{os.getcwd()}/logs/chatbot_logs/app_{datetime.today().date()}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# function to streaming the AI response
def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)


st.set_page_config(
    page_title="Wilson's Disease GPT.",
)

st.title("🩺 Wilson's disease chatbot")

# defining the logic to get user input and generating output
try:
    logging.info("Chatbot initiated.")
    retriever = (
        retrieving_data_from_db()
    )  # creating retriever to retrieve data from vector DB
    logging.info("Retriever initiated.")
    chain = create_chain(
        retriever
    )  # creating chain for capturing historical context with retriever
    logging.info("Chain initiated.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content="Hello, I am a professional wilson's disease consultant. How can i help you today?"
            )
        ]
    logging.info("Chat_history initiated.")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    if prompt := st.chat_input(placeholder="Enter prompt"):
        logging.info("User added info.")
        st.chat_message("user").write(prompt)
        response = process_chat(chain, prompt, st.session_state.chat_history)
        logging.info("AI response generated.")
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.chat_message("assistant").write_stream(stream_data(response))
        logging.info("One request-response completed.")

except Exception as e:
    st.text("Error: ", e)
    logging.error(e)
