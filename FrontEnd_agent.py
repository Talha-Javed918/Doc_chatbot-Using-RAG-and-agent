# code for the csv_agent.py 
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CSV Query Chatbot", layout="wide")
st.title("CSV Query Chatbot")

# -------------------------------
# Session State for Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# CSV Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    files = {"file": uploaded_file}

    with st.spinner("Uploading CSV..."):
        response = requests.post(f"{API_URL}/upload", files=files)

    if response.status_code == 200:
        st.success("CSV uploaded successfully")
    else:
        st.error("CSV upload failed")

# -------------------------------
# Display Previous Messages
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# User Input
# -------------------------------
question = st.chat_input("Enter your question about the CSV")

if question:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -------------------------------
    # Agent Response with Spinner
    # -------------------------------
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": question}
        )

    if response.status_code == 200:
        answer = response.json()["response"]
    else:
        answer = f"Error: {response.text}"

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
