
# coe for the rag chatbot
# frontend.py
import streamlit as st
import requests

st.set_page_config(page_title="Doc Chatbot", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

API_BASE = "http://127.0.0.1:8000"
ASK_URL = f"{API_BASE}/ask"
UPLOAD_URL = f"{API_BASE}/upload"

st.title("📘 Doc Chatbot UI")

# -----------------------
# Upload section (top)
# -----------------------
st.markdown("### Upload a PDF to chat with its content")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.info(f"Selected file: {uploaded_file.name} ({uploaded_file.type})")
    if st.button("Upload and Index Document"):
        with st.spinner("Uploading and indexing... (this may take a while for large PDFs)"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                resp = requests.post(UPLOAD_URL, files=files, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Indexed successfully: {data.get('chunks_indexed', '?')} chunks.")
                else:
                    st.error(f"Upload failed: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Upload request failed: {e}")

st.markdown("---")

# -----------------------
# Chat UI
# -----------------------
chat_container = st.container()

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question here...")
    submit_btn = st.form_submit_button("Send")

    if submit_btn and user_input.strip():
        try:
            with st.spinner("🤖 Generating the answer..."):
                response = requests.post(
                    ASK_URL,
                    json={"question": user_input},
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                st.session_state.history.append({"q": user_input, "a": answer})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")
    elif submit_btn:
        st.warning("Please enter a question!")

# Display chat history above input
with chat_container:
    for item in reversed(st.session_state.history):  # show newest first
        st.markdown(f"**🧑 You:** {item['q']}")
        st.markdown(f"**🤖 Bot:** {item['a']}")
        st.markdown("---")
    


