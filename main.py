# #==================================================================================================
# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import logging
import shutil
import uuid
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("chatbot_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Docs Chatbot")

# Configuration / Globals
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_MODEL_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_MODEL_DEPLOYMENT")

if not (AZURE_ENDPOINT and AZURE_API_KEY and EMBEDDING_DEPLOYMENT and CHAT_DEPLOYMENT):
    logger.warning("One or more Azure environment variables are missing. Make sure .env is configured.")

# Build embeddings and LLM objects once
logger.info("Initializing Azure OpenAI Embeddings and chat model...")
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    deployment=EMBEDDING_DEPLOYMENT,
)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    deployment_name=CHAT_DEPLOYMENT,
    temperature=0.3
)

# global FAISS DB + retriever (initialized empty)
faiss_db = None
retriever = None

# Try to load existing local DB if present
DB_DIR = "faiss_vector_db"
try:
    if os.path.exists(DB_DIR):
        logger.info("Found existing FAISS DB on disk. Loading...")
        faiss_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever(search_kwargs={"k": 8})
        logger.info("Loaded existing FAISS DB.")
except Exception as e:
    logger.error(f"Failed to load existing FAISS DB: {e}")


# ------------------------------
# Utility: index a document file path
# ------------------------------
def index_document(file_path: str):
    """
    Load PDF, split into chunks, embed, create FAISS DB, save it,
    update global faiss_db and retriever.
    """
    global faiss_db, retriever

    logger.info(f"Indexing document: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")

        # create FAISS db
        faiss_db = FAISS.from_documents(chunks, embeddings)
        # save locally for persistence
        faiss_db.save_local(DB_DIR)
        # create retriever for runtime use
        retriever = faiss_db.as_retriever(search_kwargs={"k": 8})
        logger.info("Indexing complete and retriever updated.")
        return True, len(chunks)
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return False, 0


# ------------------------------
# Upload endpoint
# ------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF file, saves it, indexes it (embeddings + FAISS),
    and updates the retriever used by /ask.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # save file to a temporary location
    upload_id = str(uuid.uuid4())
    tmp_dir = "uploaded_files"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{upload_id}_{file.filename}")

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")

    # index it (blocking; for large files you might want an async/background approach)
    success, chunks_count = index_document(tmp_path)
    if not success:
        raise HTTPException(status_code=500, detail="Indexing failed. See logs.")

    return {"status": "ok", "message": "File uploaded and indexed.", "chunks_indexed": chunks_count}


# ------------------------------
# API for question answering
# ------------------------------
class QueryRequest(BaseModel):
    question: str


def classify_intent(question: str, llm) -> str:
    prompt = f"""
    Classify this message into exactly ONE category:
    - greeting
    - query
    - other

    User message: "{question}"

    Return only the category name.
    """
    msg = HumanMessage(content=prompt)
    try:
        response = llm.generate([[msg]])
        intent = response.generations[0][0].text.strip().lower()
    except Exception as e:
        logger.error(f"LLM error during intent classification: {e}")
        intent = "other"
    logger.info(f"Intent Classification: {intent}")
    return intent


def rewrite_query(question: str, llm) -> str:
    prompt = f"""
    Reformulate the user question into a short list of the most important keywords 
    for a document search. Keep it fully domain-agnostic.

    Question: {question}
    Keywords:
    """
    msg = HumanMessage(content=prompt)
    try:
        response = llm.generate([[msg]])
        rewritten = response.generations[0][0].text.strip()
    except Exception as e:
        logger.error(f"LLM error during query rewriting: {e}")
        rewritten = question
    logger.info(f"Rewritten Query: {rewritten}")
    return rewritten


@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question
    logger.info(f"Received Question: {question}")

    # 1. Detect Intent
    intent = classify_intent(question, llm)
    logger.info(f"Received intent: {intent}")

    if intent == "greeting":
        return {"answer": "Hello! 👋 I am your Documents Chatbot. How can I help you?", "total_documents": 0}

    if intent == "query":
        if retriever is None:
            logger.warning("No retriever available. User must upload a document first.")
            return {"answer": "No documents indexed yet. Please upload a PDF to chat with.", "total_documents": 0}

        search_query = rewrite_query(question, llm)
        logger.info(f"Search query: {search_query}")

        # Use the retriever public method get_relevant_documents when available
        try:
            docs = retriever.get_relevant_documents(search_query)
        except Exception:
            # fall back to protected method if necessary (not ideal)
            docs = retriever._get_relevant_documents(search_query, run_manager=None)

        logger.info(f"Retrieved {len(docs)} documents.")
        if docs:
            sample_text = docs[0].page_content[:200].replace("\n", " ")
            logger.info(f"Sample Retrieved Content: {sample_text}")

        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
        You are a Retrieval-Augmented Generation (RAG) assistant.

        RULES:
        1. Use ONLY the information found in the context documents.
        2. If the question has multiple parts:
        - Answer the parts that ARE supported by the documents.
        - For the parts NOT supported by the documents, clearly say:
            "I could not find the answer to '<missing part>' in the provided Documents."
        3. NEVER give an empty answer.
        4. NEVER add information that is not in the context.

        Context Documents:
        {context}

        User Question:
        {question}

        Provide a combined answer following the rules.
        """
        msg = HumanMessage(content=prompt)
        try:
            result = llm.generate([[msg]])
            answer = result.generations[0][0].text.strip()
            if not answer:
                answer = "I could not find the answer in the provided Documents."
        except Exception as e:
            logger.error(f"LLM error during answer generation: {e}")
            answer = "I could not generate an answer due to an internal issue."

        logger.info(f"Final Answer: {answer[:200]}..")
        return {"answer": answer, "total_documents": len(docs)}

    # fallback
    return {"answer": "I could not find the answer in the provided Documents.", "total_documents": 0}


@app.get("/")
def root():
    return {"status": "Universal RAG Chatbot is running!"}
