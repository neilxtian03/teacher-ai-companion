# ==============================================================================
# FINAL (v5), CONTEXT-AWARE HYBRID AI AGENT
# ==============================================================================
import streamlit as st
import os
import uuid
import threading
import time
import base64
import fitz
from unstructured.partition.pdf import partition_pdf

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# --- Global Variables & Constants ---
ACTIVE_SESSIONS = {}
SESSION_LOCK = threading.Lock()
MAX_CONCURRENT_USERS = 30
MAX_QUESTIONS_PER_USER = 20
SESSION_TIMEOUT_SECONDS = 300

# --- Core Processing Functions ---
def process_multimodal_pdfs(pdf_files, api_key):
    all_content_chunks = []
    image_description_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
    temp_dir = "temp_pdf_files"; os.makedirs(temp_dir, exist_ok=True)
    for pdf_file in pdf_files:
        file_path = os.path.join(temp_dir, pdf_file.name);
        with open(file_path, "wb") as f: f.write(pdf_file.getbuffer())
        st.info(f"Analyzing text & tables in {pdf_file.name}...")
        elements = partition_pdf(filename=file_path, strategy="hi_res", infer_table_structure=True, model_name="yolox")
        for el in elements:
            if "unstructured.documents.elements.Table" in str(type(el)): all_content_chunks.append(f"Table Content:\n{el.metadata.text_as_html}\n")
            else: all_content_chunks.append(el.text)
        st.info(f"Analyzing images in {pdf_file.name}...");
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                for img in doc.get_page_images(page_num):
                    xref, base_image = img[0], doc.extract_image(img[0])
                    message = HumanMessage(content=[{"type": "text", "text": "Describe this image in detail..."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(base_image['image']).decode()}"}}])
                    desc = image_description_model.invoke([message]).content
                    all_content_chunks.append(f"Image Description (page {page_num + 1}):\n{desc}\n")
        except Exception as e: st.warning(f"Could not process an image in {pdf_file.name}. Error: {e}")
    return all_content_chunks

def get_text_chunks(content_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.create_documents(content_list)

def get_vector_store(documents, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_documents(documents, embedding=embeddings)

# --- AI Chains ---
def get_router_chain(api_key):
    prompt_template = """You are an expert routing agent. Based on the DOCUMENT CONTEXT and the USER'S QUESTION, your task is to choose the best tool.
    Here are your tools:
    1.  **DOCUMENT_SEARCH**: Use this tool if the question is asking directly about the content, events, or details *found within the provided DOCUMENT CONTEXT*.
    2.  **GENERAL_KNOWLEDGE_SEARCH**: Use this tool **ONLY IF** the question asks for a definition or explanation of a specific term or concept that is **explicitly mentioned in the DOCUMENT CONTEXT**.
    3.  **IRRELEVANT**: Use this tool if the USER'S QUESTION has **no connection** to the topics or terms found in the DOCUMENT CONTEXT.
    ---
    DOCUMENT CONTEXT:\n{context}\n---
    USER'S QUESTION:\n{question}\n---
    Based on the rules, which tool should be used? Answer with only the tool name: DOCUMENT_SEARCH, GENERAL_KNOWLEDGE_SEARCH, or IRRELEVANT.
    Tool:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=model, prompt=prompt)

def get_document_qa_chain(api_key):
    prompt_template = "You are an expert teaching assistant... (full bilingual prompt)"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def get_general_knowledge_chain(api_key):
    prompt_template = "You are a helpful and knowledgeable teaching assistant... (full bilingual prompt)"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return LLMChain(llm=model, prompt=prompt)

def get_synthesis_chain(api_key):
    prompt_template = "You are a master teaching assistant... (full synthesis prompt)"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "general_answer", "document_context"])
    return LLMChain(llm=model, prompt=prompt)

# --- Streamlit App UI and Main Logic ---
st.set_page_config(page_title="Hybrid AI Companion", layout="wide")
st.header("📚 Hybrid AI Companion")

if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'question_count' not in st.session_state: st.session_state.question_count = 0

def manage_concurrency():
    with SESSION_LOCK:
        current_time = time.time()
        expired_sessions = [sid for sid, t in ACTIVE_SESSIONS.items() if current_time - t > SESSION_TIMEOUT_SECONDS]
        for sid in expired_sessions: del ACTIVE_SESSIONS[sid]
        if st.session_state.session_id not in ACTIVE_SESSIONS and len(ACTIVE_SESSIONS) >= MAX_CONCURRENT_USERS:
            st.warning(f"Chatbot at max capacity ({MAX_CONCURRENT_USERS} users). Please try again in a few minutes.")
            st.stop()
        ACTIVE_SESSIONS[st.session_state.session_id] = time.time()

manage_concurrency()
st.write("This AI can answer from the document or use its general knowledge to provide more context.")

try: google_api_key = st.secrets["google_api_key"]
except (KeyError, FileNotFoundError):
    st.error("API Key not found in Streamlit secrets."); google_api_key = None

with st.sidebar:
    st.title("Teacher's Controls")
    if google_api_key:
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process Documents"):
            if not pdf_docs: st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Performing advanced analysis..."):
                    raw_chunks = process_multimodal_pdfs(pdf_docs, google_api_key)
                    docs = get_text_chunks(raw_chunks)
                    st.info("Building knowledge base...")
                    st.session_state.vector_store = get_vector_store(docs, google_api_key)
                    st.success("Documents processed! The AI is ready.")
    else: st.sidebar.error("Teacher: Add your API Key to enable.")

st.subheader("Student Q&A")
st.info(f"Questions asked: {st.session_state.question_count}/{MAX_QUESTIONS_PER_USER}")

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if st.session_state.question_count >= MAX_QUESTIONS_PER_USER:
    st.warning("You have reached the max number of questions. Please refresh to start over.")
else:
    if user_question := st.chat_input("Ask a question in English or Filipino..."):
        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"): st.markdown(user_question)

        if "vector_store" not in st.session_state:
            with st.chat_message("assistant"): st.warning("Please ask your teacher to upload and process a document first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please ask your teacher to upload and process a document first."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    vector_store = st.session_state.vector_store
                    retrieved_docs = vector_store.similarity_search(user_question, k=5)
                    document_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    router_chain = get_router_chain(google_api_key)
                    tool_choice = router_chain.run({"context": document_context, "question": user_question})
                    
                    response_text = ""
                    if "DOCUMENT_SEARCH" in tool_choice:
                        st.info("✅ Searching within the document...")
                        qa_chain = get_document_qa_chain(google_api_key)
                        response = qa_chain({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
                        response_text = response["output_text"]
                    elif "GENERAL_KNOWLEDGE_SEARCH" in tool_choice:
                        st.info("🧠 Combining general knowledge with document context...")
                        general_chain = get_general_knowledge_chain(google_api_key)
                        general_answer = general_chain.run(user_question)
                        synthesis_chain = get_synthesis_chain(google_api_key)
                        response_text = synthesis_chain.run({"question": user_question, "general_answer": general_answer, "document_context": document_context})
                    else: # IRRELEVANT
                        response_text = "I'm sorry, that question does not seem related to the content of the provided document. Let's focus on the lesson."
                    
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})