# --- FULL RUNNABLE CODE ---
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from langchain.chains import LLMChain
import time
import threading
import uuid

# --- Global Variables for Concurrency Control ---
ACTIVE_SESSIONS = {}
SESSION_LOCK = threading.Lock()
MAX_CONCURRENT_USERS = 30
MAX_QUESTIONS_PER_USER = 20
SESSION_TIMEOUT_SECONDS = 300

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_relevance_check_chain(api_key):
    prompt_template = "Based on the following document snippets, determine if the user's question is relevant to the content of the documents. Your goal is to prevent off-topic questions. Answer with only a single word: RELEVANT or IRRELEVANT.\n\nContext Snippets:\n{context}\n\nUser's Question:\n{question}\n\nAnswer (RELEVANT or IRRELEVANT):"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=model, prompt=prompt)

def get_document_qa_chain(api_key):
    prompt_template = "You are an expert teaching assistant. Your goal is to answer the student's question by synthesizing information from the provided document context.\n\nFollow these rules strictly:\n1.  Analyze the provided CONTEXT snippets to understand the core concepts.\n2.  Formulate your answer by connecting relevant information from different parts of the context.\n3.  If the context contains code examples, use them to build your answer.\n4.  Your final answer MUST be based entirely on the information that can be inferred from the provided CONTEXT. Do not use any external knowledge.\n5.  If you cannot construct a confident answer from the context, clearly state \"I cannot answer this question with the provided document.\"\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nSynthesized Answer:"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

st.set_page_config(page_title="Classroom AI Companion", layout="wide")
st.header("📚 Classroom AI Companion")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0

def manage_concurrency():
    with SESSION_LOCK:
        current_time = time.time()
        expired_sessions = [sid for sid, t in ACTIVE_SESSIONS.items() if current_time - t > SESSION_TIMEOUT_SECONDS]
        for sid in expired_sessions:
            del ACTIVE_SESSIONS[sid]
        if st.session_state.session_id not in ACTIVE_SESSIONS and len(ACTIVE_SESSIONS) >= MAX_CONCURRENT_USERS:
            st.warning(f"The chatbot has hit the maximum number of {MAX_CONCURRENT_USERS} users at this moment. Please try again in a few minutes.")
            st.stop()
        ACTIVE_SESSIONS[st.session_state.session_id] = current_time

manage_concurrency()

st.write("This AI can reason about the content of the uploaded documents.")

try:
    google_api_key = st.secrets["google_api_key"]
except (KeyError, FileNotFoundError):
    st.error("API Key not found. Please ensure it is set in your Streamlit secrets.")
    google_api_key = None

with st.sidebar:
    st.title("Lesson Files")
    if google_api_key:
        pdf_docs = st.file_uploader("Upload your PDF Lesson Files", accept_multiple_files=True)
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                with st.spinner("Processing documents... This may take a moment."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks, google_api_key)
                    st.success("Documents processed successfully!")
    else:
        st.sidebar.error("Teacher: Please add your Google API Key to the app's secrets.")

st.subheader("Student Q&A")
st.info(f"You have asked {st.session_state.question_count} out of {MAX_QUESTIONS_PER_USER} questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.question_count >= MAX_QUESTIONS_PER_USER:
    st.warning("You have reached the maximum number of questions for this session. Please refresh the page to start a new session.")
else:
    if user_question := st.chat_input("Ask a question about the lesson..."):
        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if "vector_store" not in st.session_state:
            with st.chat_message("assistant"):
                st.warning("Please ask your teacher to upload and process a document first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please ask your teacher to upload and process a document first."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    vector_store = st.session_state.vector_store
                    retrieved_docs = vector_store.similarity_search(user_question, k=5)
                    context_for_relevance = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    relevance_chain = get_relevance_check_chain(google_api_key)
                    relevance_result = relevance_chain.run({"context": context_for_relevance, "question": user_question})
                    is_relevant = "RELEVANT" in relevance_result.strip().upper()
                    if is_relevant:
                        qa_chain = get_document_qa_chain(google_api_key)
                        response = qa_chain({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
                        response_text = response["output_text"]
                    else:
                        response_text = "I'm sorry, but your question does not seem related to the content of the provided document. I can only answer questions about the lesson materials."
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})