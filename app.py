import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from langchain.chains import LLMChain
import uuid

# --- REMOVED: All global variables for concurrency (ACTIVE_SESSIONS, SESSION_LOCK) ---

# --- Constants ---
MAX_QUESTIONS_PER_USER = 20

# --- Core Functions (No changes here) ---
# ... (all your get_pdf_text, get_text_chunks, etc. functions) ...
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

@st.cache_resource # This is the key optimization for handling many users!
def create_vector_store(_uploaded_files, api_key):
    st.info("Processing new documents. This happens once per document set.")
    raw_text = get_pdf_text(_uploaded_files)
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_relevance_check_chain(api_key):
    prompt_template = "Based on the following document snippets, determine if the user's question is relevant... (full prompt)"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=model, prompt=prompt)

def get_document_qa_chain(api_key):
    prompt_template = "You are an expert teaching assistant... (full prompt)"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# --- Streamlit App ---

st.set_page_config(page_title="Classroom AI Companion", layout="wide")
st.header("📚 Classroom AI Companion")

# --- Initialize Session State (no concurrency check needed) ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.write("This AI can reason about the content of the uploaded documents.")

try:
    google_api_key = st.secrets["google_api_key"]
except (KeyError, FileNotFoundError):
    st.error("API Key not found. Please ensure it is set in your Streamlit secrets.")
    google_api_key = None

with st.sidebar:
    st.title("Lesson Files")
    if google_api_key:
        pdf_docs = st.file_uploader("Upload PDF Lesson Files", accept_multiple_files=True)
        if pdf_docs:
            # When files are uploaded, we call our cached function.
            # This is robust and handles many users efficiently.
            st.session_state.vector_store = create_vector_store(pdf_docs, google_api_key)
            if st.session_state.vector_store:
                 st.sidebar.success("Documents are processed and ready for all students!")
    else:
        st.sidebar.error("Teacher: Please add your Google API Key to secrets.")

    st.divider()

    # --- UPDATED: Simplified Session Visualizer ---
    st.title("Your Session Info")
    st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")

# --- Main Chat Interface (with robust checks) ---
st.subheader("Student Q&A")
st.info(f"You have asked {st.session_state.question_count} out of {MAX_QUESTIONS_PER_USER} questions.")

# ... (The rest of your chat logic remains the same) ...
# It will check if `st.session_state.vector_store` exists before answering.
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.question_count >= MAX_QUESTIONS_PER_USER:
    st.warning("You have reached the maximum number of questions for this session.")
else:
    if user_question := st.chat_input("Ask a question about the lesson..."):
        # Check if a document has been processed and cached first
        if not st.session_state.vector_store:
            st.warning("Please upload and process a document. Please wait.")
        else:
            st.session_state.question_count += 1
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
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
                        response_text = "I'm sorry, but your question does not seem related to the content of the provided document."
                    
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})