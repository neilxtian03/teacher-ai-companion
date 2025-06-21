# ==============================================================================
# FINAL, COMPLETE, HYBRID MULTI-MODAL AI TEACHING COMPANION
# ==============================================================================
import streamlit as st
import os
import uuid
import threading
import time
import base64
import fitz  # PyMuPDF for image extraction
from unstructured.partition.pdf import partition_pdf # For advanced PDF parsing

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# --- Global Variables & Constants for Concurrency and Limits ---
ACTIVE_SESSIONS = {}
SESSION_LOCK = threading.Lock()
MAX_CONCURRENT_USERS = 30
MAX_QUESTIONS_PER_USER = 20
SESSION_TIMEOUT_SECONDS = 300  # 5 minutes

# --- Core Multi-Modal Processing Engine ---

def process_multimodal_pdfs(pdf_files, api_key):
    """
    Processes PDF files by extracting text, tables, and generating descriptions for images.
    Returns a list of strings, where each string is a chunk of content.
    """
    all_content_chunks = []
    image_description_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
    
    temp_dir = "temp_pdf_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for pdf_file in pdf_files:
        file_path = os.path.join(temp_dir, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.info(f"Analyzing text and tables in {pdf_file.name}...")
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            model_name="yolox"
        )
        for el in elements:
            if "unstructured.documents.elements.Table" in str(type(el)):
                all_content_chunks.append(f"Table Content:\n{el.metadata.text_as_html}\n")
            else:
                all_content_chunks.append(el.text)
        
        st.info(f"Analyzing images in {pdf_file.name}...")
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                for img in doc.get_page_images(page_num):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this image in detail. What information does it convey? If it's a chart or graph, explain what it shows. Answer in English or Filipino."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(base_image['image']).decode()}"}}
                        ]
                    )
                    description_response = image_description_model.invoke([message])
                    all_content_chunks.append(f"Image Description (from page {page_num + 1}):\n{description_response.content}\n")
        except Exception as e:
            st.warning(f"Could not process an image in {pdf_file.name}. Error: {e}")

    return all_content_chunks

def get_text_chunks(content_list):
    """Takes a list of content strings and splits them into documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.create_documents(content_list)

def get_vector_store(documents, api_key):
    """Creates a FAISS vector store from document chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_documents(documents, embedding=embeddings)

# --- AI Chains: Router, Document QA, and General Knowledge ---

def get_router_chain(api_key):
    """This chain decides which tool to use."""
    prompt_template = """You are an expert routing agent. Your task is to analyze the user's question and decide which tool is best to answer it.
You must choose from the following tools:

1.  **DOCUMENT_SEARCH**: Use this tool if the question is asking directly about the content, events, characters, or specific details found *within* the provided document context.
    Examples: "What did the main character do?", "Summarize the first chapter.", "Show me the Python code for loops from the document."

2.  **GENERAL_KNOWLEDGE_SEARCH**: Use this tool if the question asks for a definition, a biography, historical context, or an explanation of a concept *mentioned* in the document, but which is likely not explained in detail *in* the document. This is for providing enriching, external information.
    Examples: "Who was Napoleon Bonaparte?", "What is 'dramatic irony'?", "What is Python?"

3.  **IRRELEVANT**: Use this tool for any other question that is off-topic or conversational.
    Examples: "Hello", "What's the weather like?", "Who are you?"

Based on the User's Question below, which tool should be used?
Answer with only the tool name: DOCUMENT_SEARCH, GENERAL_KNOWLEDGE_SEARCH, or IRRELEVANT.

User's Question: {question}

Tool:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return LLMChain(llm=model, prompt=prompt)

def get_document_qa_chain(api_key):
    """This chain answers questions based ONLY on the document."""
    prompt_template = """You are an expert teaching assistant. Ikaw ay isang dalubhasang teaching assistant.
Your goal is to answer the user's question by synthesizing information from the provided document context, which may include text, table data, and image descriptions.
Answer in the language of the user's question (English or Filipino).

Follow these rules:
1.  Analyze the provided CONTEXT to understand the core concepts.
2.  Synthesize your answer by connecting relevant information from the context.
3.  If the context describes tables or images, use that information to answer the question.
4.  Your final answer MUST be based entirely on the information from the CONTEXT. Do not use external knowledge.
5.  If you cannot answer, state "I cannot answer this with the provided document." or "Hindi ko ito masasagot gamit ang dokumento."

CONTEXT:
{context}

QUESTION:
{question}

Synthesized Answer (in English or Filipino):"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def get_general_knowledge_chain(api_key):
    """This chain answers questions using the AI's general knowledge."""
    prompt_template = """You are a helpful and knowledgeable teaching assistant. Ikaw ay isang matulungin at maalam na teaching assistant.
Answer the user's question clearly and concisely. Answer in the language of the user's question (English or Filipino).

Question: {question}
Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return LLMChain(llm=model, prompt=prompt)

# --- Streamlit App UI and Main Logic ---

st.set_page_config(page_title="Hybrid AI Companion", layout="wide")
st.header("📚 Hybrid AI Companion")

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
            st.warning(f"The chatbot has hit the maximum number of {MAX_CONCURRENT_USERS} users. Please try again in a few minutes.")
            st.stop()
        ACTIVE_SESSIONS[st.session_state.session_id] = time.time()

manage_concurrency()

st.write("This AI can answer from the document or use its general knowledge to provide more context.")

try:
    google_api_key = st.secrets["google_api_key"]
except (KeyError, FileNotFoundError):
    st.error("API Key not found. Please ensure it is set in your Streamlit secrets.")
    google_api_key = None

with st.sidebar:
    st.title("Teacher's Controls")
    if google_api_key:
        pdf_docs = st.file_uploader("Upload PDF files (with text, tables, images)", accept_multiple_files=True)
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                with st.spinner("Performing advanced document analysis... This may take a while."):
                    raw_chunks_list = process_multimodal_pdfs(pdf_docs, google_api_key)
                    documents = get_text_chunks(raw_chunks_list)
                    st.info("Building searchable knowledge base...")
                    st.session_state.vector_store = get_vector_store(documents, google_api_key)
                    st.success("Documents processed! The AI is ready.")
    else:
        st.sidebar.error("Teacher: Add your Google API Key to the app's secrets.")

st.subheader("Student Q&A")
st.info(f"You have asked {st.session_state.question_count} out of {MAX_QUESTIONS_PER_USER} questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.question_count >= MAX_QUESTIONS_PER_USER:
    st.warning("You have reached the maximum number of questions for this session. Please refresh to start over.")
else:
    if user_question := st.chat_input("Ask a question in English or Filipino..."):
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
                    router_chain = get_router_chain(google_api_key)
                    tool_choice = router_chain.run(user_question)
                    response_text = ""
                    
                    if "DOCUMENT_SEARCH" in tool_choice:
                        st.info("✅ Searching within the document...")
                        vector_store = st.session_state.vector_store
                        retrieved_docs = vector_store.similarity_search(user_question, k=5)
                        qa_chain = get_document_qa_chain(google_api_key)
                        response = qa_chain({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
                        response_text = response["output_text"]
                    
                    elif "GENERAL_KNOWLEDGE_SEARCH" in tool_choice:
                        st.info("🧠 Providing additional context from general knowledge...")
                        general_chain = get_general_knowledge_chain(google_api_key)
                        response_text = general_chain.run(user_question)

                    else: # IRRELEVANT
                        response_text = "I'm sorry, that question seems off-topic. Let's focus on the lesson."
                    
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})