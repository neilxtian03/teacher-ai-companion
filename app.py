# ==============================================================================
# FINAL (v3), TWO-STEP REASONING AI TEACHING COMPANION
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

# --- Global Variables & Constants (Unchanged) ---
ACTIVE_SESSIONS = {}
SESSION_LOCK = threading.Lock()
MAX_CONCURRENT_USERS = 30
MAX_QUESTIONS_PER_USER = 20
SESSION_TIMEOUT_SECONDS = 300

# --- Core Multi-Modal & Text Processing Functions (Unchanged) ---
def process_multimodal_pdfs(pdf_files, api_key):
    # ... (This function is the same as the previous version)
    all_content_chunks = []
    image_description_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
    temp_dir = "temp_pdf_files"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    for pdf_file in pdf_files:
        file_path = os.path.join(temp_dir, pdf_file.name)
        with open(file_path, "wb") as f: f.write(pdf_file.getbuffer())
        st.info(f"Analyzing text and tables in {pdf_file.name}...")
        elements = partition_pdf(filename=file_path, strategy="hi_res", infer_table_structure=True, model_name="yolox")
        for el in elements:
            if "unstructured.documents.elements.Table" in str(type(el)): all_content_chunks.append(f"Table Content:\n{el.metadata.text_as_html}\n")
            else: all_content_chunks.append(el.text)
        st.info(f"Analyzing images in {pdf_file.name}...")
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                for img in doc.get_page_images(page_num):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    message = HumanMessage(content=[{"type": "text", "text": "Describe this image in detail. What information does it convey? If it's a chart or graph, explain what it shows. Answer in English or Filipino."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"}}])
                    description_response = image_description_model.invoke([message])
                    all_content_chunks.append(f"Image Description (from page {page_num + 1}):\n{description_response.content}\n")
        except Exception as e: st.warning(f"Could not process an image in {pdf_file.name}. Error: {e}")
    return all_content_chunks

def get_text_chunks(content_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.create_documents(content_list)

def get_vector_store(documents, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_documents(documents, embedding=embeddings)

# --- AI Chains ---

def get_relevance_check_chain(api_key):
    # (Unchanged)
    prompt_template = "Based on the following document snippets, determine if the user's question is relevant. Ang iyong layunin ay iwasan ang mga tanong na \"off-topic\". Answer with only a single word: RELEVANT or IRRELEVANT.\n\nContext Snippets:\n{context}\n\nUser's Question:\n{question}\n\nAnswer (RELEVANT or IRRELEVANT):"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=model, prompt=prompt)

def get_document_qa_chain(api_key):
    # (Updated to be very strict and factual)
    prompt_template = """
    You are a factual, direct teaching assistant. Your ONLY job is to answer the user's question based strictly on the provided CONTEXT.
    - Synthesize information from the text, tables, and image descriptions if necessary.
    - Do NOT use any external knowledge. Behave as if the CONTEXT is the only information in the world.
    - Answer in the language of the user's QUESTION (English or Filipino).

    CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFactual Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# --- NEW: The "Mentor" Chain ---
def get_enhancement_offer_chain(api_key):
    """This chain crafts a follow-up question to offer more help."""
    prompt_template = """
    You are a friendly and encouraging teaching mentor. A student has just received a direct, factual answer from their textbook. Your task is to offer further assistance in an engaging way.

    - **DO NOT** re-answer the question.
    - **Your entire output should be a single, short follow-up question.**
    - Offer to provide a simpler explanation, a real-world example, or more context.
    - Frame it naturally in the language of the original question (English or Filipino).

    Original Question: {question}
    Factual Answer Provided: {document_answer}

    Example Output (if question was in English): "That's the direct answer from your material. Would you like me to explain it more simply, or maybe provide a real-world example?"
    Example Output (if question was in Filipino): "Iyan ang direktang sagot mula sa iyong materyal. Gusto mo bang ipaliwanag ko ito sa mas simpleng paraan, o magbigay ng isang halimbawa mula sa totoong buhay?"

    Your Follow-up Question:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "document_answer"])
    return LLMChain(llm=model, prompt=prompt)

# --- Streamlit App Main Logic ---

st.set_page_config(page_title="Mentor AI Companion", layout="wide")
st.header("📚 Mentor AI Companion")

# (Initialize session state and concurrency - Unchanged)
# ...
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'question_count' not in st.session_state: st.session_state.question_count = 0
def manage_concurrency():
    with SESSION_LOCK:
        current_time = time.time()
        expired_sessions = [sid for sid, t in ACTIVE_SESSIONS.items() if current_time - t > SESSION_TIMEOUT_SECONDS]
        for sid in expired_sessions: del ACTIVE_SESSIONS[sid]
        if st.session_state.session_id not in ACTIVE_SESSIONS and len(ACTIVE_SESSIONS) >= MAX_CONCURRENT_USERS:
            st.warning(f"The chatbot is at maximum capacity. Please try again in a few minutes.")
            st.stop()
        ACTIVE_SESSIONS[st.session_state.session_id] = time.time()
manage_concurrency()

st.write("This AI first answers from your documents, then offers to explain further.")

# (API Key and Teacher Controls - Unchanged)
# ...
try:
    google_api_key = st.secrets["google_api_key"]
except (KeyError, FileNotFoundError):
    st.error("API Key not found in Streamlit secrets.")
    google_api_key = None
with st.sidebar:
    st.title("Teacher's Controls")
    if google_api_key:
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process Documents"):
            if not pdf_docs: st.warning("Please upload a PDF.")
            else:
                with st.spinner("Performing advanced document analysis..."):
                    raw_chunks = process_multimodal_pdfs(pdf_docs, google_api_key)
                    documents = get_text_chunks(raw_chunks)
                    st.info("Building knowledge base...")
                    st.session_state.vector_store = get_vector_store(documents, google_api_key)
                    st.success("Documents processed and ready!")
    else: st.sidebar.error("Teacher: Add Google API Key to secrets.")

# (Chat History and Question Limit - Unchanged)
# ...
st.subheader("Student Q&A")
st.info(f"Questions asked: {st.session_state.question_count}/{MAX_QUESTIONS_PER_USER}")
if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

# --- UPDATED: Main Chat Logic with Two-Step Chain ---
if st.session_state.question_count >= MAX_QUESTIONS_PER_USER:
    st.warning("You have reached the maximum number of questions.")
else:
    if user_question := st.chat_input("Ask a question in English or Filipino..."):
        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"): st.markdown(user_question)

        if "vector_store" not in st.session_state:
            with st.chat_message("assistant"): st.warning("Please ask your teacher to process a document first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please ask your teacher to process a document first."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Finding answer in document..."):
                    vector_store = st.session_state.vector_store
                    retrieved_docs = vector_store.similarity_search(user_question, k=5)
                    context_for_relevance = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    relevance_chain = get_relevance_check_chain(google_api_key)
                    relevance_result = relevance_chain.run({"context": context_for_relevance, "question": user_question})
                    is_relevant = "RELEVANT" in relevance_result.strip().upper()
                    
                    if is_relevant:
                        # --- Step 1: Get the factual, document-based answer ---
                        qa_chain = get_document_qa_chain(google_api_key)
                        doc_answer_response = qa_chain({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
                        document_answer = doc_answer_response["output_text"]

                        # --- Step 2: Get the enhancement offer from the Mentor ---
                        st.spinner("Thinking of how I can help more...")
                        enhancement_chain = get_enhancement_offer_chain(google_api_key)
                        enhancement_offer = enhancement_chain.run({
                            "question": user_question,
                            "document_answer": document_answer
                        })

                        # --- Step 3: Combine them into one response ---
                        final_response_text = f"{document_answer}\n\n---\n\n*_{enhancement_offer}_*"
                    else:
                        final_response_text = "I'm sorry, but your question does not seem related to the content of the provided document."

                st.markdown(final_response_text)
                st.session_state.messages.append({"role": "assistant", "content": final_response_text})