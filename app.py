# ==============================================================================
# FINAL (v10), FULLY MODERNIZED WITH LCEL - COMPLETE CODE
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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Global Variables & Constants ---
ACTIVE_SESSIONS = {}
SESSION_LOCK = threading.Lock()
MAX_CONCURRENT_USERS = 30
MAX_QUESTIONS_PER_USER = 20
SESSION_TIMEOUT_SECONDS = 300  # 5 minutes

# --- Core Processing Functions ---

def process_multimodal_pdfs(pdf_files, api_key):
    """Processes PDF files by extracting text, tables, and image descriptions."""
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

# --- AI Chains (All modernized with LCEL) ---

def get_router_chain(api_key):
    """This context-aware chain decides which tool to use."""
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
    return prompt | model | StrOutputParser()

def get_document_qa_chain(api_key):
    """This chain answers questions based ONLY on the document."""
    prompt_template_str = """You are an expert teaching assistant. Ikaw ay isang dalubhasang teaching assistant.
Your goal is to answer the user's question by synthesizing information from the provided document context, which may include text, table data, and image descriptions.
Answer in the language of the user's question (English or Filipino).

Follow these rules:
1.  Analyze the provided CONTEXT to understand the core concepts.
2.  Synthesize your answer by connecting relevant information from the context.
3.  If the context describes tables or images, use that information to answer the question.
4.  Your final answer MUST be based entirely on the information from the CONTEXT. Do not use external knowledge.
5.  If you cannot answer, state "I cannot answer this with the provided document." or "Hindi ko ito masasagot gamit ang dokumento."

CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nSynthesized Answer (in English or Filipino):"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate.from_template(prompt_template_str)
    return create_stuff_documents_chain(llm=model, prompt=prompt)

def get_general_knowledge_chain(api_key):
    """This chain gets a generic answer using the AI's general knowledge."""
    prompt_template = """You are a helpful and knowledgeable teaching assistant. Ikaw ay isang matulungin at maalam na teaching assistant.
Answer the user's question clearly and concisely. Answer in the language of the user's question (English or Filipino).

Question: {question}
Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return prompt | model | StrOutputParser()

def get_synthesis_chain(api_key):
    """This chain combines general knowledge with document context into a final, relevant answer."""
    prompt_template = """You are a master teaching assistant. Your job is to create a comprehensive and relevant answer for a student by combining two pieces of information: a general knowledge answer and specific context from a document they are studying.

Follow these steps:
1.  Start with the GENERAL KNOWLEDGE ANSWER to provide the main definition or explanation.
2.  Then, seamlessly connect this general knowledge to the specific DOCUMENT CONTEXT. Explain how the concept applies to the document. Use phrases like "In the document you're reading...", "This relates to the story because...", or "For example, in the provided text...".
3.  Answer in the language of the user's original question (English or Filipino).

---
USER'S ORIGINAL QUESTION:\n{question}\n\nGENERAL KNOWLEDGE ANSWER:\n{general_answer}\n\nDOCUMENT CONTEXT:\n{document_context}\n---

Your Final, Synthesized Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "general_answer", "document_context"])
    return prompt | model | StrOutputParser()

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
        pdf_docs = st.file_uploader("Upload PDF files (with text, tables, images)", accept_multiple_files=True)
        if st.button("Process Documents"):
            if not pdf_docs: st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Performing advanced document analysis... This may take a while."):
                    raw_chunks_list = process_multimodal_pdfs(pdf_docs, google_api_key)
                    documents = get_text_chunks(raw_chunks_list)
                    st.info("Building searchable knowledge base...")
                    st.session_state.vector_store = get_vector_store(documents, google_api_key)
                    st.success("Documents processed! The AI is ready.")
    else: st.sidebar.error("Teacher: Add your Google API Key to enable.")

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
                    tool_choice = router_chain.invoke({"context": document_context, "question": user_question})
                    
                    response_text = ""
                    if "DOCUMENT_SEARCH" in tool_choice:
                        st.info("✅ Searching within the document...")
                        qa_chain = get_document_qa_chain(google_api_key)
                        response_text = qa_chain.invoke({"context": retrieved_docs, "question": user_question})
                    
                    elif "GENERAL_KNOWLEDGE_SEARCH" in tool_choice:
                        st.info("🧠 Combining general knowledge with document context...")
                        general_chain = get_general_knowledge_chain(google_api_key)
                        general_answer = general_chain.invoke({"question": user_question})
                        
                        synthesis_chain = get_synthesis_chain(google_api_key)
                        response_text = synthesis_chain.invoke({
                            "question": user_question, 
                            "general_answer": general_answer, 
                            "document_context": document_context
                        })
                    else: # IRRELEVANT
                        response_text = "I'm sorry, that question does not seem related to the content of the provided document. Let's focus on the lesson."
                    
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})