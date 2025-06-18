import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from langchain.chains import LLMChain

# --- Core Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Creates and saves a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# --- NEW: Relevance Check Chain ---
# This chain acts as a "Router". It decides if the question is on-topic.
def get_relevance_check_chain(api_key):
    """Creates a chain to check if the question is relevant to the document context."""
    prompt_template = """
    Based on the following document snippets, determine if the user's question is relevant to the content of the documents.
    Your goal is to prevent off-topic questions.
    Answer with only a single word: RELEVANT or IRRELEVANT.

    Context Snippets:
    {context}

    User's Question:
    {question}

    Answer (RELEVANT or IRRELEVANT):
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# --- This is our original RAG chain for answering the question ---
def get_document_qa_chain(api_key):
    """Creates the main question-answering chain that uses the document context."""
    prompt_template = """
    You are a helpful teaching assistant. Your goal is to answer the student's question based ONLY on the provided context.
    - Do not use any external knowledge.
    - If the answer is not found in the context, clearly state that you cannot find the answer in the document.
    - Be friendly, concise, and clear in your explanation.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit App ---

st.set_page_config(page_title="Smart AI Companion", layout="wide")
st.header("📚 Smart AI Companion")
st.write("This AI will only answer questions based on the content of the uploaded documents.")

try:
    google_api_key = st.secrets["google_api_key"]
except (KeyError, FileNotFoundError):
    st.error("API Key not found. Please ensure it is set in your Streamlit secrets.")
    google_api_key = None

# Sidebar for Teacher's Controls
with st.sidebar:
    st.title("Teacher's Controls")
    
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

# Main Chat Interface
st.subheader("Student Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about the lesson..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Main logic starts here
    if "vector_store" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please ask your teacher to upload and process a document first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please ask your teacher to upload and process a document first."})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                vector_store = st.session_state.vector_store
                
                # Step 1: Check for relevance
                retrieved_docs = vector_store.similarity_search(user_question, k=3)
                context_for_relevance = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                relevance_chain = get_relevance_check_chain(google_api_key)
                relevance_result = relevance_chain.run({"context": context_for_relevance, "question": user_question})
                
                # Clean up the result for a clean check
                is_relevant = "RELEVANT" in relevance_result.strip().upper()
                
                # Step 2: Act based on relevance
                if is_relevant:
                    # If relevant, proceed with the main QA chain
                    qa_chain = get_document_qa_chain(google_api_key)
                    response = qa_chain({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
                    response_text = response["output_text"]
                else:
                    # If not relevant, provide the canned response
                    response_text = "I'm sorry, but your question does not seem related to the content of the provided document. I can only answer questions about the lesson materials."
                
                st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})