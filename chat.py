import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Streamlit app title
st.title("Chat with your PDF 📄🤖")
st.write("Upload a PDF file (up to 300 MB) and ask questions!")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Initialize session state for chat
if "history" not in st.session_state:
    st.session_state.history = []

# Load Hugging Face model for Q&A
@st.cache_resource
def load_models():
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return qa_model, embedding_model

qa_pipeline, embedder = load_models()

# Extract text from PDF
def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

if uploaded_file:
    # Display uploaded file name
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Extract text from the PDF
    with st.spinner("Extracting text from the PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)

    # Split PDF content into smaller chunks
    def split_into_chunks(text, max_tokens=200):
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    pdf_chunks = split_into_chunks(pdf_text)

    # Generate embeddings for the PDF chunks
    with st.spinner("Processing PDF for efficient querying..."):
        pdf_embeddings = [embedder.encode(chunk, convert_to_tensor=True) for chunk in pdf_chunks]

    # Chat interface
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        # Add user query to chat history
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.spinner("Generating a response..."):
            # Find the most relevant chunk using similarity
            query_embedding = embedder.encode(user_input, convert_to_tensor=True)
            similarities = [util.pytorch_cos_sim(query_embedding, chunk)[0][0].item() for chunk in pdf_embeddings]
            most_relevant_idx = similarities.index(max(similarities))
            most_relevant_chunk = pdf_chunks[most_relevant_idx]

            # Use the QA pipeline to answer the question
            response = qa_pipeline(question=user_input, context=most_relevant_chunk)
            answer = response["answer"]

            # Add assistant response to chat history
            st.session_state.history.append({"role": "assistant", "content": answer})

    # Display chat history
    for message in st.session_state.history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
else:
    st.info("Please upload a PDF to get started.")
