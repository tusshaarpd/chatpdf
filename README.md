# chatpdf

How It Works
PDF Upload:

Users upload a PDF file through the st.file_uploader.
The app extracts text using PyPDF2 and splits it into manageable chunks for processing.
Chunk Embedding:

Each chunk of the PDF is converted into embeddings using the all-MiniLM-L6-v2 model from Sentence Transformers.
Query Handling:

When the user asks a question, the app calculates the similarity between the question embedding and PDF chunk embeddings to find the most relevant chunk.
The QA pipeline from Hugging Face (distilbert-base-uncased-distilled-squad) extracts the answer from the most relevant chunk.
Chat Interface:

The app maintains a chat history using st.session_state to display past interactions.
