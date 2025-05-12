import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Step 1: Load all text from all PDFs in the folder ===
def load_all_pdfs_text(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Reading: {filename}")
            doc = fitz.open(file_path)
            for page in doc:
                all_text += page.get_text()
    return all_text

# === Step 2: Split into chunks (adjust chunk size) ===
def split_into_chunks(text):
    # Adjust chunk_size to stay within token limits
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# === Step 3: Embed and store chunks (with batching) ===
def embed_chunks(chunks):
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Break chunks into manageable batches
    chunk_size = 50  # Adjust based on API limits
    chunk_batches = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]

    faiss_index = None

    for batch in chunk_batches:
        batch_texts = [chunk.page_content for chunk in batch]
        batch_vectors = embeddings.embed_documents(batch_texts)
        text_embedding_pairs = list(zip(batch_texts, batch_vectors))

        if faiss_index is None:
            faiss_index = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        else:
            faiss_index.add_embeddings(text_embedding_pairs)

    return faiss_index

# === Step 4: Save Vector Store ===
def save_vector_store(vstore, path="faiss_index"):
    vstore.save_local(path)

# === Main Runner ===
if __name__ == "__main__":
    folder = "docs"  # Put your PDFs in a folder called "docs"
    raw_text = load_all_pdfs_text(folder)
    chunks = split_into_chunks(raw_text)
    vectorstore = embed_chunks(chunks)
    save_vector_store(vectorstore)
    print("✅ All documents ingested and vector store saved.")
