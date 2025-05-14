# ingest.py

import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# === Setup ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

DOCS_DIR = "docs"
JSON_DIR = "json"
os.makedirs(JSON_DIR, exist_ok=True)

# === Prompt Template ===
template = """
You are a helpful assistant extracting key financial fields from U.S. tax documents.
Extract only the fields that are present, and return them as clean JSON.

Fields:
- Adjusted Gross Income (AGI)
- Total Income
- Taxable Income
- Federal Income Tax Withheld
- Exemptions
- Wages, Salaries, Tips
- Rental Income
- Capital Gains
- IRA Deduction
- Student Loan Interest Deduction
- Refund Amount
- Tax Year (if found)

Text:
{text}

Return ONLY the JSON.
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
chain = prompt | llm

# === Helper Functions ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_structured_fields(text):
    truncated_text = text[:16000]
    try:
        response = chain.invoke({"text": truncated_text})

        # The result is an AIMessage object; extract content
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = response

        parsed = json.loads(response_text)

        # Ensure all expected fields are included
        EXPECTED_FIELDS = [
            "Adjusted Gross Income (AGI)",
            "Total Income",
            "Taxable Income",
            "Federal Income Tax Withheld",
            "Exemptions",
            "Wages, Salaries, Tips",
            "Rental Income",
            "Capital Gains",
            "IRA Deduction",
            "Student Loan Interest Deduction",
            "Refund Amount",
            "Tax Year"
        ]

        for field in EXPECTED_FIELDS:
            if field not in parsed:
                parsed[field] = None  # This will serialize as `null` in the JSON
        return parsed        
    except Exception as e:
        print("Error extracting fields:", e)
        return {}

def save_json(data, filename):
    filepath = os.path.join(JSON_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")

def chunk_for_vectorstore(text, source):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

# === Main Process ===
all_chunks = []
for fname in os.listdir(DOCS_DIR):
    if not fname.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(DOCS_DIR, fname)
    print(f"🧠 Extracting fields from: {fname}")
    raw_text = extract_text_from_pdf(pdf_path)

    fields = extract_structured_fields(raw_text)
    base_name = os.path.splitext(fname)[0]
    save_json(fields, f"{base_name}.json")

    doc_chunks = chunk_for_vectorstore(raw_text, source=base_name)
    all_chunks.extend(doc_chunks)

# === Save Vector Index for RAG ===
print("📚 Building vector store...")
vectorstore = FAISS.from_documents(all_chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))
vectorstore.save_local("vectorstore")
print("✅ Vector store saved to ./vectorstore")
