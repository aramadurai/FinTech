# query.py

import os
import json
import re
import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === Setup ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
JSON_DIR = "json"

# === Load Structured JSON Data ===
def load_all_json():
    data = []
    for fname in os.listdir(JSON_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(JSON_DIR, fname)) as f:
                try:
                    fields = json.load(f)
                    year = fields.get("Tax Year")
                    data.append({"filename": fname, "year": year, "fields": fields})
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
    return data

structured_data = load_all_json()

# === Extract Field and Year from Question ===
FIELD_KEYWORDS = {
    "agi": "Adjusted Gross Income (AGI)",
    "adjusted gross income": "Adjusted Gross Income (AGI)",
    "income": "Total Income",
    "wages": "Wages, Salaries, Tips",
    "withheld": "Federal Income Tax Withheld",
    "federal": "Federal Income Tax Withheld",
    "refund": "Refund Amount",
    "capital gains": "Capital Gains",
    "rental": "Rental Income",
    "exemption": "Exemptions",
    "ira": "IRA Deduction",
    "student loan": "Student Loan Interest Deduction",
    "taxable": "Taxable Income"
}

def extract_year(question):
    match = re.search(r"(20\d{2})", question)
    if match:
        return int(match.group(1))
    if "last year" in question:
        return datetime.datetime.now().year - 1
    return None

def extract_field(question):
    q = question.lower()
    for keyword, field in FIELD_KEYWORDS.items():
        if keyword in q:
            return field
    return None

# === Try Structured Lookup ===
def structured_lookup(question):
    field = extract_field(question)
    year = extract_year(question)
    if not field or not year:
        return None

    for entry in structured_data:
        if entry["year"] == year:
            value = entry["fields"].get(field)
            if value:
                return f"{field} in {year}: {value}"
    return None

# === Fallback to RAG ===
def rag_fallback(question):
    vectorstore = FAISS.load_local(
        "vectorstore",
        OpenAIEmbeddings(openai_api_key=openai_api_key),
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a tax assistant answering questions based on U.S. tax documents.

Context:
{context}

Question:
{question}

Answer concisely and cite the numbers or phrases from the context.
""",
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=False)
    return qa.run(question)

# === Main ===
if __name__ == "__main__":
    while True:
        question = input("\n❓ Ask a question (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        result = structured_lookup(question)
        if result:
            print("📊 Structured Answer:", result)
        else:
            print("🤖 Using RAG...")
            answer = rag_fallback(question)
            print("💬 Answer:", answer)
