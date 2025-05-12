from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create a retriever
retriever = vectorstore.as_retriever()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),  # You can specify model_name="gpt-4" if you have access
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "what was my gross income in 2024?"
response = qa_chain(query)

# Print the answer
print("\nAnswer:")
print(response["result"])