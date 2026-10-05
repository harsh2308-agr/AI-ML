import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# 1. Define where you want to save your database folder on your disk
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "chroma_db")

chunks = [
    Document(page_content="The MacBook Pro features an M4 chip and high-performance unified memory.", metadata={"source": "tech_notes"}),
    Document(page_content="Neural Networks learn complicated patterns by adjusting numerical weights through backpropagation.", metadata={"source": "ai_notes"}),
    Document(page_content="The capital city of France is Paris, famous for its art and landmarks.", metadata={"source": "geo_notes"})
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

print("Generating embeddings and saving to local folder database...")

# 2. Add 'persist_directory' to force Chroma to save to your hard drive
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=db_dir
)

print(f"Vector store successfully saved to: {db_dir}")

# 3. Perform your search exactly the same way
query = "Tell me about machine learning architectures"
matching_docs = db.similarity_search(query, k=1)

print("\n================ SEARCH RESULT ================")
for doc in matching_docs:
    print(f"Content Found: {doc.page_content}")
print("===============================================\n")

retriever = db.as_retriever()
docs = retriever.invoke("What is capital city of france")

for d in docs:
    print(d.page_content)
