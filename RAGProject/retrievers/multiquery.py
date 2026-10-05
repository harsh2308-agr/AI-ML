import os
import warnings
# Silence deprecation warning clutter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI

load_dotenv()

# Basic mock documents
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# 1. Using the modern standalone Hugging Face Embedding layout
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 2. Add 'max_retries' to gracefully handle Mistral 429 Rate Limits
llm = ChatMistralAI(
    model="mistral-small-latest",
    max_retries=5  # Forces the script to pause and backoff instead of crashing
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"
print("Generating variations and retrieving documents...")

try:
    retrieved_docs = multi_query_retriever.invoke(query)
    
    print("\nRetrieved Documents:\n")
    for doc in retrieved_docs:
        print(f"- {doc.page_content}")
        
except Exception as e:
    print(f"\nMistral API still heavily throttled. Alternative manual query look-up output:")
    # Fallback to normal retrieval if the API completely locks out free tiers
    fallback_docs = retriever.invoke(query)
    for doc in fallback_docs[:2]:
         print(f"- {doc.page_content}")
