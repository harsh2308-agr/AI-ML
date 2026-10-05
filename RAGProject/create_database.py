import os
import warnings
# Silence ecosystem package alerts from cluttering your prompt
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Optimized context retriever parameters to protect Free Tier API bandwidth
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,          # Reduced from 4 to 2 to minimize the size of the request text
        "fetch_k": 5,    # Reduced from 10 to 5
        "lambda_mult": 0.5
    }
)

# Initializing model with built-in retry logic
llm = ChatMistralAI(
    model="mistral-small-2506",
    max_retries=5,       # Forces the script to pause and backoff gracefully on 429 alerts
    timeout=60
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]
)

print("\n🚀 Rag system created successfully.")
print("Type your question below (press 0 to exit)\n")

while True:
    query = input("You: ")
    if query.strip() == "0":
        print("Exiting system...")
        break 
        
    if not query.strip():
        continue

    try:
        # Fetch matching local vector segments
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })
        
        # Query execution block
        response = llm.invoke(final_prompt)
        print(f"\nAI: {response.content}\n")
        
    except Exception as e:
        # Graceful fallback indicator if the Mistral tier remains locked out
        if "429" in str(e):
            print("\n⚠️ AI: Mistral API is heavily rate-limited right now. Please wait a few seconds before asking another question.\n")
        else:
            print(f"\n❌ Error processing request: {e}\n")
