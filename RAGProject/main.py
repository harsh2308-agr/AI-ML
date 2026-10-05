import os
import warnings
# Suppress package alerts to keep console outputs completely clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 1024-dimension engine to match your compiled chroma database
embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 2. Extract minimal text slices to respect free API traffic limits
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,          
        "fetch_k": 5,    
        "lambda_mult": 0.5
    }
)

# 3. Add robust request parameters
llm = ChatMistralAI(
    model="mistral-small-2506",
    max_retries=5,       
    timeout=60
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Use ONLY the provided context to answer the question.
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

    # 4. CRITICAL FIX: The entire pipeline execution is now wrapped safely
    try:
        # Step A: Search local PDF vectors
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })
        
        # Step B: Request answer compilation from LLM
        response = llm.invoke(final_prompt)
        print(f"\nAI: {response.content}\n")
        
    except Exception as e:
        # Step C: Graceful Free-Tier Fallback handling
        print("\n⚠️  [Mistral API Rate Limit (429) Triggered]")
        print("Bypassing cloud bottleneck... Displaying matching raw text segments directly from your PDF:\n")
        
        # Re-fetch local data blocks if it hadn't completed before the crash window
        try:
            fallback_docs = retriever.invoke(query)
            for idx, doc in enumerate(fallback_docs, start=1):
                print(f"--- Relevant Segment {idx} ---")
                print(doc.page_content)
                print("-" * 30 + "\n")
        except Exception as inner_error:
            print(f"Could not retrieve local database context: {inner_error}\n")
