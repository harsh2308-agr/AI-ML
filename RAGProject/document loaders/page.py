import warnings
# Silence the LangChain deprecation warning from cluttering your terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"

# Load the data using WebBaseLoader
data = WebBaseLoader(url)
docs = data.load()

print("\n================ SUCCESS ==================")
print(f"Total documents parsed: {len(docs)}")
print(f"Source URL verified: {docs[0].metadata['source']}")
print(f"Character Count Loaded: {len(docs[0].page_content)}")
print("===========================================\n")
