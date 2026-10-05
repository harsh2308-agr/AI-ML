# import os
# from pypdf import PdfReader
# from langchain_core.documents import Document

# # 1. Clear terminal output safely
# print("\033[H\033[J", end="") 
# print("--- STARTING LIGHTWEIGHT PDF LOADER ---")

# # 2. Track the path to your PDF file (Change "document.pdf" to your actual PDF file name)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "GRU.pdf")
# print(f"Looking for PDF at: {file_path}")

# try:
#     # 3. Read the PDF file natively using pypdf
#     reader = PdfReader(file_path)
#     docs = []

#     # 4. Loop through every page to extract text
#     for page_num, page in enumerate(reader.pages, start=1):
#         text_content = page.extract_text()
        
#         # Only add pages that actually contain text
#         if text_content.strip():
#             # Wrap each page into a standalone LangChain Document block
#             doc = Document(
#                 page_content=text_content, 
#                 metadata={"source": "GRU.pdf", "page": page_num}
#             )
#             docs.append(doc)

#     print("\n================ SUCCESS ==================")
#     print(f"Total PDF pages parsed into documents: {len(docs)}")
#     print("First Page Content Preview:")
#     # if docs:
#     #     print(docs[0])  # Prints the first page container to check structure
#     # else:
#     #     print("[Warning: PDF was read but no text could be extracted. It might be scanned/an image]")
#     # print("===========================================\n")

# except FileNotFoundError:
#     print(f"\n❌ FILE ERROR: Cannot find your PDF file!")
#     print(f"Please check that your PDF is saved inside: {current_dir}")
# except Exception as e:
#     print(f"\n❌ UNEXPECTED ERROR: {e}")

# input("Press Enter to close...")



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)