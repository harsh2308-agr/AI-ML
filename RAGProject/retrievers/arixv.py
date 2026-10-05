import arxiv
import warnings
from langchain_core.documents import Document

# Initialize a clean client using the modern arxiv API syntax
client = arxiv.Client()

search = arxiv.Search(
    query="large language models",
    max_results=2
)

print("Fetching papers directly from arXiv...")

docs = []
# Use the modern .results() generator loop required by the new arxiv library
for result in client.results(search):
    # Wrap each paper cleanly into standard LangChain Document block structures
    doc = Document(
        page_content=result.summary,
        metadata={
            "Title": result.title,
            "Authors": ", ".join(author.name for author in result.authors)
        }
    )
    docs.append(doc)

# Print your results
for i, doc in enumerate(docs):
    print(f"\n================ RESULT {i+1} ================")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("\nSummary Preview:")
    print(doc.page_content[:500] + "...")
    print("==========================================")
