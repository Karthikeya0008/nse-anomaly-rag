"""Quick check: what date range does the business_standard source actually cover in Chroma?"""
import chromadb
from pathlib import Path
from collections import Counter

VECTORSTORE_DIR = Path(__file__).resolve().parents[3] / "db" / "vectorstore"
COLLECTION_NAME = "market_context"

client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

result = collection.get(where={"source": {"$eq": "business_standard"}}, include=["metadatas"])
dates = sorted(set(m["date"] for m in result["metadatas"]))
years = Counter(d[:4] for d in dates)

print(f"Total business_standard chunks: {len(result['metadatas'])}")
print(f"Unique dates covered: {len(dates)}")
print(f"Earliest date: {dates[0] if dates else 'N/A'}")
print(f"Latest date: {dates[-1] if dates else 'N/A'}")
print(f"\nArticles per year:")
for year in sorted(years):
    print(f"  {year}: {years[year]}")