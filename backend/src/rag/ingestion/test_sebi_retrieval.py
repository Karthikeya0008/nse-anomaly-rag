"""
Quick retrieval sanity check for the SEBI Chroma collection.
Run this after ingestion to confirm queries return sensible, relevant chunks
-- both with plain semantic search and with date-filtered search (the
retrieval pattern your anomaly-explanation pipeline will actually use).
"""

import chromadb
from pathlib import Path

VECTORSTORE_DIR = Path(__file__).resolve().parents[4] / "db" / "vectorstore"
COLLECTION_NAME = "sebi_circulars"


def main():
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Collection has {collection.count()} chunks.\n")

    # Test 1: plain semantic search, no filter
    print("=== Test 1: plain semantic search ===")
    query = "margin requirements for stock brokers"
    results = collection.query(query_texts=[query], n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n[{i+1}] {meta['title']} ({meta['date']})")
        print(f"    {doc[:200]}...")

    # Test 2: date-filtered search -- this is the pattern the anomaly
    # pipeline will actually use: constrain to a window around the anomaly
    # date, THEN do semantic search within that window.
    print("\n\n=== Test 2: date-filtered search (near March 2020 COVID crash) ===")
    results = collection.query(
        query_texts=["market volatility circuit breaker trading halt"],
        n_results=3,
        where={
            "$and": [
                {"date_int": {"$gte": 20200301}},
                {"date_int": {"$lte": 20200415}}
            ]
        }
    )
    if results["documents"][0]:
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n[{i+1}] {meta['title']} ({meta['date']})")
            print(f"    {doc[:200]}...")
    else:
        print("No results found in this date range for this query -- "
              "try a broader date window or different query terms.")


if __name__ == "__main__":
    main()