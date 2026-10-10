"""
Unified Dataset B Ingestion
-----------------------------
Scans db/raw_text/<source>/*.json (SEBI circulars, Business Standard news,
Wikipedia events -- any source following the common schema), chunks each
record's text, embeds, and stores everything in ONE Chroma collection so
retrieval can search across all sources in a single query, optionally
filtered by source/date/sector metadata.

Duplicate handling:
- Duplicate documents with the same URL are skipped.
- Duplicate chunk IDs are skipped as a final safety check.
"""

import hashlib
import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_TEXT_ROOT = Path(__file__).resolve().parents[4] / "db" / "raw_text"
VECTORSTORE_DIR = Path(__file__).resolve().parents[4] / "db" / "vectorstore"

COLLECTION_NAME = "market_context"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
BATCH_SIZE = 500


def load_all_records():
    """Load all JSON records containing non-empty text."""
    records = []

    if not RAW_TEXT_ROOT.exists():
        print(f"Raw text directory does not exist: {RAW_TEXT_ROOT}")
        return records

    for source_dir in RAW_TEXT_ROOT.iterdir():
        if not source_dir.is_dir():
            continue

        json_files = [
            f
            for f in source_dir.glob("*.json")
            if not f.name.startswith("_")
        ]

        for fpath in json_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    record = json.load(f)

                if record.get("text"):
                    records.append(record)

            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: could not load {fpath}: {exc}")

    return records


def remove_duplicate_records(records):
    """
    Remove duplicate documents based on source + URL.

    This prevents the same underlying document from being ingested more
    than once even if it exists in multiple JSON files.
    """
    seen_documents = set()
    unique_records = []
    skipped = 0

    for record in records:
        source = record.get("source", "unknown")
        url = record.get("url", "").strip()

        # Prefer source + URL so the same URL from different sources
        # is not accidentally considered the same document.
        document_key = f"{source}|{url}"

        if document_key in seen_documents:
            skipped += 1
            print(
                f"Skipping duplicate document: "
                f"[{source}] {url}"
            )
            continue

        seen_documents.add(document_key)
        unique_records.append(record)

    return unique_records, skipped


def make_chunk_id(record, chunk_index):
    """
    Create a stable, collision-resistant chunk ID.

    We hash source + URL + title rather than relying only on the last
    part of the URL, which can collide after truncation.
    """
    source = str(record.get("source", "unknown"))
    url = str(record.get("url", ""))
    title = str(record.get("title", ""))

    record_key = f"{source}|{url}|{title}"

    record_hash = hashlib.sha1(
        record_key.encode("utf-8")
    ).hexdigest()[:16]

    return f"{source}_{record_hash}_chunk{chunk_index}"


def chunk_and_ingest():
    records = load_all_records()

    # ------------------------------------------------------------------
    # 1. Show original record counts
    # ------------------------------------------------------------------
    by_source = {}

    for record in records:
        source = record.get("source", "unknown")
        by_source[source] = by_source.get(source, 0) + 1

    print("Records found by source:")
    for source, count in by_source.items():
        print(f"  {source}: {count}")

    print(f"Total records loaded: {len(records)}\n")

    # ------------------------------------------------------------------
    # 2. Remove duplicate documents
    # ------------------------------------------------------------------
    records, skipped_records = remove_duplicate_records(records)

    print(
        f"Unique records after duplicate removal: {len(records)}"
    )
    print(
        f"Duplicate documents skipped: {skipped_records}\n"
    )

    # ------------------------------------------------------------------
    # 3. Set up text splitter
    # ------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # ------------------------------------------------------------------
    # 4. Connect to Chroma
    # ------------------------------------------------------------------
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR)
    )

    # Since this is a full rebuild, remove the previous collection.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(
            f"Dropped existing '{COLLECTION_NAME}' collection "
            f"before re-ingesting.\n"
        )
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # ------------------------------------------------------------------
    # 5. Chunk all records
    # ------------------------------------------------------------------
    all_ids = []
    all_documents = []
    all_metadatas = []

    seen_chunk_ids = set()
    skipped_chunks = 0

    for record in records:
        source = record.get("source", "unknown")
        url = record.get("url", "")
        title = record.get("title", "")
        date = record.get("date", "")

        chunks = splitter.split_text(record["text"])

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(
                record,
                chunk_index,
            )

            # Final safety check:
            # Chroma requires every ID to be unique.
            if chunk_id in seen_chunk_ids:
                skipped_chunks += 1
                print(
                    f"Skipping duplicate chunk ID: {chunk_id}"
                )
                continue

            seen_chunk_ids.add(chunk_id)

            # Make sure date_int does not crash if date is missing
            # or unexpectedly formatted.
            try:
                date_int = int(date.replace("-", ""))
            except (AttributeError, ValueError):
                date_int = 0

            all_ids.append(chunk_id)
            all_documents.append(chunk)

            all_metadatas.append(
                {
                    "source": source,
                    "date": date,
                    "date_int": date_int,
                    "title": title,
                    "url": url,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                }
            )

    # ------------------------------------------------------------------
    # 6. Summary before ingestion
    # ------------------------------------------------------------------
    print(f"Total chunks to embed: {len(all_documents)}")
    print(f"Duplicate chunks skipped: {skipped_chunks}")
    print("Embedding and storing in Chroma...\n")

    # ------------------------------------------------------------------
    # 7. Batch ingestion
    # ------------------------------------------------------------------
    total_chunks = len(all_documents)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_docs = all_documents[i:i + BATCH_SIZE]
        batch_meta = all_metadatas[i:i + BATCH_SIZE]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )

        stored_so_far = min(
            i + BATCH_SIZE,
            total_chunks,
        )

        print(
            f"Embedded and stored "
            f"{stored_so_far}/{total_chunks} chunks"
        )

    # ------------------------------------------------------------------
    # 8. Final verification
    # ------------------------------------------------------------------
    final_count = collection.count()

    print(
        f"\nDone. Collection '{COLLECTION_NAME}' "
        f"now has {final_count} chunks."
    )
    print(f"Stored at: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    chunk_and_ingest()