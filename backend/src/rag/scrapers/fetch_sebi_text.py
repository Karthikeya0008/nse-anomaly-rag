"""
SEBI Circular Text Fetcher (Stage 2)
-------------------------------------
Reads the JSON files produced by sebi_scraper.py (title/date/url only, empty
"text" field), visits each circular's detail page, finds its attached PDF,
downloads it, extracts the text, and fills in the "text" field.

Why a separate script from the scraper: the scraper's job (paginate + collect
metadata) and this job (fetch + parse PDFs) fail differently and at different
speeds. Keeping them separate means a PDF-parsing bug doesn't force you to
re-scrape the listing, and vice versa.

WHAT I CONFIRMED vs WHAT I'M ASSUMING:
- CONFIRMED (via direct fetch of a real circular page): detail pages have a
  link matching pattern "../../../web/?file=<pdf_url>" pointing to the actual
  PDF attachment.
- ASSUMING: this pattern holds for all 1223 circulars. Some circulars (rare)
  might have no PDF, multiple PDFs, or an amendment/corrigendum structure
  that differs. The script logs and skips anything it can't parse rather than
  crashing, so check the "failed" log at the end and inspect those manually.

This uses plain `requests`, not Selenium -- these detail/PDF pages are static,
no JS rendering needed, so it's faster and lighter than the scraper.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
import pdfplumber
import io

RAW_TEXT_DIR = Path(__file__).resolve().parents[4] / "db" / "raw_text" / "sebi"
FAILED_LOG = RAW_TEXT_DIR / "_failed_fetches.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

POLITE_DELAY = 1.5  # seconds between requests -- do not lower much


BASE_URL = "https://www.sebi.gov.in"


def _resolve(path: str) -> str:
    """Turn a relative or absolute PDF path into a full, fetchable URL."""
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return BASE_URL + path


def extract_pdf_url(detail_page_html: str) -> str | None:
    """Find the PDF attachment link on a circular's detail page.

    Tries structured <a href> parsing first (works when BeautifulSoup's
    parser agrees with the page structure), then falls back to a raw regex
    scan of the whole HTML for a sebi_data/attachdocs/....pdf path -- this
    catches cases where the link sits in unexpected markup, differs in
    quoting, or BeautifulSoup's parser skips it for some reason.
    """
    soup = BeautifulSoup(detail_page_html, "html.parser")

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "?file=" in href:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if "file" in qs:
                return _resolve(unquote(qs["file"][0]))
        if href.lower().endswith(".pdf"):
            return _resolve(href)

    # Fallback: raw regex scan across the entire HTML text, independent of
    # tag structure. Matches paths like /sebi_data/attachdocs/.../xyz.pdf
    # or sebi_data/attachdocs/xyz.pdf, with or without a leading slash.
    match = re.search(r'/?sebi_data/attachdocs/[^\s"\'<>)]+\.pdf', detail_page_html, re.IGNORECASE)
    if match:
        return _resolve(match.group(0))

    return None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def fetch_circular_text(detail_url: str) -> tuple[str | None, str | None]:
    """Returns (extracted_text, error_message). Exactly one will be None."""
    try:
        resp = requests.get(detail_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return None, f"Failed to fetch detail page: {e}"

    pdf_url = extract_pdf_url(resp.text)
    if not pdf_url:
        # Save the raw HTML for the first few failures so we can inspect
        # exactly what the server sent, rather than guessing again.
        debug_dir = RAW_TEXT_DIR / "_debug_html"
        debug_dir.mkdir(exist_ok=True)
        existing_debug = list(debug_dir.glob("*.html"))
        if len(existing_debug) < 5:
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", detail_url.split("/")[-1])[:80]
            with open(debug_dir / f"{slug}.html", "w", encoding="utf-8") as f:
                f.write(resp.text)
        return None, "No PDF link found on detail page"

    try:
        pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
        pdf_resp.raise_for_status()
    except Exception as e:
        return None, f"Failed to download PDF ({pdf_url}): {e}"

    try:
        text = extract_pdf_text(pdf_resp.content)
    except Exception as e:
        return None, f"Failed to extract text from PDF ({pdf_url}): {e}"

    if not text or len(text.strip()) < 20:
        return None, f"Extracted text suspiciously short/empty ({pdf_url})"

    return text, None


def load_failed_log() -> dict:
    if FAILED_LOG.exists():
        with open(FAILED_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_failed_log(failed: dict):
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)


def process_all():
    json_files = sorted(RAW_TEXT_DIR.glob("*.json"))
    json_files = [f for f in json_files if f.name != "_checkpoint.json"
                  and f.name != "_failed_fetches.json"]

    print(f"Found {len(json_files)} circular records to process.\n")

    failed = load_failed_log()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for i, fpath in enumerate(json_files, 1):
        with open(fpath, "r", encoding="utf-8") as f:
            record = json.load(f)

        if record.get("text"):
            n_skipped += 1
            continue  # already has text from a previous run

        text, error = fetch_circular_text(record["url"])

        if text:
            record["text"] = text
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            n_done += 1
        else:
            failed[record["url"]] = error
            n_failed += 1

        if i % 50 == 0:
            print(f"Progress: {i}/{len(json_files)} "
                  f"(done={n_done}, skipped={n_skipped}, failed={n_failed})")
            save_failed_log(failed)  # checkpoint the failed log periodically

        time.sleep(POLITE_DELAY)

    save_failed_log(failed)

    print(f"\n=== Done ===")
    print(f"Fetched text successfully: {n_done}")
    print(f"Already had text (skipped): {n_skipped}")
    print(f"Failed: {n_failed}")
    if n_failed:
        print(f"See {FAILED_LOG} for details on failures -- these need manual review.")


if __name__ == "__main__":
    process_all()
    