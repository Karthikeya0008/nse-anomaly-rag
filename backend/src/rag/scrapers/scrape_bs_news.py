"""
Business Standard Markets News Scraper (Stage 1)
---------------------------------------------------
Paginates through Business Standard markets-journalist author/topic pages
and collects article title/date/url.

Stops per-author once past START_DATE.
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime

import requests


OUTPUT_DIR = Path(__file__).resolve().parents[4] / "db" / "raw_text" / "news_bs"
CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint.json"

AUTHOR_SLUGS = [
    "nifty",
    "sensex",
    "reserve-bank-of-india",
    "sebi",
    "stock-market-crash",
]

BASE = "https://www.business-standard.com"

START_DATE = datetime(2016, 8, 1)
MAX_PAGES_PER_AUTHOR = 300

POLITE_DELAY = 4.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 60
COOLDOWN_AFTER_BLOCK_SECONDS = 900


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_with_retry(url: str):
    """
    GET a URL, retrying with growing backoff on 403/429 and other
    request failures.
    """

    for cooldown_round in range(2):
        delay = RETRY_BACKOFF_SECONDS
        last_exc = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=20,
                )

                if resp.status_code in (403, 429):
                    if attempt < MAX_RETRIES:
                        print(
                            f"  got {resp.status_code}, backing off "
                            f"{delay}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(delay)
                        delay *= 2
                        continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.RequestException as e:
                last_exc = e

                if attempt < MAX_RETRIES:
                    print(
                        f"  request failed ({e}), backing off "
                        f"{delay}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    delay *= 2

        if cooldown_round == 0:
            print(
                "  retries exhausted -- this looks like a sustained "
                "block, not a transient rate limit. "
                f"Cooling down for "
                f"{COOLDOWN_AFTER_BLOCK_SECONDS // 60} min "
                "before one more try..."
            )

            time.sleep(COOLDOWN_AFTER_BLOCK_SECONDS)

    raise last_exc


NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>'
    r"(.*?)"
    r"</script>",
    re.DOTALL,
)


def parse_author_page(html: str):
    """
    Extract article rows from the embedded __NEXT_DATA__ JSON blob.

    Business Standard may return different structures for `data`:
        data = {"rows": [...]}
    or:
        data = [...]

    This function handles both.
    """

    match = NEXT_DATA_RE.search(html)

    if not match:
        print("  __NEXT_DATA__ not found")
        return []

    try:
        data = json.loads(match.group(1))

        page_props = data.get("props", {}).get("pageProps", {})
        page_data = page_props.get("data", {})

        # Normal structure:
        # data = {"rows": [...]}
        if isinstance(page_data, dict):
            rows = page_data.get("rows", [])

        # Some Business Standard pages return:
        # data = [...]
        elif isinstance(page_data, list):
            rows = page_data

        else:
            rows = []

    except (TypeError, AttributeError, json.JSONDecodeError) as e:
        print(f"  failed to parse __NEXT_DATA__: {e}")
        return []

    if not isinstance(rows, list):
        print(f"  unexpected rows type: {type(rows)}")
        return []

    results = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        title = (
            row.get("heading1")
            or row.get("title")
            or row.get("headline")
        )

        url_path = (
            row.get("article_url")
            or row.get("url")
            or row.get("articleUrl")
        )

        ts = (
            row.get("published_date")
            or row.get("publishedDate")
            or row.get("timestamp")
        )

        if not title or not url_path or not ts:
            continue

        # Handle timestamp safely.
        try:
            ts = int(ts)

            # If timestamp is milliseconds instead of seconds.
            if ts > 10_000_000_000:
                ts = ts / 1000

            date_obj = datetime.fromtimestamp(ts)

        except (TypeError, ValueError, OverflowError):
            continue

        # Build absolute URL.
        if isinstance(url_path, str):
            if url_path.startswith("http://") or url_path.startswith("https://"):
                url = url_path
            else:
                if not url_path.startswith("/"):
                    url_path = "/" + url_path

                url = BASE + url_path
        else:
            continue

        results.append(
            {
                "date": date_obj,
                "title": str(title).strip(),
                "url": url,
            }
        )

    return results


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(
                CHECKPOINT_FILE,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)

        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load checkpoint: {e}")

    return {}


def save_checkpoint(cp: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cp, f, indent=2)


def save_article(entry: dict):
    # Remove problematic characters from filename.
    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        entry["url"].split("/")[-1],
    ).strip("-")[:100]

    if not slug:
        slug = "article"

    out_path = OUTPUT_DIR / f"{entry['date']}_{slug}.json"

    if out_path.exists():
        return False

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            entry,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return True


def scrape_author(slug: str, checkpoint: dict):
    start_page = checkpoint.get(slug, 1)

    page = start_page
    total_saved = 0

    already_have = len(
        [
            f
            for f in OUTPUT_DIR.glob("*.json")
            if not f.name.startswith("_")
        ]
    )

    while page <= MAX_PAGES_PER_AUTHOR:

        if page == 1:
            url = f"{BASE}/topic/{slug}"
        else:
            url = f"{BASE}/topic/{slug}/page-{page}"

        print(f"  fetching page {page}: {url}")

        try:
            resp = fetch_with_retry(url)

        except Exception as e:
            print(
                f"  page {page}: fetch failed after retries "
                f"({e}), stopping this author"
            )
            break

        entries = parse_author_page(resp.text)

        if not entries:
            print(
                f"  page {page}: no entries parsed, stopping this "
                f"author (reached end, or page structure differs)"
            )
            break

        stop = False

        for entry in entries:

            if entry["date"] < START_DATE:
                stop = True
                continue

            record = {
                "source": "business_standard",
                "date": entry["date"].strftime("%Y-%m-%d"),
                "title": entry["title"],
                "text": "",
                "url": entry["url"],
                "sector_tags": [],
            }

            if save_article(record):
                total_saved += 1

        # Save checkpoint after successfully processing the page.
        checkpoint[slug] = page + 1
        save_checkpoint(checkpoint)

        print(
            f"  page {page}: saved "
            f"(new this run: {total_saved}, "
            f"total on disk: {already_have + total_saved})"
        )

        if stop:
            print(
                f"  reached articles older than "
                f"{START_DATE.date()} -- stopping this author"
            )
            break

        page += 1

        time.sleep(POLITE_DELAY)

    return total_saved


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = load_checkpoint()

    grand_total = 0

    for slug in AUTHOR_SLUGS:

        print(
            f"\n=== Scraping author: {slug} ==="
        )

        grand_total += scrape_author(
            slug,
            checkpoint,
        )

    print(
        f"\nDone. Total new articles saved: "
        f"{grand_total}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )