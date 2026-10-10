"""
Wikipedia Event Scraper
-------------------------
Fetches full plain-text extracts of Wikipedia articles about major events
relevant to Indian markets, using Wikipedia's official API. Resumable --
skips titles already saved to disk.

UPDATE: added "2019 Indian general election" to specifically close a gap
found in real pipeline output -- the 2019-05-20 anomaly (Sensex +1422 pts)
is the exit-poll rally reaction, confirmed via search. Date set to
2019-05-19 (exit poll release date, the actual trigger) rather than the
May 23 results date.
"""

import json
import time
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).resolve().parents[4] / "db" / "raw_text" / "wikipedia"

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "NSE-Anomaly-Detection-Research/1.0 (student research project at VIT Vellore; https://github.com/)"
}
POLITE_DELAY = 3.0
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 15

ARTICLES = {
    "2016 Indian banknote demonetisation": "2016-11-08",
    "Goods and Services Tax (India)": "2017-07-01",
    "IL&FS": "2018-09-01",
    "2018 Indian infrastructure and banking crisis": "2018-09-01",
    "COVID-19 pandemic in India": "2020-01-30",
    "2020 stock market crash": "2020-03-01",
    "Yes Bank": "2020-03-05",
    "Adani Group": "2023-01-24",
    "Hindenburg Research": "2023-01-24",
    "2022 Russian invasion of Ukraine": "2022-02-24",
    "Union budget of India": "2020-02-01",
    "Reserve Bank of India": "2016-08-01",
    "Securities and Exchange Board of India": "2016-08-01",
    "National Stock Exchange of India": "2016-08-01",
    "2019 Indian general election": "2019-05-19",  # exit polls -> May 20 rally
}


def fetch_extract(title: str):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }

    delay = RETRY_BACKOFF_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                print(f"    request error ({e}), backing off {delay}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                delay *= 2
                continue
            print(f"    giving up on '{title}' after {MAX_RETRIES} retries ({e})")
            return None, None

        if resp.status_code == 429:
            if attempt < MAX_RETRIES:
                print(f"    got 429, backing off {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                delay *= 2
                continue
            print(f"    giving up on '{title}' after {MAX_RETRIES} retries (still rate-limited)")
            return None, None

        if resp.status_code != 200:
            print(f"    unexpected status {resp.status_code} for '{title}', skipping")
            return None, None

        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "missing" in page:
                return None, None
            text = page.get("extract", "")
            canonical_title = page.get("title", title)
            url = "https://en.wikipedia.org/wiki/" + canonical_title.replace(" ", "_")
            return text, url

        return None, None

    return None, None


def save_article(title: str, date: str, text: str, url: str):
    slug = title.replace(" ", "_").replace("/", "-")[:100]
    out_path = OUTPUT_DIR / f"{date}_{slug}.json"
    if out_path.exists():
        return False
    record = {
        "source": "wikipedia",
        "date": date,
        "title": title,
        "text": text,
        "url": url,
        "sector_tags": []
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return True


def scrape_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    found = 0
    missing = []

    for title, date in ARTICLES.items():
        slug = title.replace(" ", "_").replace("/", "-")[:100]
        if (OUTPUT_DIR / f"{date}_{slug}.json").exists():
            print(f"Already saved: '{title}' -- skipping")
            found += 1
            continue

        text, url = fetch_extract(title)

        if text is None:
            print(f"NOT FOUND / FAILED: '{title}' -- check exact spelling/title on Wikipedia")
            missing.append(title)
            time.sleep(POLITE_DELAY)
            continue

        save_article(title, date, text, url)
        print(f"Saved: '{title}' ({len(text)} chars) -> {date}")
        found += 1
        time.sleep(POLITE_DELAY)

    print(f"\n=== Done ===")
    print(f"Found and saved: {found}/{len(ARTICLES)}")
    if missing:
        print(f"Missing titles: {missing}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    scrape_all()
    