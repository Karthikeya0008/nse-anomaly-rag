"""
SEBI Circulars Scraper
----------------------
Scrapes the SEBI circulars listing (https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0)
by driving a real browser (Selenium), since pagination is JavaScript-triggered
and there is no confirmed public AJAX/query-string endpoint to hit directly.

Output: one JSON file per circular in db/raw_text/sebi/, matching the common
Dataset B schema: source, date, title, text, url, sector_tags (empty for now,
filled in later during ingestion).

IMPORTANT CAVEATS before you run this:
1. You need Google Chrome installed. webdriver-manager auto-downloads the
   matching chromedriver, but Chrome itself must already be on your machine.
2. SEBI's site structure could change or the table/link selectors below may
   need adjustment. If the scraper collects 0 circulars on the first page,
   open the site in Chrome, right-click a circular row -> Inspect, and check
   whether it's still a <table> with <a> tags inside <td> cells (that's what
   this script assumes). Update the CSS selectors in `parse_page` if not.
3. This is a "be polite" scraper -- it sleeps between page loads. Do not
   remove the sleep or crank concurrency; SEBI's site is a shared government
   resource and hammering it can get your IP rate-limited or blocked.
4. Full run across ~10 years may take a while (each page load = a few
   seconds, doing this sequentially, not in parallel). Let it run in the
   background; it checkpoints as it goes (see CHECKPOINT_FILE) so a crash
   partway through does not lose earlier progress.
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).resolve().parents[4] / "db" / "raw_text" / "sebi"
CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint.json"

LISTING_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0"

# Your Dataset A window -- stop scraping once we page back past this date,
# since the listing is sorted newest-first.
START_DATE = datetime(2016, 8, 1)

# Seconds to wait between page loads. Do not lower this much.
POLITE_DELAY = 2.5


def setup_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def parse_date(date_str: str):
    """SEBI dates appear like 'Aug 28, 2026'."""
    date_str = date_str.strip()
    try:
        return datetime.strptime(date_str, "%b %d, %Y")
    except ValueError:
        return None


def parse_page(html: str):
    """Extract (date, title, url) tuples from one listing page's HTML.

    ASSUMPTION: circulars are rendered as table rows, each containing a date
    cell and a title cell with an <a href> link. If SEBI changes their markup,
    this is the function to fix -- inspect the live page and adjust the
    selector below.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        date_text = cells[0].get_text(strip=True)
        date_obj = parse_date(date_text)
        if not date_obj:
            continue  # header row or malformed row, skip

        link = cells[1].find("a", href=True)
        if not link:
            continue

        title = link.get_text(strip=True)
        url = link["href"]
        if url.startswith("/"):
            url = "https://www.sebi.gov.in" + url

        results.append({"date": date_obj, "title": title, "url": url})

    return results


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_page_scraped": 0, "seen_urls": []}


def save_checkpoint(page_num: int, seen_urls: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_page_scraped": page_num, "seen_urls": list(seen_urls)}, f)


def save_circular(entry: dict):
    """Save one circular as a JSON file, filename derived from a slug of its URL."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", entry["url"].split("/")[-1])[:100]
    out_path = OUTPUT_DIR / f"{entry['date']}_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


def scrape_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    seen_urls = set(checkpoint["seen_urls"])
    start_page = checkpoint["last_page_scraped"]

    driver = setup_driver(headless=True)
    driver.get(LISTING_URL)
    time.sleep(POLITE_DELAY)

    # Fast-forward to the checkpointed page if resuming
    for _ in range(start_page):
        try:
            next_link = driver.find_element(By.PARTIAL_LINK_TEXT, "Next")
            driver.execute_script("arguments[0].click();", next_link)
            time.sleep(POLITE_DELAY)
        except Exception as e:
            print(f"Could not fast-forward to page {start_page}: {e}")
            break

    page_num = start_page
    total_saved = 0
    stop = False

    while not stop:
        html = driver.page_source
        entries = parse_page(html)

        if not entries:
            print(f"Page {page_num}: no entries parsed -- check selectors "
                  f"in parse_page() against the live page structure.")
            break

        for entry in entries:
            if entry["url"] in seen_urls:
                continue
            if entry["date"] < START_DATE:
                stop = True
                continue

            record = {
                "source": "sebi",
                "date": entry["date"].strftime("%Y-%m-%d"),
                "title": entry["title"],
                "text": "",  # circular full text -- fetched separately, see note below
                "url": entry["url"],
                "sector_tags": []
            }
            save_circular(record)
            seen_urls.add(entry["url"])
            total_saved += 1

        print(f"Page {page_num}: saved {len(entries)} entries "
              f"(running total: {total_saved})")

        page_num += 1
        save_checkpoint(page_num, seen_urls)

        if stop:
            print(f"Reached circulars older than {START_DATE.date()} -- stopping.")
            break

        try:
            next_link = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Next"))
            )
            driver.execute_script("arguments[0].click();", next_link)
            time.sleep(POLITE_DELAY)
        except Exception as e:
            print(f"No more pages or 'Next' link not found: {e}")
            break

    driver.quit()
    print(f"\nDone. Total circulars saved: {total_saved}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    scrape_all()