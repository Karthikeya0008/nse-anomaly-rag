"""
Moneycontrol Nifty Indices News Scraper
=======================================

Stage 1:
    Discover Moneycontrol article URLs
    -> Fetch individual article pages
    -> Extract publication date
    -> Identify Nifty indices
    -> Save metadata JSON

Target period:
    2016-08-01 -> 2021-12-31

Output:
    db/raw_text/news_mc/

Example:

{
    "source": "moneycontrol",
    "date": "2020-03-18",
    "title": "What should investors do as Nifty falls...",
    "text": "",
    "url": "...",
    "indices": ["NIFTY 50"],
    "discovery_source": "nifty_50_tag"
}

Stage 2 can then fetch the article text.
"""

import json
import re
import time

from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIG
# ============================================================

START_DATE = datetime(
    2016,
    8,
    1,
)

END_DATE = datetime(
    2021,
    12,
    31,
)


# How many listing pages to inspect per discovery source.
MAX_PAGES_PER_SOURCE = 1000


# Delay between requests.
# Keep this reasonably high.
REQUEST_DELAY = 2.5


# Number of retries for failed requests.
MAX_RETRIES = 3


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "db"
    / "raw_text"
    / "news_mc"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "_checkpoint.json"
)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),

    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),

    "Accept-Language":
        "en-US,en;q=0.9",

    "Connection":
        "keep-alive",
}


SESSION = requests.Session()

SESSION.headers.update(
    HEADERS
)


# ============================================================
# MONEYCONTROL DISCOVERY SOURCES
# ============================================================

DISCOVERY_SOURCES = {

    # Main Nifty tag
    "nifty_50_tag":
        "https://www.moneycontrol.com/news/tags/nifty-50.html",

    # Generic Nifty tag
    "nifty_tag":
        "https://www.moneycontrol.com/news/tags/nifty.html",

    # Alternative spelling
    "nifty50_tag":
        "https://www.moneycontrol.com/news/tags/nifty50.html",

    # Bank Nifty
    "bank_nifty_tag":
        "https://www.moneycontrol.com/news/tags/bank-nifty.html",

    # Nifty Index
    "nifty_index_tag":
        "https://www.moneycontrol.com/news/tags/nifty-index.html",

    # General markets section
    "markets":
        "https://www.moneycontrol.com/news/business/markets/",
}


# ============================================================
# NIFTY INDEX DICTIONARY
# ============================================================

NIFTY_INDEXES = {

    "NIFTY 50": [
        "nifty 50",
        "nifty50",
        "nifty",
    ],

    "NIFTY NEXT 50": [
        "nifty next 50",
        "nifty next",
    ],

    "NIFTY 100": [
        "nifty 100",
    ],

    "NIFTY 200": [
        "nifty 200",
    ],

    "NIFTY 500": [
        "nifty 500",
    ],

    "NIFTY MIDCAP 50": [
        "nifty midcap 50",
    ],

    "NIFTY MIDCAP 100": [
        "nifty midcap 100",
    ],

    "NIFTY MIDCAP 150": [
        "nifty midcap 150",
    ],

    "NIFTY SMALLCAP 50": [
        "nifty smallcap 50",
    ],

    "NIFTY SMALLCAP 100": [
        "nifty smallcap 100",
    ],

    "NIFTY SMALLCAP 250": [
        "nifty smallcap 250",
    ],

    "NIFTY BANK": [
        "nifty bank",
        "bank nifty",
        "banknifty",
    ],

    "NIFTY FINANCIAL SERVICES": [
        "nifty financial services",
        "nifty finserv",
        "nifty financial",
    ],

    "NIFTY IT": [
        "nifty it",
    ],

    "NIFTY AUTO": [
        "nifty auto",
    ],

    "NIFTY PHARMA": [
        "nifty pharma",
    ],

    "NIFTY FMCG": [
        "nifty fmcg",
    ],

    "NIFTY METAL": [
        "nifty metal",
    ],

    "NIFTY REALTY": [
        "nifty realty",
    ],

    "NIFTY ENERGY": [
        "nifty energy",
    ],

    "NIFTY PSU BANK": [
        "nifty psu bank",
    ],

    "NIFTY PRIVATE BANK": [
        "nifty private bank",
    ],

    "NIFTY MEDIA": [
        "nifty media",
    ],

    "NIFTY INFRASTRUCTURE": [
        "nifty infrastructure",
        "nifty infra",
    ],

    "NIFTY CONSUMPTION": [
        "nifty consumption",
    ],

    "NIFTY PSE": [
        "nifty pse",
    ],

    "NIFTY MNC": [
        "nifty mnc",
    ],

    "NIFTY CPSE": [
        "nifty cpse",
    ],
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(
    text
):

    if not text:
        return ""

    text = str(
        text
    ).lower()

    text = text.replace(
        "\xa0",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(
    url
):

    if not url:
        return None

    url = url.strip()

    if not url.startswith(
        "http"
    ):
        url = urljoin(
            "https://www.moneycontrol.com",
            url,
        )

    parsed = urlparse(
        url
    )

    if "moneycontrol.com" not in parsed.netloc.lower():
        return None

    path = parsed.path

    if not path:
        return None

    # Remove query string / fragment.
    clean_url = (
        "https://www.moneycontrol.com"
        + path
    )

    return clean_url.rstrip("/")


# ============================================================
# ARTICLE URL TEST
# ============================================================

def is_article_url(
    url
):

    if not url:
        return False

    parsed = urlparse(
        url
    )

    if "moneycontrol.com" not in parsed.netloc.lower():
        return False

    path = parsed.path.lower()

    # Moneycontrol news articles.
    if not path.startswith(
        "/news/"
    ):
        return False

    # Things that are not individual articles.
    excluded = [
        "/tags/",
        "/search/",
        "/video/",
        "/videos/",
        "/photos/",
        "/podcast/",
        "/live-tv/",
        "/author/",
        "/category/",
    ]

    for item in excluded:

        if item in path:
            return False

    # Need enough URL components.
    parts = [
        x for x in path.split("/")
        if x
    ]

    if len(parts) < 3:
        return False

    return True


# ============================================================
# HTTP FETCH
# ============================================================

def fetch_url(
    url
):

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = SESSION.get(
                url,
                timeout=30,
            )

            status = (
                response.status_code
            )

            if status == 404:

                return None

            if status in (
                403,
                429,
            ):

                wait = (
                    30
                    * (attempt + 1)
                )

                print(
                    f"    HTTP {status}; "
                    f"waiting {wait}s"
                )

                time.sleep(
                    wait
                )

                continue

            if status >= 500:

                wait = (
                    15
                    * (attempt + 1)
                )

                print(
                    f"    HTTP {status}; "
                    f"retrying in {wait}s"
                )

                time.sleep(
                    wait
                )

                continue

            response.raise_for_status()

            return response

        except requests.RequestException as e:

            if attempt == MAX_RETRIES - 1:

                print(
                    f"    request failed: {e}"
                )

                return None

            wait = (
                10
                * (attempt + 1)
            )

            print(
                f"    request error; "
                f"retrying in {wait}s"
            )

            time.sleep(
                wait
            )

    return None


# ============================================================
# DATE PARSING
# ============================================================

def parse_date_string(
    value
):

    if not value:
        return None

    value = str(
        value
    ).strip()

    # ISO timestamps:
    #
    # 2020-03-18T20:19:00+05:30
    # 2020-03-18T20:19:00
    #
    iso_match = re.search(
        r"(20\d{2}-\d{2}-\d{2})",
        value,
    )

    if iso_match:

        try:

            return datetime.strptime(
                iso_match.group(1),
                "%Y-%m-%d",
            )

        except ValueError:
            pass

    formats = [

        "%B %d, %Y",
        "%b %d, %Y",

        "%B %d %Y",
        "%b %d %Y",

        "%d %B %Y",
        "%d %b %Y",

        "%Y/%m/%d",
        "%Y-%m-%d",

    ]

    # Remove IST etc.
    cleaned = re.sub(
        r"\s+(IST|UTC|GMT)$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    for fmt in formats:

        try:

            return datetime.strptime(
                cleaned,
                fmt,
            )

        except ValueError:
            continue

    return None


# ============================================================
# DATE FROM TEXT
# ============================================================

def extract_date_from_text(
    text
):

    if not text:
        return None

    text = str(
        text
    )

    patterns = [

        # January 15, 2020
        (
            r"\b"
            r"(January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r"\s+\d{1,2},\s+\d{4}"
            r"\b"
        ),

        # Jan 15, 2020
        (
            r"\b"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|"
            r"Oct|Nov|Dec)"
            r"\.?\s+\d{1,2},\s+\d{4}"
            r"\b"
        ),

        # 15 January 2020
        (
            r"\b"
            r"\d{1,2}\s+"
            r"(January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r"\s+\d{4}"
            r"\b"
        ),

        # 15 Jan 2020
        (
            r"\b"
            r"\d{1,2}\s+"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|"
            r"Oct|Nov|Dec)"
            r"\.?\s+\d{4}"
            r"\b"
        ),

        # 2020-01-15
        (
            r"\b20\d{2}-\d{2}-\d{2}\b"
        ),
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        date = parse_date_string(
            match.group(0)
        )

        if date:
            return date

    return None


# ============================================================
# DATE FROM HTML ELEMENT
# ============================================================

def extract_date_from_element(
    element
):

    if element is None:
        return None

    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    attributes = [
        "datetime",
        "data-date",
        "data-datetime",
        "data-published",
        "data-published-date",
        "data-publish-date",
        "data-time",
        "content",
        "value",
    ]

    for attr in attributes:

        value = element.get(
            attr
        )

        if not value:
            continue

        date = parse_date_string(
            value
        )

        if date:
            return date

        date = extract_date_from_text(
            value
        )

        if date:
            return date

    # --------------------------------------------------------
    # <time>
    # --------------------------------------------------------

    for time_element in element.find_all(
        "time"
    ):

        for attr in attributes:

            value = time_element.get(
                attr
            )

            if not value:
                continue

            date = parse_date_string(
                value
            )

            if date:
                return date

            date = extract_date_from_text(
                value
            )

            if date:
                return date

        date = extract_date_from_text(
            time_element.get_text(
                " ",
                strip=True,
            )
        )

        if date:
            return date

    return None


# ============================================================
# DATE FROM URL
# ============================================================

def extract_date_from_url(
    url
):

    if not url:
        return None

    patterns = [

        # 2020/03/18
        (
            r"/(20\d{2})/"
            r"(\d{1,2})/"
            r"(\d{1,2})/"
        ),

        # 2020-03-18
        (
            r"/(20\d{2})-"
            r"(\d{1,2})-"
            r"(\d{1,2})"
        ),

        # 18-mar-2020
        (
            r"/(\d{1,2})-"
            r"(jan|feb|mar|apr|may|jun|jul|aug|"
            r"sep|oct|nov|dec)-"
            r"(20\d{2})"
        ),
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        groups = match.groups()

        try:

            # YYYY MM DD
            if (
                len(groups) == 3
                and groups[0].isdigit()
            ):

                return datetime(
                    int(groups[0]),
                    int(groups[1]),
                    int(groups[2]),
                )

        except Exception:
            pass

    return None


# ============================================================
# JSON-LD DATE EXTRACTION
# ============================================================

def extract_jsonld_date(
    soup
):

    scripts = soup.find_all(
        "script",
        type="application/ld+json",
    )

    for script in scripts:

        raw = script.string

        if not raw:
            continue

        try:

            data = json.loads(
                raw
            )

        except Exception:
            continue

        objects = []

        if isinstance(
            data,
            dict,
        ):

            objects.append(
                data
            )

            graph = data.get(
                "@graph"
            )

            if isinstance(
                graph,
                list,
            ):

                objects.extend(
                    graph
                )

        elif isinstance(
            data,
            list,
        ):

            objects.extend(
                data
            )

        for obj in objects:

            if not isinstance(
                obj,
                dict,
            ):
                continue

            for key in (
                "datePublished",
                "dateCreated",
                "dateModified",
            ):

                value = obj.get(
                    key
                )

                if not value:
                    continue

                date = parse_date_string(
                    value
                )

                if date:
                    return date

    return None


# ============================================================
# META DATE EXTRACTION
# ============================================================

def extract_meta_date(
    soup
):

    meta_candidates = [

        {
            "property":
                "article:published_time"
        },

        {
            "property":
                "article:modified_time"
        },

        {
            "name":
                "publish-date"
        },

        {
            "name":
                "published_date"
        },

        {
            "name":
                "date"
        },

        {
            "itemprop":
                "datePublished"
        },

        {
            "itemprop":
                "dateCreated"
        },
    ]

    for attrs in meta_candidates:

        element = soup.find(
            "meta",
            attrs=attrs,
        )

        if not element:
            continue

        value = element.get(
            "content"
        )

        if not value:
            continue

        date = parse_date_string(
            value
        )

        if date:
            return date

    return None


# ============================================================
# ARTICLE DATE
# ============================================================

def extract_article_date(
    soup,
    url
):

    # --------------------------------------------------------
    # 1. JSON-LD
    # --------------------------------------------------------

    date = extract_jsonld_date(
        soup
    )

    if date:
        return date

    # --------------------------------------------------------
    # 2. Meta tags
    # --------------------------------------------------------

    date = extract_meta_date(
        soup
    )

    if date:
        return date

    # --------------------------------------------------------
    # 3. <time>
    # --------------------------------------------------------

    for time_element in soup.find_all(
        "time"
    ):

        date = extract_date_from_element(
            time_element
        )

        if date:
            return date

    # --------------------------------------------------------
    # 4. Elements with date-related classes
    # --------------------------------------------------------

    date_class_patterns = [
        "date",
        "publish",
        "published",
        "timestamp",
        "time",
    ]

    for element in soup.find_all(
        True
    ):

        classes = element.get(
            "class",
            [],
        )

        if not classes:
            continue

        class_text = (
            " ".join(classes)
            .lower()
        )

        if not any(
            x in class_text
            for x in date_class_patterns
        ):
            continue

        date = extract_date_from_element(
            element
        )

        if date:
            return date

    # --------------------------------------------------------
    # 5. Search page text
    # --------------------------------------------------------

    # Don't search the entire page blindly for arbitrary
    # dates. Look near likely article metadata first.

    selectors = [
        ".article_info",
        ".article_author",
        ".article-details",
        ".author_details",
        ".author",
        ".published",
        ".publish",
        ".date",
        ".article_date",
        "[class*='date']",
        "[class*='publish']",
    ]

    for selector in selectors:

        try:

            elements = soup.select(
                selector
            )

        except Exception:
            continue

        for element in elements:

            date = extract_date_from_element(
                element
            )

            if date:
                return date

            date = extract_date_from_text(
                element.get_text(
                    " ",
                    strip=True,
                )
            )

            if date:
                return date

    # --------------------------------------------------------
    # 6. URL fallback
    # --------------------------------------------------------

    date = extract_date_from_url(
        url
    )

    if date:
        return date

    return None


# ============================================================
# ARTICLE TITLE
# ============================================================

def extract_article_title(
    soup
):

    # --------------------------------------------------------
    # JSON-LD
    # --------------------------------------------------------

    scripts = soup.find_all(
        "script",
        type="application/ld+json",
    )

    for script in scripts:

        if not script.string:
            continue

        try:

            data = json.loads(
                script.string
            )

        except Exception:
            continue

        objects = []

        if isinstance(
            data,
            dict,
        ):

            objects.append(
                data
            )

            graph = data.get(
                "@graph"
            )

            if isinstance(
                graph,
                list,
            ):

                objects.extend(
                    graph
                )

        elif isinstance(
            data,
            list,
        ):

            objects.extend(
                data
            )

        for obj in objects:

            if not isinstance(
                obj,
                dict,
            ):
                continue

            headline = obj.get(
                "headline"
            )

            if headline:
                return str(
                    headline
                ).strip()

    # --------------------------------------------------------
    # OpenGraph
    # --------------------------------------------------------

    meta = soup.find(
        "meta",
        property="og:title",
    )

    if meta:

        value = meta.get(
            "content"
        )

        if value:
            return value.strip()

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    h1 = soup.find(
        "h1"
    )

    if h1:

        value = h1.get_text(
            " ",
            strip=True,
        )

        if value:
            return value

    # --------------------------------------------------------
    # <title>
    # --------------------------------------------------------

    if soup.title:

        value = soup.title.get_text(
            " ",
            strip=True,
        )

        if value:
            return value

    return None


# ============================================================
# INDEX MATCHING
# ============================================================

def find_matching_indices(
    title
):

    if not title:
        return []

    text = normalize_text(
        title
    )

    matches = []

    # More specific indexes must be checked before generic
    # "NIFTY" so that Bank Nifty etc. are correctly classified.

    ordered_indexes = sorted(
        NIFTY_INDEXES.items(),
        key=lambda x: max(
            len(alias)
            for alias in x[1]
        ),
        reverse=True,
    )

    for canonical, aliases in ordered_indexes:

        for alias in aliases:

            alias = normalize_text(
                alias
            )

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(alias)
                + r"(?![a-z0-9])"
            )

            if re.search(
                pattern,
                text,
            ):

                matches.append(
                    canonical
                )

                break

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(
            matches
        )
    )


# ============================================================
# LISTING PAGE ARTICLE DISCOVERY
# ============================================================

def discover_article_urls(
    html
):

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    results = []

    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):

        url = normalize_url(
            anchor.get(
                "href"
            )
        )

        if not is_article_url(
            url
        ):
            continue

        if url in seen:
            continue

        title = anchor.get_text(
            " ",
            strip=True,
        )

        title = re.sub(
            r"\s+",
            " ",
            title,
        ).strip()

        results.append(
            {
                "url": url,
                "listing_title": title,
            }
        )

        seen.add(url)

    return results


# ============================================================
# PAGINATION URL
# ============================================================

def page_url(
    base_url,
    page
):

    if page == 1:
        return base_url

    return (
        base_url.rstrip("/")
        + f"/page-{page}/"
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():
        return {}

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(
                f
            )

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception as e:

        print(
            f"Warning: checkpoint "
            f"could not be read: {e}"
        )

    return {}


def save_checkpoint(
    checkpoint
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=2,
        )


# ============================================================
# EXISTING URL INDEX
# ============================================================

def load_existing_urls():

    existing = set()

    if not OUTPUT_DIR.exists():
        return existing

    for path in OUTPUT_DIR.glob(
        "*.json"
    ):

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(
                    f
                )

            url = data.get(
                "url"
            )

            if url:
                existing.add(
                    normalize_url(
                        url
                    )
                )

        except Exception:
            continue

    return existing


# ============================================================
# OUTPUT FILENAME
# ============================================================

def make_output_path(
    date,
    url
):

    slug = (
        url
        .rstrip("/")
        .split("/")[-1]
    )

    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        slug,
    )

    slug = slug.strip(
        "-"
    )

    if not slug:
        slug = "article"

    slug = slug[
        :150
    ]

    return (
        OUTPUT_DIR
        / f"{date.strftime('%Y-%m-%d')}"
        f"_{slug}.json"
    )


# ============================================================
# SAVE ARTICLE
# ============================================================

def save_article(
    article,
    existing_urls
):

    url = normalize_url(
        article["url"]
    )

    if url in existing_urls:
        return False

    date = article.get(
        "date"
    )

    if not date:
        return False

    output_path = make_output_path(
        date,
        url,
    )

    # Avoid filename collision.
    if output_path.exists():

        counter = 2

        while True:

            candidate = (
                output_path.parent
                / (
                    output_path.stem
                    + f"_{counter}"
                    + output_path.suffix
                )
            )

            if not candidate.exists():

                output_path = candidate

                break

            counter += 1

    record = {

        "source":
            "moneycontrol",

        "date":
            date.strftime(
                "%Y-%m-%d"
            ),

        "title":
            article["title"],

        "text":
            "",

        "url":
            url,

        "indices":
            article.get(
                "indices",
                [],
            ),

        "discovery_source":
            article.get(
                "discovery_source"
            ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            record,
            f,
            ensure_ascii=False,
            indent=2,
        )

    existing_urls.add(
        url
    )

    return True


# ============================================================
# PROCESS ONE ARTICLE
# ============================================================

def process_article(
    url,
    discovery_source
):

    response = fetch_url(
        url
    )

    if response is None:
        return None

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date = extract_article_date(
        soup,
        url,
    )

    if date is None:

        return {
            "status":
                "no_date",
            "url":
                url,
        }

    # --------------------------------------------------------
    # Date filter
    # --------------------------------------------------------

    if date < START_DATE:

        return {
            "status":
                "too_old",
            "url":
                url,
            "date":
                date,
        }

    if date > END_DATE:

        return {
            "status":
                "too_new",
            "url":
                url,
            "date":
                date,
        }

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = extract_article_title(
        soup
    )

    if not title:
        return {
            "status":
                "no_title",
            "url":
                url,
            "date":
                date,
        }

    # --------------------------------------------------------
    # Index classification
    # --------------------------------------------------------

    indices = find_matching_indices(
        title
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "status":
            "valid",

        "date":
            date,

        "title":
            title,

        "url":
            url,

        "indices":
            indices,

        "discovery_source":
            discovery_source,
    }


# ============================================================
# SCRAPE DISCOVERY SOURCE
# ============================================================

def scrape_source(
    source_name,
    base_url,
    checkpoint,
    existing_urls,
):

    source_checkpoint = checkpoint.get(
        source_name,
        {},
    )

    start_page = source_checkpoint.get(
        "page",
        1,
    )

    processed_urls = set(
        source_checkpoint.get(
            "processed_urls",
            [],
        )
    )

    saved = 0

    print()
    print(
        "=" * 75
    )

    print(
        f"SOURCE: {source_name}"
    )

    print(
        "=" * 75
    )

    for page in range(
        start_page,
        MAX_PAGES_PER_SOURCE + 1,
    ):

        listing_url = page_url(
            base_url,
            page,
        )

        print()
        print(
            f"[{source_name}] "
            f"page {page}"
        )

        print(
            f"  {listing_url}"
        )

        response = fetch_url(
            listing_url
        )

        if response is None:

            print(
                "  Could not fetch listing page."
            )

            break

        discovered = discover_article_urls(
            response.text
        )

        print(
            f"  discovered: "
            f"{len(discovered)} article URLs"
        )

        if not discovered:

            print(
                "  No article URLs found."
            )

            print(
                "  Stopping this source."
            )

            break

        page_new = 0

        for item in discovered:

            url = item["url"]

            if url in processed_urls:

                continue

            print(
                f"    -> {url}"
            )

            result = process_article(
                url,
                source_name,
            )

            processed_urls.add(
                url
            )

            # ------------------------------------------------
            # Save checkpoint after EVERY article.
            # This is important if Moneycontrol blocks or
            # connection drops halfway through a page.
            # ------------------------------------------------

            checkpoint[
                source_name
            ] = {
                "page":
                    page,

                "processed_urls":
                    list(
                        processed_urls
                    ),
            }

            save_checkpoint(
                checkpoint
            )

            if result is None:
                continue

            status = result.get(
                "status"
            )

            # ------------------------------------------------
            # Valid article
            # ------------------------------------------------

            if status == "valid":

                # IMPORTANT:
                # We only save articles that explicitly mention
                # at least one Nifty index in the title.
                #
                # This keeps Stage 1 focused.
                if not result[
                    "indices"
                ]:

                    print(
                        "       valid date, "
                        "but no Nifty index "
                        "in title"
                    )

                    time.sleep(
                        REQUEST_DELAY
                    )

                    continue

                if save_article(
                    result,
                    existing_urls,
                ):

                    saved += 1
                    page_new += 1

                    print(
                        f"       SAVED | "
                        f"{result['date'].date()} | "
                        f"{result['indices']}"
                    )

                else:

                    print(
                        "       already exists"
                    )

            elif status == "no_date":

                print(
                    "       no date found"
                )

            elif status == "too_old":

                print(
                    f"       too old: "
                    f"{result['date'].date()}"
                )

            elif status == "too_new":

                print(
                    f"       too new: "
                    f"{result['date'].date()}"
                )

            elif status == "no_title":

                print(
                    "       no title found"
                )

            time.sleep(
                REQUEST_DELAY
            )

        print(
            f"  saved from page: "
            f"{page_new}"
        )

        # ----------------------------------------------------
        # Advance checkpoint to next page.
        # ----------------------------------------------------

        checkpoint[
            source_name
        ] = {
            "page":
                page + 1,

            "processed_urls":
                list(
                    processed_urls
                ),
        }

        save_checkpoint(
            checkpoint
        )

    return saved


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 75
    )

    print(
        "MONEYCONTROL NIFTY INDEX SCRAPER"
    )

    print(
        f"Date range: "
        f"{START_DATE.date()} -> "
        f"{END_DATE.date()}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint()

    # --------------------------------------------------------
    # Existing articles
    # --------------------------------------------------------

    existing_urls = load_existing_urls()

    print(
        f"Existing articles: "
        f"{len(existing_urls)}"
    )

    total_saved = 0

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    for source_name, base_url in (
        DISCOVERY_SOURCES.items()
    ):

        try:

            saved = scrape_source(
                source_name,
                base_url,
                checkpoint,
                existing_urls,
            )

            total_saved += saved

        except KeyboardInterrupt:

            print()
            print(
                "Interrupted by user."
            )

            print(
                "Checkpoint saved."
            )

            return

        except Exception as e:

            print()
            print(
                f"ERROR in "
                f"{source_name}: {e}"
            )

            print(
                "Moving to next source..."
            )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print(
        "=" * 75
    )

    print(
        "DONE"
    )

    print(
        f"New articles saved: "
        f"{total_saved}"
    )

    print(
        f"Total articles now: "
        f"{len(existing_urls)}"
    )

    print(
        f"Output directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()