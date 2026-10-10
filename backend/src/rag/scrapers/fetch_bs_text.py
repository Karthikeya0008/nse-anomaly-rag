"""
Business Standard Article Text Fetcher (Stage 2)
------------------------------------------------

Reads article JSON records created by Stage 1:

    E:/nse-anomaly-rag/db/raw_text/news_bs/*.json

Supports:
    - Old Business Standard /article/... pages
    - New Business Standard article pages
    - /markets/news/... pages
    - /podcast/... pages
    - Old Stage 1 JSON records
    - New Stage 1 JSON records
    - __NEXT_DATA__ where data is dict OR list
    - JSON-LD articleBody
    - Older HTML article structures
    - Newer HTML article structures

It fills:

    "text": ""

with the extracted article body.

Failed URLs are stored in:

    _failed_text_fetches.json

Debug HTML for failed pages is stored in:

    _debug_html/
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup


# ============================================================
# PATHS
# ============================================================

RAW_TEXT_DIR = (
    Path(__file__).resolve().parents[4]
    / "db"
    / "raw_text"
    / "news_bs"
)

FAILED_LOG = RAW_TEXT_DIR / "_failed_text_fetches.json"
DEBUG_DIR = RAW_TEXT_DIR / "_debug_html"


# ============================================================
# REQUEST SETTINGS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

POLITE_DELAY = 3.0

MAX_RETRIES = 3

# For 403 / 429.
RETRY_BACKOFF_SECONDS = 30

# For 500 / 502 / 503 / 504.
SERVER_ERROR_DELAYS = [10, 20]

# Minimum acceptable article body.
MIN_TEXT_LENGTH = 200


# ============================================================
# NEXT DATA
# ============================================================

NEXT_DATA_RE = re.compile(
    r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>'
    r"(.*?)"
    r"</script>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# POSSIBLE ARTICLE CONTENT KEYS
# ============================================================

CONTENT_KEYS = {
    "htmlContent",
    "html_content",
    "articleBody",
    "article_body",
    "articleContent",
    "article_content",
    "content",
    "body",
    "storyBody",
    "story_body",
    "storyContent",
    "story_content",
    "description",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize extracted article text.
    """

    if not text:
        return None

    text = str(text)

    # HTML/Unicode whitespace.
    text = text.replace("\xa0", " ")

    # Zero-width characters.
    text = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        text,
    )

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
    )

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if line:
            lines.append(line)

    text = "\n\n".join(lines)

    return text.strip() if text.strip() else None


# ============================================================
# HTML -> TEXT
# ============================================================

def html_to_text(html_content):
    """
    Convert an HTML fragment to clean plain text.
    """

    if not isinstance(
        html_content,
        str,
    ):
        return None

    if not html_content.strip():
        return None

    try:

        soup = BeautifulSoup(
            html_content,
            "html.parser",
        )

        for tag in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "iframe",
                "svg",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator="\n\n",
            strip=True,
        )

        return clean_text(text)

    except Exception:
        return None


# ============================================================
# RECURSIVE JSON CONTENT SEARCH
# ============================================================

def find_content_recursive(obj):
    """
    Recursively search JSON dictionaries/lists for article
    content fields.

    Returns the longest useful candidate.
    """

    candidates = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            key_lower = str(key).lower()

            # Match known content keys case-insensitively.
            if (
                key_lower
                in {
                    x.lower()
                    for x in CONTENT_KEYS
                }
            ):

                if isinstance(
                    value,
                    str,
                ):

                    if len(
                        value.strip()
                    ) >= MIN_TEXT_LENGTH:

                        candidates.append(
                            value
                        )

            nested = find_content_recursive(
                value
            )

            if nested:
                candidates.append(
                    nested
                )

    elif isinstance(obj, list):

        for item in obj:

            nested = find_content_recursive(
                item
            )

            if nested:
                candidates.append(
                    nested
                )

    if not candidates:
        return None

    return max(
        candidates,
        key=len,
    )


# ============================================================
# METHOD 1: NEXT_DATA
# ============================================================

def try_next_data_extraction(html):
    """
    Extract article body from __NEXT_DATA__.

    Supports both:

        data = {...}

    and:

        data = [...]

    and nested variants.
    """

    match = NEXT_DATA_RE.search(
        html
    )

    if not match:
        return None

    try:

        data = json.loads(
            match.group(1)
        )

    except json.JSONDecodeError:
        return None

    # --------------------------------------------------------
    # Direct path
    # --------------------------------------------------------

    try:

        props = data.get(
            "props",
            {},
        )

        page_props = props.get(
            "pageProps",
            {},
        )

        page_data = page_props.get(
            "data"
        )

        # New/old structure where data is a dict.
        if isinstance(
            page_data,
            dict,
        ):

            for key in CONTENT_KEYS:

                value = page_data.get(
                    key
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and len(
                        value.strip()
                    ) >= MIN_TEXT_LENGTH
                ):

                    text = html_to_text(
                        value
                    )

                    if (
                        text
                        and len(text)
                        >= MIN_TEXT_LENGTH
                    ):
                        return text

        # If page_data is a list, don't index it
        # with string keys.
        #
        # The recursive search below handles it.

    except (
        AttributeError,
        TypeError,
    ):
        pass

    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    content = find_content_recursive(
        data
    )

    if content:

        # First assume content is HTML.
        text = html_to_text(
            content
        )

        # If HTML parsing didn't produce useful text,
        # treat it as plain text.
        if not text:
            text = clean_text(
                content
            )

        if (
            text
            and len(text)
            >= MIN_TEXT_LENGTH
        ):
            return text

    return None


# ============================================================
# METHOD 2: JSON-LD
# ============================================================

def try_jsonld_extraction(html):
    """
    Search JSON-LD for articleBody.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    scripts = soup.find_all(
        "script",
        type=re.compile(
            r"application/ld\+json",
            re.IGNORECASE,
        ),
    )

    candidates = []

    def search_jsonld(obj):

        if isinstance(
            obj,
            dict,
        ):

            for key, value in obj.items():

                if str(key).lower() in {
                    "articlebody",
                    "article_body",
                }:

                    if (
                        isinstance(
                            value,
                            str,
                        )
                        and len(
                            value.strip()
                        ) >= MIN_TEXT_LENGTH
                    ):

                        candidates.append(
                            value
                        )

                search_jsonld(
                    value
                )

        elif isinstance(
            obj,
            list,
        ):

            for item in obj:

                search_jsonld(
                    item
                )

    for script in scripts:

        raw = script.string

        if not raw:
            raw = script.get_text()

        if not raw:
            continue

        try:

            data = json.loads(
                raw
            )

        except json.JSONDecodeError:
            continue

        search_jsonld(
            data
        )

    if not candidates:
        return None

    best = max(
        candidates,
        key=len,
    )

    text = clean_text(
        best
    )

    if (
        text
        and len(text)
        >= MIN_TEXT_LENGTH
    ):
        return text

    return None


# ============================================================
# METHOD 3: OLD BUSINESS STANDARD HTML
# ============================================================

def try_old_bs_extraction(html):
    """
    Handles older Business Standard article pages.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
        ]
    ):
        tag.decompose()

    selectors = [
        ".story-content",
        ".story-content p",
        ".article-content",
        ".article-content p",
        ".article-body",
        ".article-body p",
        ".story-body",
        ".story-body p",
        ".content-area",
        ".content-area p",
        ".story-details",
        ".story-details p",
        ".article-detail",
        ".article-detail p",
        ".detail-content",
        ".detail-content p",
        ".article_text",
        ".article_text p",
        ".story_text",
        ".story_text p",
        ".articleBody",
        ".articleBody p",
        ".storyBody",
        ".storyBody p",
    ]

    candidates = []

    for selector in selectors:

        try:

            elements = soup.select(
                selector
            )

        except Exception:
            continue

        for element in elements:

            if element.name == "p":

                text = element.get_text(
                    " ",
                    strip=True,
                )

                if len(text) >= 20:
                    candidates.append(
                        text
                    )

            else:

                paragraphs = (
                    element.find_all(
                        "p"
                    )
                )

                if paragraphs:

                    texts = []

                    for p in paragraphs:

                        text = p.get_text(
                            " ",
                            strip=True,
                        )

                        if len(text) >= 20:
                            texts.append(
                                text
                            )

                    if texts:

                        candidates.append(
                            "\n\n".join(
                                texts
                            )
                        )

                else:

                    text = element.get_text(
                        " ",
                        strip=True,
                    )

                    if (
                        len(text)
                        >= MIN_TEXT_LENGTH
                    ):
                        candidates.append(
                            text
                        )

    if not candidates:
        return None

    best = max(
        candidates,
        key=len,
    )

    best = clean_text(
        best
    )

    if (
        best
        and len(best)
        >= MIN_TEXT_LENGTH
    ):
        return best

    return None


# ============================================================
# METHOD 4: BYLINE -> ARTICLE PARAGRAPHS
# ============================================================

def try_html_paragraph_extraction(html):
    """
    Search for article paragraphs following common
    Business Standard byline/read-time markers.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
        ]
    ):
        tag.decompose()

    start_patterns = [
        r"min read",
        r"\bmins?\s+read\b",
        r"read time",
        r"published on",
        r"updated on",
    ]

    start_tag = None

    for pattern in start_patterns:

        marker = soup.find(
            string=re.compile(
                pattern,
                re.IGNORECASE,
            )
        )

        if marker:

            start_tag = marker.parent
            break

    if not start_tag:
        return None

    paragraphs = []

    for p in start_tag.find_all_next(
        "p"
    ):

        text = p.get_text(
            " ",
            strip=True,
        )

        if not text:
            continue

        # Stop at article footer.
        if re.match(
            r"^(Topics?|Tags?)\s*:",
            text,
            re.IGNORECASE,
        ):
            break

        if re.match(
            r"^(Also Read|Related Stories?|Recommended)",
            text,
            re.IGNORECASE,
        ):
            break

        if len(text) < 20:
            continue

        paragraphs.append(
            text
        )

        if len(paragraphs) >= 300:
            break

    if not paragraphs:
        return None

    text = clean_text(
        "\n\n".join(
            paragraphs
        )
    )

    if (
        text
        and len(text)
        >= MIN_TEXT_LENGTH
    ):
        return text

    return None


# ============================================================
# METHOD 5: GENERIC ARTICLE CONTAINER
# ============================================================

def try_generic_article_extraction(html):
    """
    Generic fallback for newer/unknown page structures.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
            "nav",
            "header",
            "footer",
        ]
    ):
        tag.decompose()

    selectors = [
        "article",
        "main",
        "[class*='article-body']",
        "[class*='articleBody']",
        "[class*='article-content']",
        "[class*='articleContent']",
        "[class*='story-body']",
        "[class*='storyBody']",
        "[class*='story-content']",
        "[class*='storyContent']",
        "[class*='content-body']",
        "[class*='contentBody']",
        "[class*='news-body']",
        "[class*='newsBody']",
        "[class*='detail-content']",
        "[class*='detailContent']",
    ]

    candidates = []

    for selector in selectors:

        try:

            elements = soup.select(
                selector
            )

        except Exception:
            continue

        for element in elements:

            paragraphs = (
                element.find_all(
                    "p"
                )
            )

            if not paragraphs:
                continue

            texts = []

            for p in paragraphs:

                text = p.get_text(
                    " ",
                    strip=True,
                )

                if len(text) >= 20:
                    texts.append(
                        text
                    )

            if texts:

                combined = clean_text(
                    "\n\n".join(
                        texts
                    )
                )

                if combined:
                    candidates.append(
                        combined
                    )

    if not candidates:
        return None

    best = max(
        candidates,
        key=len,
    )

    if len(best) >= MIN_TEXT_LENGTH:
        return best

    return None


# ============================================================
# METHOD 6: PODCAST
# ============================================================

def try_podcast_extraction(html):
    """
    Podcast-specific fallback.
    """

    if "/podcast/" not in html.lower():
        return None

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
        ]
    ):
        tag.decompose()

    selectors = [
        "[class*='transcript']",
        "[class*='Transcript']",
        "[class*='podcast-content']",
        "[class*='podcastContent']",
        "[class*='podcast-body']",
        "[class*='podcastBody']",
        "[class*='episode-content']",
        "[class*='episodeContent']",
    ]

    candidates = []

    for selector in selectors:

        try:

            elements = soup.select(
                selector
            )

        except Exception:
            continue

        for element in elements:

            text = element.get_text(
                "\n\n",
                strip=True,
            )

            text = clean_text(
                text
            )

            if (
                text
                and len(text)
                >= MIN_TEXT_LENGTH
            ):
                candidates.append(
                    text
                )

    if candidates:

        return max(
            candidates,
            key=len,
        )

    return None


# ============================================================
# FETCH ARTICLE
# ============================================================

def fetch_article_text(url):
    """
    Fetch article and run all extraction strategies.
    """

    delay = RETRY_BACKOFF_SECONDS

    resp = None

    # --------------------------------------------------------
    # HTTP RETRIES
    # --------------------------------------------------------

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=30,
            )

            status = resp.status_code

            # ------------------------------------------------
            # 403 / 429
            # ------------------------------------------------

            if status in (
                403,
                429,
            ):

                if attempt < MAX_RETRIES:

                    print(
                        f"    HTTP {status}; "
                        f"retrying in {delay}s..."
                    )

                    time.sleep(
                        delay
                    )

                    delay *= 2
                    continue

                return (
                    None,
                    f"Blocked "
                    f"(HTTP {status}) "
                    f"after retries",
                )

            # ------------------------------------------------
            # Server errors
            # ------------------------------------------------

            if status >= 500:

                if attempt < len(
                    SERVER_ERROR_DELAYS
                ):

                    wait = (
                        SERVER_ERROR_DELAYS[
                            attempt
                        ]
                    )

                    print(
                        f"    HTTP {status}; "
                        f"server error, "
                        f"retrying in {wait}s..."
                    )

                    time.sleep(
                        wait
                    )

                    continue

                return (
                    None,
                    f"Business Standard "
                    f"server error "
                    f"(HTTP {status})",
                )

            # ------------------------------------------------
            # Other HTTP errors
            # ------------------------------------------------

            resp.raise_for_status()

            break

        except requests.exceptions.RequestException as e:

            if attempt < MAX_RETRIES:

                print(
                    f"    request failed: "
                    f"{e}; retrying in "
                    f"{delay}s..."
                )

                time.sleep(
                    delay
                )

                delay *= 2

            else:

                return (
                    None,
                    f"Request failed: {e}",
                )

    if resp is None:

        return (
            None,
            "No response received",
        )

    html = resp.text

    # ========================================================
    # EXTRACTION METHODS
    # ========================================================

    methods = [
        (
            "next_data",
            try_next_data_extraction,
        ),
        (
            "jsonld",
            try_jsonld_extraction,
        ),
        (
            "old_bs_html",
            try_old_bs_extraction,
        ),
        (
            "byline_paragraphs",
            try_html_paragraph_extraction,
        ),
        (
            "generic_article",
            try_generic_article_extraction,
        ),
        (
            "podcast",
            try_podcast_extraction,
        ),
    ]

    for method_name, method in methods:

        try:

            text = method(
                html
            )

        except Exception as e:

            print(
                f"    [debug] "
                f"{method_name} error: {e}"
            )

            continue

        if (
            text
            and len(text)
            >= MIN_TEXT_LENGTH
        ):

            print(
                f"    ✓ {method_name}: "
                f"{len(text):,} chars"
            )

            return (
                text,
                None,
            )

    # ========================================================
    # SAVE DEBUG HTML
    # ========================================================

    print(
        f"    ✗ extraction failed "
        f"(HTML={len(html):,} chars)"
    )

    try:

        DEBUG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_debug = list(
            DEBUG_DIR.glob("*.html")
        )

        if len(existing_debug) < 20:

            # Decode URL for readable filename.
            decoded_url = unquote(
                url
            )

            slug = re.sub(
                r"[^a-zA-Z0-9]+",
                "-",
                decoded_url.rstrip("/")
                .split("/")[-1],
            ).strip("-")[:100]

            if not slug:
                slug = "unknown_article"

            debug_path = (
                DEBUG_DIR
                / f"{slug}.html"
            )

            with open(
                debug_path,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    html
                )

            print(
                f"    [debug] HTML saved: "
                f"{debug_path}"
            )

    except Exception:
        pass

    return (
        None,
        "Could not extract article body "
        "with any extraction method",
    )


# ============================================================
# FAILED LOG
# ============================================================

def load_failed_log():

    if not FAILED_LOG.exists():
        return {}

    try:

        with open(
            FAILED_LOG,
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

    except Exception:
        pass

    return {}


def save_failed_log(
    failed
):

    RAW_TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FAILED_LOG,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            failed,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# PROCESS ALL
# ============================================================

def process_all():

    # --------------------------------------------------------
    # Verify directory
    # --------------------------------------------------------

    if not RAW_TEXT_DIR.exists():

        print(
            "ERROR: Article directory "
            "does not exist:"
        )

        print(
            RAW_TEXT_DIR
        )

        return

    # --------------------------------------------------------
    # Find article JSON files
    # --------------------------------------------------------

    json_files = sorted(
        RAW_TEXT_DIR.glob(
            "*.json"
        )
    )

    json_files = [
        f
        for f in json_files
        if not f.name.startswith("_")
    ]

    total = len(
        json_files
    )

    print(
        f"Found {total} article "
        f"records."
    )

    print(
        f"Directory: "
        f"{RAW_TEXT_DIR}"
    )

    print()

    if not json_files:

        print(
            "No article JSON files found."
        )

        return

    failed = load_failed_log()

    n_done = 0
    n_skipped = 0
    n_failed = 0

    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    for i, fpath in enumerate(
        json_files,
        1,
    ):

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        try:

            with open(
                fpath,
                "r",
                encoding="utf-8",
            ) as f:

                record = json.load(
                    f
                )

        except Exception as e:

            print(
                f"[{i}/{total}] "
                f"JSON read error: "
                f"{fpath.name}: {e}"
            )

            n_failed += 1
            continue

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        url = record.get(
            "url"
        )

        if not url:

            print(
                f"[{i}/{total}] "
                f"Missing URL: "
                f"{fpath.name}"
            )

            n_failed += 1
            continue

        # ----------------------------------------------------
        # Skip already completed records
        # ----------------------------------------------------

        existing_text = record.get(
            "text"
        )

        if (
            isinstance(
                existing_text,
                str,
            )
            and len(
                existing_text.strip()
            ) >= MIN_TEXT_LENGTH
        ):

            n_skipped += 1
            continue

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        title = str(
            record.get(
                "title",
                "",
            )
        )

        print(
            f"[{i}/{total}] "
            f"{record.get('date', '')} | "
            f"{title[:100]}"
        )

        # ----------------------------------------------------
        # Fetch
        # ----------------------------------------------------

        text, error = (
            fetch_article_text(
                url
            )
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if text:

            record["text"] = text

            try:

                with open(
                    fpath,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        record,
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                n_done += 1

                # Remove from failed list if it succeeds.
                failed.pop(
                    url,
                    None,
                )

            except OSError as e:

                n_failed += 1

                failed[
                    url
                ] = (
                    f"Could not save "
                    f"JSON: {e}"
                )

                print(
                    f"    ✗ save error: {e}"
                )

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        else:

            n_failed += 1

            failed[
                url
            ] = (
                error
                or "Unknown extraction failure"
            )

        # ----------------------------------------------------
        # Save progress every 25
        # ----------------------------------------------------

        if i % 25 == 0:

            save_failed_log(
                failed
            )

            print()
            print(
                f"Progress: "
                f"{i}/{total} | "
                f"done={n_done} | "
                f"skipped={n_skipped} | "
                f"failed={n_failed}"
            )
            print()

        # ----------------------------------------------------
        # Polite delay
        # ----------------------------------------------------

        time.sleep(
            POLITE_DELAY
        )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    save_failed_log(
        failed
    )

    print()
    print(
        "========================================"
    )
    print(
        "                 DONE"
    )
    print(
        "========================================"
    )

    print(
        f"Fetched successfully : {n_done}"
    )

    print(
        f"Already had text     : {n_skipped}"
    )

    print(
        f"Failed               : {n_failed}"
    )

    print(
        f"Total records        : {total}"
    )

    print()

    print(
        "Failed log:"
    )

    print(
        FAILED_LOG
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_all()