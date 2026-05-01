"""Fetch Wikipedia article text using the Wikipedia REST API.

Uses only the `requests` library — no wikipedia-api wrapper.
"""

import requests
import time

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WikiRAG/1.0 (educational project)"}


def fetch_article(title: str) -> tuple[str, str]:
    """
    Fetch the plain-text extract and canonical URL for a Wikipedia article.

    Returns:
        (text_content, page_url)

    Raises:
        ValueError if the article is not found or returns empty content.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": True,        # plain text, no HTML
        "exsectionformat": "plain", # no wikitext section markers
        "inprop": "url",
        "redirects": True,          # follow redirects (e.g. "Eiffel Tower" → actual page)
    }

    response = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No pages returned for: {title}")

    page = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title}")

    content = page.get("extract", "").strip()
    if not content:
        raise ValueError(f"Empty content for: {title}")

    url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

    return content, url


def fetch_with_retry(title: str, retries: int = 3, delay: float = 2.0) -> tuple[str, str]:
    """Fetch with simple retry logic for transient network errors."""
    last_error = None
    for attempt in range(retries):
        try:
            return fetch_article(title)
        except requests.RequestException as e:
            last_error = e
            time.sleep(delay * (attempt + 1))
    raise last_error
