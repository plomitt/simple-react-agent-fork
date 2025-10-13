from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import re
import time
import os
import json
from urllib.parse import urljoin, urlparse

from struct_agent.instructor_based import ToolSpec

try:
    from playwright.sync_api import sync_playwright, Browser, Page
except ImportError:
    raise ImportError("playwright is required. Install with: pip install playwright && playwright install")

load_dotenv()

class PlaywrightSearchArgs(BaseModel):
    """Inputs for the Playwright search tool."""
    queries: List[str] = Field(..., description="List of search queries to execute.")
    search_engine: str = Field("google", description="Search engine to use (google, bing, duckduckgo)")
    max_results: int = Field(10, description="Maximum number of results to return per query (default: 10).")
    timeout: int = Field(30, description="Timeout in seconds for each search (default: 30).")
    headless: bool = Field(True, description="Run browser in headed mode for debugging (default: True for headless).")

class SearchEngine:
    """Base class for search engines."""

    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url

    def get_search_url(self, query: str) -> str:
        """Get the search URL for a query."""
        raise NotImplementedError

    def extract_results(self, page: Page) -> List[Dict[str, Any]]:
        """Extract search results from the page."""
        raise NotImplementedError

class GoogleSearchEngine(SearchEngine):
    """Google search engine implementation."""

    def __init__(self):
        super().__init__("google", "https://www.google.com")

    def get_search_url(self, query: str) -> str:
        return f"{self.base_url}/search?q={query}"

    def extract_results(self, page: Page) -> List[Dict[str, Any]]:
        """Extract search results from Google SERP."""
        results = []

        # Try multiple selectors for Google results
        selectors = [
            "div.g",  # Standard result div
            "div[data-ved]",  # Alternative selector
            ".g",  # Class-based selector
        ]

        for selector in selectors:
            try:
                result_elements = page.query_selector_all(selector)
                if result_elements:
                    break
            except:
                continue

        for element in result_elements[:10]:  # Limit to first 10 results
            try:
                # Extract title and URL
                title_element = element.query_selector("h3")
                link_element = element.query_selector("a")

                if not title_element or not link_element:
                    continue

                title = title_element.inner_text().strip()
                url = link_element.get_attribute("href")

                if not url or url.startswith("#"):
                    continue

                # Clean up Google redirect URLs
                if url.startswith("/url?"):
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(url[5:])
                    url = parsed.get('q', [None])[0]

                if not url:
                    continue

                # Extract snippet/description
                snippet_selectors = [
                    ".VwiC3b",  # Current Google snippet class
                    ".s",  # Older snippet class
                    "[data-ved] span",  # Alternative
                ]

                snippet = ""
                for snippet_sel in snippet_selectors:
                    snippet_element = element.query_selector(snippet_sel)
                    if snippet_element:
                        snippet = snippet_element.inner_text().strip()
                        break

                results.append({
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "source": self.name
                })

            except Exception as e:
                # Skip problematic elements
                continue

        return results

class BingSearchEngine(SearchEngine):
    """Bing search engine implementation."""

    def __init__(self):
        super().__init__("bing", "https://www.bing.com")

    def get_search_url(self, query: str) -> str:
        return f"{self.base_url}/search?q={query}"

    def extract_results(self, page: Page) -> List[Dict[str, Any]]:
        """Extract search results from Bing SERP."""
        results = []

        # Bing result selectors
        selectors = [
            ".b_algo",  # Main result class
            "li.b_algo",  # Alternative
        ]

        for selector in selectors:
            try:
                result_elements = page.query_selector_all(selector)
                if result_elements:
                    break
            except:
                continue

        for element in result_elements[:10]:
            try:
                # Extract title and URL
                title_element = element.query_selector("h2 a")

                if not title_element:
                    continue

                title = title_element.inner_text().strip()
                url = title_element.get_attribute("href")

                if not url:
                    continue

                # Extract snippet
                snippet_element = element.query_selector(".b_caption p")
                snippet = snippet_element.inner_text().strip() if snippet_element else ""

                results.append({
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "source": self.name
                })

            except Exception:
                continue

        return results

class DuckDuckGoSearchEngine(SearchEngine):
    """DuckDuckGo search engine implementation."""

    def __init__(self):
        super().__init__("duckduckgo", "https://duckduckgo.com")

    def get_search_url(self, query: str) -> str:
        return f"{self.base_url}/?q={query}"

    def extract_results(self, page: Page) -> List[Dict[str, Any]]:
        """Extract search results from DuckDuckGo SERP."""
        results = []

        # DuckDuckGo result selectors
        selectors = [
            ".result",  # Main result class
            ".web-result",  # Alternative
        ]

        for selector in selectors:
            try:
                result_elements = page.query_selector_all(selector)
                if result_elements:
                    break
            except:
                continue

        for element in result_elements[:10]:
            try:
                # Extract title and URL
                title_element = element.query_selector(".result__a")

                if not title_element:
                    continue

                title = title_element.inner_text().strip()
                url = title_element.get_attribute("href")

                if not url:
                    continue

                # Extract snippet
                snippet_element = element.query_selector(".result__snippet")
                snippet = snippet_element.inner_text().strip() if snippet_element else ""

                results.append({
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "source": self.name
                })

            except Exception:
                continue

        return results

def create_search_engine(engine_name: str) -> SearchEngine:
    """Create a search engine instance."""
    engines = {
        "google": GoogleSearchEngine,
        "bing": BingSearchEngine,
        "duckduckgo": DuckDuckGoSearchEngine,
    }

    if engine_name not in engines:
        raise ValueError(f"Unsupported search engine: {engine_name}. Supported: {list(engines.keys())}")

    return engines[engine_name]()

def search_with_playwright(
    queries: List[str],
    search_engine: str = "google",
    max_results: int = 10,
    timeout: int = 30,
    headless: bool = True
) -> Dict[str, Any]:
    """
    Perform web search using Playwright browser automation.

    Args:
        queries: List of search queries to execute
        search_engine: Search engine to use (google, bing, duckduckgo)
        max_results: Maximum number of results per query
        timeout: Timeout in seconds for each search
        headless: Whether to run browser in headless mode

    Returns:
        Dictionary containing search results and metadata
    """
    engine = create_search_engine(search_engine)
    all_results = []

    with sync_playwright() as p:
        # Launch browser with realistic settings
        browser = p.chromium.launch(
            headless=headless,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images',  # Speed up page loads
                # Note: We're NOT disabling JavaScript as modern search engines need it
            ]
        )

        try:
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )

            page = context.new_page()

            for query in queries:
                try:
                    # Navigate to search results page
                    search_url = engine.get_search_url(query)
                    page.goto(search_url, timeout=timeout * 1000)

                    # Wait for page to load - wait for a specific selector
                    try:
                        # Wait for search results to appear (engine-specific)
                        if search_engine == "google":
                            page.wait_for_selector("div.g", timeout=5000)
                        elif search_engine == "bing":
                            page.wait_for_selector(".b_algo", timeout=5000)
                        elif search_engine == "duckduckgo":
                            page.wait_for_selector(".result", timeout=5000)
                    except:
                        # If selectors don't appear, just continue
                        time.sleep(2)

                    # Extract results
                    results = engine.extract_results(page)

                    # Add query to each result
                    for result in results:
                        result["query"] = query

                    all_results.extend(results)

                    # Brief delay between searches to be respectful
                    time.sleep(0.5)

                except Exception as e:
                    # Add error result for this query
                    all_results.append({
                        "query": query,
                        "title": "Search Failed",
                        "url": "",
                        "content": f"Failed to search for '{query}': {str(e)}",
                        "source": engine.name,
                        "error": True
                    })

        finally:
            browser.close()

    # Process and rank results
    processed_results = process_search_results(all_results, max_results)

    return {
        "results": processed_results,
        "search_engine": search_engine,
        "total_queries": len(queries),
        "total_results": len(processed_results)
    }

def process_search_results(
    results: List[Dict[str, Any]],
    max_results: int
) -> List[Dict[str, Any]]:
    """
    Process and deduplicate search results.

    Args:
        results: Raw search results
        max_results: Maximum number of results to return

    Returns:
        Processed and deduplicated results
    """
    # Remove error entries for final processing
    successful_results = [r for r in results if not r.get("error")]
    error_results = [r for r in results if r.get("error")]

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []

    for result in successful_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Sort by position (keeping original order)
    # In a more sophisticated implementation, we could rank by relevance

    # Combine successful results with errors at the end
    final_results = unique_results[:max_results] + error_results

    return final_results[:max_results]

def playwright_search_sync(
    queries: List[str],
    search_engine: str = "google",
    max_results: int = 10,
    timeout: int = 30,
    headless: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for Playwright search.

    Args:
        queries: List of search queries
        search_engine: Search engine to use
        max_results: Maximum results per query
        timeout: Timeout in seconds
        headless: Whether to run browser in headless mode

    Returns:
        Search results dictionary
    """
    try:
        return search_with_playwright(queries, search_engine, max_results, timeout, headless)
    except Exception as e:
        return {
            "results": [],
            "search_engine": search_engine,
            "total_queries": len(queries),
            "total_results": 0,
            "error": f"Playwright search failed: {str(e)}"
        }

def make_playwright_search_tool() -> ToolSpec:
    """Creates a ToolSpec for Playwright-based web search."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = PlaywrightSearchArgs(**args)
            results = playwright_search_sync(
                queries=parsed_args.queries,
                search_engine=parsed_args.search_engine,
                max_results=parsed_args.max_results,
                timeout=parsed_args.timeout,
                headless=parsed_args.headless
            )
            return results
        except Exception as e:
            return {"error": f"playwright_search failed: {e}"}

    return ToolSpec(
        name="playwright_search",
        description="Performs reliable web search using browser automation with Playwright. Supports multiple search engines and provides real-time search results without relying on third-party APIs.",
        args_model=PlaywrightSearchArgs,
        handler=handler,
        parameters={
            "queries": "list of search queries to execute",
            "search_engine": "search engine to use (google, bing, duckduckgo)",
            "max_results": "maximum number of results to return per query (default: 10)",
            "timeout": "timeout in seconds for each search (default: 30)",
            "headless": "whether to run browser in headed mode for debugging (default: true)",
        },
    )

__all__ = [
    "make_playwright_search_tool",
    "playwright_search_sync",
    "search_with_playwright",
    "PlaywrightSearchArgs"
]

if __name__ == "__main__":
    # Example usage
    # For debugging, set headless=False to see the browser window
    results = playwright_search_sync(
        queries=["current weather in Paris", "capital of France"],
        search_engine="google",
        max_results=5,
        headless=False  # Set to True for production use
    )

    print(json.dumps(results, indent=2))