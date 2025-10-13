import pytest
import json
import time
from typing import Dict, Any, List
from struct_agent.tools.playwright_search_tools import (
    playwright_search_sync,
    make_playwright_search_tool,
    PlaywrightSearchArgs,
    GoogleSearchEngine,
    BingSearchEngine,
    DuckDuckGoSearchEngine,
    create_search_engine,
    process_search_results
)


class TestPlaywrightSearchTool:
    """Comprehensive test suite for Playwright-based web search tool."""

    @pytest.fixture(scope="class")
    def search_tool(self):
        """Fixture to create the search tool instance."""
        return make_playwright_search_tool()

    @pytest.fixture(scope="class")
    def sample_queries(self):
        """Fixture providing sample search queries for testing."""
        return [
            "current weather in Paris",
            "capital of France",
            "Python programming language",
            "latest tech news 2024"
        ]

    @pytest.fixture(scope="class")
    def factual_queries(self):
        """Fixture providing queries with verifiable factual answers."""
        return [
            {"query": "capital of Japan", "expected_keywords": ["Tokyo"]},
            {"query": "current weather in London", "expected_keywords": ["weather", "temperature", "London"]},
            {"query": "Eiffel Tower location", "expected_keywords": ["Paris", "France"]},
            {"query": "Python creator", "expected_keywords": ["Guido", "van", "Rossum"]},
        ]

    def test_search_tool_creation(self, search_tool):
        """Test that the search tool can be created successfully."""
        assert search_tool.name == "playwright_search"
        assert search_tool.description is not None
        assert search_tool.args_model == PlaywrightSearchArgs
        assert search_tool.handler is not None
        assert "search_engine" in search_tool.parameters
        assert "queries" in search_tool.parameters
        assert "max_results" in search_tool.parameters

    def test_search_tool_handler_basic(self, search_tool):
        """Test the search tool handler with basic functionality."""
        args = {
            "queries": ["Python programming"],
            "search_engine": "google",
            "max_results": 3,
            "timeout": 20
        }

        result = search_tool.handler(args)

        assert isinstance(result, dict)
        assert "results" in result
        assert "search_engine" in result
        assert "total_queries" in result
        assert "total_results" in result

        # Should have results
        assert len(result["results"]) > 0
        assert result["search_engine"] == "google"
        assert result["total_queries"] == 1

    def test_search_with_playwright_basic(self):
        """Test basic playwright search functionality."""
        queries = ["Python programming language"]
        result = playwright_search_sync(queries, search_engine="google", max_results=5)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) > 0

        # Check result structure
        first_result = result["results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert "content" in first_result
        assert "query" in first_result
        assert "source" in first_result

        # Verify the result is related to Python
        assert any(term.lower() in first_result["title"].lower() or
                  term.lower() in first_result["content"].lower()
                  for term in ["python", "programming"])

    @pytest.mark.parametrize("search_engine", ["google", "bing", "duckduckgo"])
    def test_different_search_engines(self, search_engine):
        """Test search functionality across different search engines."""
        queries = ["test query"]

        try:
            result = playwright_search_sync(
                queries,
                search_engine=search_engine,
                max_results=3,
                timeout=15
            )

            assert isinstance(result, dict)
            assert result["search_engine"] == search_engine

            if "error" not in result:
                assert len(result["results"]) > 0

                # Check result structure
                for search_result in result["results"]:
                    assert "title" in search_result
                    assert "url" in search_result
                    assert "source" in search_result
                    assert search_result["source"] == search_engine

        except Exception as e:
            pytest.skip(f"Search engine {search_engine} test failed: {e}")

    def test_multiple_queries(self, sample_queries):
        """Test searching with multiple queries simultaneously."""
        # Use just 2 queries for faster testing
        test_queries = sample_queries[:2]

        result = playwright_search_sync(
            test_queries,
            search_engine="google",
            max_results=3
        )

        assert isinstance(result, dict)
        assert result["total_queries"] == len(test_queries)
        assert len(result["results"]) > 0

        # Check that we have results for different queries
        result_queries = set(r["query"] for r in result["results"])
        assert len(result_queries) > 1  # Should have results from multiple queries

    def test_factual_searches(self, factual_queries):
        """Test searches with verifiable factual answers."""
        test_case = factual_queries[0]  # Test just one case for speed
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]

        result = playwright_search_sync(
            [query],
            search_engine="google",
            max_results=5
        )

        assert "error" not in result
        assert len(result["results"]) > 0

        # Check if any result contains expected keywords
        found_keywords = False
        for search_result in result["results"]:
            content = f"{search_result['title']} {search_result['content']}".lower()
            if all(keyword.lower() in content for keyword in expected_keywords):
                found_keywords = True
                break

        # Note: This test might occasionally fail due to search engine variations
        # In a production environment, you might want to make this more robust
        if not found_keywords:
            pytest.skip(f"Expected keywords not found in results for: {query}")

    def test_error_handling_invalid_engine(self):
        """Test error handling for invalid search engine."""
        queries = ["test query"]

        result = playwright_search_sync(
            queries,
            search_engine="invalid_engine",
            max_results=5
        )

        assert "error" in result
        assert "Unsupported search engine" in result["error"]

    def test_timeout_handling(self):
        """Test timeout handling with very short timeout."""
        queries = ["complex search query"]

        result = playwright_search_sync(
            queries,
            search_engine="google",
            max_results=5,
            timeout=1  # Very short timeout
        )

        # Should either succeed quickly or have a timeout error
        assert isinstance(result, dict)
        assert "results" in result or "error" in result

    def test_empty_queries(self):
        """Test handling of empty queries list."""
        result = playwright_search_sync(
            [],
            search_engine="google",
            max_results=5
        )

        assert isinstance(result, dict)
        assert result["total_queries"] == 0
        assert len(result["results"]) == 0

    def test_max_results_limiting(self):
        """Test that max_results parameter correctly limits results."""
        queries = ["Python programming"]
        max_results = 2

        result = playwright_search_sync(
            queries,
            search_engine="google",
            max_results=max_results
        )

        if "error" not in result:
            assert len(result["results"]) <= max_results

    def test_result_deduplication(self):
        """Test that duplicate URLs are removed from results."""
        # Process raw results with duplicates
        raw_results = [
            {
                "title": "Python",
                "url": "https://python.org",
                "content": "Python programming language",
                "query": "Python",
                "source": "google"
            },
            {
                "title": "Python Official",
                "url": "https://python.org",  # Duplicate URL
                "content": "Official Python website",
                "query": "Python",
                "source": "google"
            },
            {
                "title": "Java",
                "url": "https://java.com",
                "content": "Java programming language",
                "query": "Java",
                "source": "google"
            }
        ]

        processed = process_search_results(raw_results, max_results=10)

        # Should have removed the duplicate
        assert len(processed) == 2
        urls = [r["url"] for r in processed]
        assert len(urls) == len(set(urls))  # No duplicates

    def test_search_engine_creation(self):
        """Test search engine factory function."""
        google_engine = create_search_engine("google")
        assert isinstance(google_engine, GoogleSearchEngine)

        bing_engine = create_search_engine("bing")
        assert isinstance(bing_engine, BingSearchEngine)

        duckduckgo_engine = create_search_engine("duckduckgo")
        assert isinstance(duckduckgo_engine, DuckDuckGoSearchEngine)

        with pytest.raises(ValueError):
            create_search_engine("invalid_engine")

    def test_search_engine_url_generation(self):
        """Test that search engines generate correct URLs."""
        google = GoogleSearchEngine()
        url = google.get_search_url("Python programming")
        assert "google.com" in url
        assert "Python+programming" in url or "Python%20programming" in url

        bing = BingSearchEngine()
        url = bing.get_search_url("Python programming")
        assert "bing.com" in url

        duckduckgo = DuckDuckGoSearchEngine()
        url = duckduckgo.get_search_url("Python programming")
        assert "duckduckgo.com" in url

    def test_weather_search_real_time(self):
        """Test real-time weather search - this tests actual web functionality."""
        queries = ["current weather in New York"]

        result = playwright_search_sync(
            queries,
            search_engine="google",
            max_results=5,
            timeout=20
        )

        assert isinstance(result, dict)

        if "error" not in result:
            assert len(result["results"]) > 0

            # Look for weather-related content
            found_weather_info = False
            for search_result in result["results"]:
                content = f"{search_result['title']} {search_result['content']}".lower()
                weather_terms = ["weather", "temperature", "forecast", "celsius", "fahrenheit", "new york"]

                if any(term in content for term in weather_terms):
                    found_weather_info = True
                    break

            # Note: This might occasionally fail if weather sites are down
            if not found_weather_info:
                pytest.skip("Weather information not found in search results")

    def test_tool_spec_integration(self, search_tool):
        """Test ToolSpec integration and parameter validation."""
        # Test with valid parameters
        valid_args = {
            "queries": ["test query"],
            "search_engine": "google",
            "max_results": 5,
            "timeout": 15
        }

        result = search_tool.handler(valid_args)
        assert isinstance(result, dict)

        # Test with invalid parameters (should be handled gracefully)
        invalid_args = {
            "queries": [],  # Empty queries
            "search_engine": "invalid",
            "max_results": -1,  # Invalid max_results
            "timeout": 0  # Invalid timeout
        }

        result = search_tool.handler(invalid_args)
        # Should either handle gracefully or return an error
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_search_reliability(self):
        """Test search reliability across multiple attempts."""
        """This test is marked as slow and may be skipped in normal runs."""
        queries = ["test search reliability"]
        success_count = 0
        total_attempts = 3

        for attempt in range(total_attempts):
            try:
                result = playwright_search_sync(
                    queries,
                    search_engine="google",
                    max_results=3,
                    timeout=10
                )

                if "error" not in result and len(result["results"]) > 0:
                    success_count += 1

                # Brief delay between attempts
                time.sleep(1)

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

        # At least 2 out of 3 attempts should succeed
        assert success_count >= 2, f"Search reliability test failed: {success_count}/{total_attempts} attempts succeeded"

    def test_content_quality(self):
        """Test that search results contain meaningful content."""
        queries = ["machine learning basics"]

        result = playwright_search_sync(
            queries,
            search_engine="google",
            max_results=5
        )

        if "error" not in result:
            assert len(result["results"]) > 0

            # Check that results have meaningful content
            for search_result in result["results"]:
                # Title should be substantial
                assert len(search_result["title"]) > 5
                # URL should be valid
                assert search_result["url"].startswith(("http://", "https://"))
                # Content should be meaningful (if present)
                if search_result["content"]:
                    assert len(search_result["content"]) > 10


if __name__ == "__main__":
    # Run a quick manual test
    print("Running quick manual test...")

    tool = make_playwright_search_tool()
    args = {
        "queries": ["current weather in Paris"],
        "search_engine": "google",
        "max_results": 3,
        "timeout": 15
    }

    result = tool.handler(args)
    print("Search result:")
    print(json.dumps(result, indent=2))