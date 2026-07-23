import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv() 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY not found. Make sure you have a .env file "
    )
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def tavily_search_tool(query: str, max_results: int = 5) -> list:
    """
      Search the web for a given query using the Tavily Search API
        title:   the page title
        url:     the page URL
        content: a short content snippet/summary from Tavily
    """
    response = tavily_client.search( query=query, max_results=max_results, search_depth="advanced" )
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", "No title"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )
    return results


@tool
def web_scraper_tool(url: str) -> str:
    """
    Scrape the full readable text content of a webpage given its URL,
    using requests + BeautifulSoup

    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"[SCRAPE_FAILED] Could not fetch {url}: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[TRUNCATED]"

    if not text.strip():
        return f"[SCRAPE_EMPTY] No readable text extracted from {url}"

    return text
