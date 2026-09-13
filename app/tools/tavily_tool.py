from tavily import TavilyClient
from app.config import settings

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    client = _get_client()
    response = client.search(
        query=query, max_results=max_results, search_depth="advanced"
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
