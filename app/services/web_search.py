"""
External web search service for KnowledgeMesh.

Uses Tavily only when the agent decides that external
information is required.
"""

from __future__ import annotations

import os
from typing import Any

import logfire
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


_tavily_client: TavilyClient | None = None


def get_tavily_client() -> TavilyClient:
    global _tavily_client

    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

        _tavily_client = TavilyClient(api_key=api_key)

    return _tavily_client


def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> list[dict[str, Any]]:
    """
    Search the public web and normalize results into the
    same structure used by internal retrieval.
    """

    query = query.strip()

    if not query:
        return []

    with logfire.span(
        "🌐 External Web Search",
        query=query,
        max_results=max_results,
    ):
        client = get_tavily_client()

        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )

        results = response.get("results", [])

        normalized: list[dict[str, Any]] = []

        for index, result in enumerate(results, start=1):
            normalized.append(
                {
                    "id": f"web-{index}",
                    "content": (
                        result.get("content") or result.get("raw_content") or ""
                    ),
                    "source": (
                        result.get("title")
                        or result.get("url")
                        or "External web source"
                    ),
                    "source_type": "web",
                    "url": result.get("url"),
                    "score": result.get("score"),
                    "rerank_score": None,
                }
            )

        logfire.info(
            "External web search completed",
            result_count=len(normalized),
        )

        return normalized
