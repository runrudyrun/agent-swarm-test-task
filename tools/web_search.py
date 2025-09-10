"""Web search tool for additional information."""

import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Simple web search tool using Brave Search API or fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
    async def search(self, query: str, num_results: int = 3) -> List[dict]:
        """Perform web search and return results."""
        if not self.api_key:
            logger.warning("No API key provided for web search")
            return []
        
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": num_results,
                "text_decorations": False,
                "text_format": "Raw"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.base_url,
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                
                results = []
                for result in data.get("web", {}).get("results", []):
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "age": result.get("age", "")
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []


def web_search(query: str, num_results: int = 3) -> str:
    """Synchronous wrapper for web search.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results
    """
    import asyncio
    
    async def _search():
        tool = WebSearchTool()
        results = await tool.search(query, num_results)
        
        if not results:
            return "Não encontrei informações adicionais na web."
        
        response = "🔍 **Resultados da Pesquisa**\n\n"
        
        for i, result in enumerate(results, 1):
            response += f"**{i}. {result['title']}**\n"
            response += f"{result['description']}\n"
            response += f"🔗 {result['url']}\n"
            if result.get('age'):
                response += f"📅 {result['age']}\n"
            response += "\n"
        
        return response.strip()
    
    try:
        return asyncio.run(_search())
    except RuntimeError:
        # If there's already an event loop running
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_search())


# Simple fallback search using web scraping
async def simple_web_search(query: str, num_results: int = 3) -> List[dict]:
    """Simple web search using DuckDuckGo scraping (fallback)."""
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "description": result.get("body", "")
                })
            return results
            
    except ImportError:
        logger.warning("duckduckgo-search not available for fallback search")
        return []
    except Exception as e:
        logger.error(f"Fallback web search failed: {e}")
        return []