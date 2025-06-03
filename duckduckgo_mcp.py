from fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("duckduckgo-search")


@mcp.tool()
async def search_web(query: str) -> str:
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])
if __name__ == "__main__":
    mcp.run(transport="stdio")
