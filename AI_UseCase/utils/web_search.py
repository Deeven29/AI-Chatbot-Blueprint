from duckduckgo_search import DDGS

def search_web(query):
  try:
    results=[]
    with DDGS() as ddgs:
      for r in ddgs.text(query, max_results=3):
        results.append(r["body"])
    return " ".join(results)
  except Exception as e:
    raise f"web search failed: {str(e)}"