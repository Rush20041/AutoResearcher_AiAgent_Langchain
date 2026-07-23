# 🔎 AI Research & Critic Agent

A multi-agent research pipeline built with **LangChain**, **Google Gemini**, and **Tavily**. It searches the web for a topic, scrapes real source content, writes a grounded research report, and then critiques its own report against the same sources to catch unsupported claims, bias, and gaps.

```
topic
  |
  ▼
🔍 Search Agent (Tavily API)
  |    → finds relevant pages: title, url, snippet
  ▼
📄 Reader Agent (BeautifulSoup scraper)
  |    → scrapes full page text for each source
  ▼
✍️ Writer Chain (Gemini)
  |    → drafts a structured report, grounded only in scraped sources
  ▼
🧐 Critic Chain (Gemini)
  |    → reviews the report against the same sources for accuracy,
  |      bias, missing angles, and citation quality
  ▼
📋 Final Output (report + critique)
```

---

## Features

- **Real web search** via the Tavily Search API — not just the LLM's own knowledge
- **Real page scraping** via `requests` + `BeautifulSoup`, with graceful fallback to the search snippet if a site blocks scraping
- **Grounded writing** — the writer chain is explicitly instructed not to invent facts, and must cite sources inline (`[1]`, `[2]`, ...)
- **Independent fact-checking** — the critic chain reviews the *same raw sources*, not just the writer's report, so it can catch hallucinations rather than just critiquing style
- **`.env`-based config** — no API keys hardcoded anywhere

---

## Tech Stack

- [LangChain](https://python.langchain.com/) — orchestration (`ChatPromptTemplate`, chains)
- [Google Gemini](https://ai.google.dev/) via `langchain-google-genai` — LLM for writing and critiquing
- [Tavily](https://tavily.com/) — web search API
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — HTML scraping

---

## Notes & Limitations

- Scraping can fail on sites that block bots or require JavaScript rendering; in that case the pipeline falls back to Tavily's short content snippet for that source.
- Free-tier Gemini and Tavily quotas are limited — expect rate limits if you run many topics back to back.
- Google frequently deprecates older Gemini model names. If you hit a `404 NOT_FOUND` error mentioning a model, open `agents.py` → `get_llm()` and swap in a currently-supported model name (check [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for the current free-tier list).
- The critic chain reviews the writer's report against the sources, but this is not a substitute for manual fact-checking of anything you plan to publish or rely on.
