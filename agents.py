import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import tavily_search_tool, web_scraper_tool

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Make sure you have a .env file "
    )

def get_llm(temperature: float = 0.4) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )


def run_search_agent(topic: str, max_results: int = 5) -> list:
    """
    Uses the Tavily search tool to find sources relevant to topic
    Returns:
        list[dict] -> [{title, url, content}, ...]
    """
    print(f"\n[SEARCH AGENT] Searching Tavily for: '{topic}' ...")
    results = tavily_search_tool.invoke({"query": topic, "max_results": max_results})
    print(f"[SEARCH AGENT] Found {len(results)} sources.")
    return results


def run_reader_agent(search_results: list) -> list:
    """
    Takes the search_results from run_search_agent and scrapes the full
    text content of each URL using the BeautifulSoup scraper tool.
    If scraping a URL fails, falls back to the short Tavily content
    snippet so the pipeline doesn't lose that source entirely.
    Returns:
        list[dict] -> [{title, url, full_content}, ...]
    """
    print("\n[READER AGENT] Scraping full page content for each source...")
    enriched_sources = []

    for item in search_results:
        url = item.get("url", "")
        title = item.get("title", "No title")
        fallback_snippet = item.get("content", "")

        scraped_text = web_scraper_tool.invoke({"url": url}) if url else ""

        if scraped_text.startswith("[SCRAPE_FAILED]") or scraped_text.startswith("[SCRAPE_EMPTY]"):
            print(f"  - {url} -> scrape failed, using search snippet instead.")
            full_content = fallback_snippet
        else:
            print(f"  - {url} -> scraped {len(scraped_text)} chars.")
            full_content = scraped_text

        enriched_sources.append(
            {
                "title": title,
                "url": url,
                "full_content": full_content,
            }
        )

    return enriched_sources


#sources into a single block of text for the LLM prompts
def format_sources_for_prompt(sources: list) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"Source [{i}]\n"
            f"Title: {s['title']}\n"
            f"URL: {s['url']}\n"
            f"Content: {s['full_content']}\n"
        )
    return "\n---\n".join(blocks)


writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research analyst and technical writer. "
            "Your job is to synthesize multiple raw web sources into a single, "
            "well-organized, factually grounded research report.\n\n"
            "Rules you MUST follow:\n"
            "1. Base your report ONLY on the information given in the sources below. "
            "Do not invent facts, statistics, or events that are not present in the sources.\n"
            "2. If sources disagree with each other, explicitly point out the disagreement "
            "instead of silently picking one side.\n"
            "3. Cite sources inline using their bracket number, e.g. [1], [2], matching the "
            "Source numbering given to you.\n"
            "4. Structure the report with clear sections: \n"
            "   - Title\n"
            "   - Executive Summary (3-5 sentences)\n"
            "   - Key Findings (bullet points)\n"
            "   - Detailed Analysis (multiple paragraphs, organized by sub-topic)\n"
            "   - Conclusion\n"
            "   - Sources (numbered list of title + URL)\n"
            "5. Be objective and neutral in tone. Do not add personal opinions.\n"
            "6. If the provided sources are insufficient to answer some part of the topic, "
            "explicitly say so rather than guessing.",
        ),
        (
            "human",
            "Research topic: {topic}\n\n"
            "Here are the raw sources gathered from the web:\n\n"
            "{sources}\n\n"
            "Write the full research report now, following all the rules above.",
        ),
    ]
)

writer_chain = writer_prompt | get_llm(temperature=0.4) | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a rigorous, skeptical peer reviewer and fact-checking editor. "
            "You will be given a research report along with the raw sources it was "
            "based on. Your job is to critically evaluate the report - you are NOT "
            "writing a new report, you are reviewing the one given to you.\n\n"
            "Evaluate the report on these dimensions:\n"
            "1. Factual accuracy - does every claim in the report actually trace back "
            "to something present in the sources? Flag any claim that looks invented "
            "or unsupported (hallucination check).\n"
            "2. Completeness - are there important angles, counterpoints, or sub-topics "
            "from the sources that the report missed?\n"
            "3. Bias / neutrality - is the report neutral, or does it lean toward one "
            "framing without acknowledging alternatives present in the sources?\n"
            "4. Citation quality - are claims properly attributed to sources? Are there "
            "unsupported/uncited claims?\n"
            "5. Clarity & structure - is the report well organized and easy to follow?\n\n"
            "Output format (use exactly these section headers):\n"
            "- Overall Verdict: (one line - e.g. 'Strong', 'Adequate with gaps', 'Weak / needs revision')\n"
            "- Strengths: (bullet points)\n"
            "- Issues Found: (bullet points, be specific, quote the problematic line if possible)\n"
            "- Missing Angles: (bullet points, based on what's in the sources but not in the report)\n"
            "- Suggested Improvements: (concrete, actionable bullet points)",
        ),
        (
            "human",
            "Research topic: {topic}\n\n"
            "Raw sources used:\n\n"
            "{sources}\n\n"
            "Report to critique:\n\n"
            "{report}\n\n"
            "Provide your critique now, following the exact output format above.",
        ),
    ]
)

critic_chain = critic_prompt | get_llm(temperature=0.2) | StrOutputParser()
