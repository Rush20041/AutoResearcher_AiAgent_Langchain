import os
import re

from agents import (
    run_search_agent,
    run_reader_agent,
    format_sources_for_prompt,
    writer_chain,
    critic_chain,
)

def topic_to_filename(topic: str) -> str:
    """
    Converts a topic string into a safe filename
    """
    safe = re.sub(r"[^\w\s-]", "", topic)
    safe = re.sub(r"\s+", "_", safe.strip())
    safe = safe[:100] if len(safe) > 100 else safe
    return f"{safe}.txt"

def save_output_to_file(topic: str, content: str, folder: str = None) -> str:
    if folder is None:
        folder = os.path.dirname(os.path.abspath(__file__))

    filename = topic_to_filename(topic)
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def run_research_pipeline(topic: str, max_results: int = 5) -> dict:
    """
    Runs the full pipeline end-to-end for a given topic.

    Returns a dict with:
        - sources: the enriched source list (title, url, full_content)
        - report: the writer chain's output
        - critique: the critic chain's output
        - final_output: a combined, nicely formatted string of everything
    """

    # Step 1: Search Agent
    search_results = run_search_agent(topic, max_results=max_results)

    if not search_results:
        return {
            "sources": [],
            "report": "No search results found - cannot generate a report.",
            "critique": "N/A",
            "final_output": "No search results found for this topic. Try rephrasing it.",
        }

    # Step 2: Reader Agent
    enriched_sources = run_reader_agent(search_results)
    sources_text = format_sources_for_prompt(enriched_sources)

    # Step 3a: Writer Chain
    print("\n[WRITER CHAIN] Drafting research report...")
    report = writer_chain.invoke({"topic": topic, "sources": sources_text})

    # Step 3b: Critic Chain (critiques the writer's report against the same sources)
    print("[CRITIC CHAIN] Critiquing the draft report...")
    critique = critic_chain.invoke(
        {"topic": topic, "sources": sources_text, "report": report}
    )

    # Step 4: Combine into final output
    final_output = (
        f"{'=' * 40}\n"
        f"RESEARCH REPORT: {topic}\n"
        f"{'=' * 40}\n\n"
        f"{report}\n\n"
        f"{'=' * 40}\n"
        f"CRITIC REVIEW\n"
        f"{'=' * 40}\n\n"
        f"{critique}\n"
    )

    return {
        "sources": enriched_sources,
        "report": report,
        "critique": critique,
        "final_output": final_output,
    }


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()

    if not topic:
        print("Please enter a valid topic.")
    else:
        result = run_research_pipeline(topic)
        print("\n\n" + result["final_output"])

        saved_path = save_output_to_file(topic, result["final_output"])
        print(f"\n[SAVED] Result written to: {saved_path}")
