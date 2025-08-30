# %% [markdown]
# MamaBot — Jupyter Notebook
#
# Purpose: A reproducible Jupyter notebook skeleton for an LLM-powered helper for new parents.
# It demonstrates how to: 
#  - pull information from reliable public sources (CDC, AAP, WHO, Mayo Clinic, NHS)
#  - sanitize & summarize content
#  - answer user questions with traceable source citations
#  - display a prominent warning banner for medical/legal safety
#
# IMPORTANT: This notebook is a template. You must supply API keys (e.g., OpenAI) and
# review compliance before using for clinical or safety-critical advice.

# %% [markdown]
# Setup
# 1) Install dependencies (run in a notebook cell)
#  - requests, beautifulsoup4, openai (or any LLM client), tiktoken (optional), rich (for pretty banner)
#
# Example:
# !pip install requests beautifulsoup4 openai rich

# %%
# Imports
import os
import re
import json
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from openai import OpenAI

# # Optional: LLM client (OpenAI example)
# try:     
    
#     client = OpenAI(api_key=key)
# except Exception:
#     openai = None

# For pretty terminal/banner output in notebook
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except Exception:
    Console = None
    Panel = None
    Text = None

RELIABLE_SOURCES = {
    "CDC": "https://www.cdc.gov/",
    "AAP (HealthyChildren.org)": "https://www.healthychildren.org/",
    "WHO": "https://www.who.int/"
}


HEADERS = {
    "User-Agent": "MamaBot/1.0 (+https://example.org/mamabot - contact: you@example.org)"
}


def fetch_page_text(url: str, timeout: int = 10) -> Tuple[str, str]:
    """Fetches a URL and returns (title, visible_text).
    Keeps the returned text short (truncates) to avoid huge payloads.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # get title
        title = soup.title.string.strip() if soup.title else url
        # remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        texts = soup.stripped_strings
        text = "\n".join(texts)
        # basic truncation (first 3000 chars)
        if len(text) > 3000:
            text = text[:3000] + "\n...[truncated]"
        return title, text
    except Exception as e:
        return url, f"ERROR_FETCHING: {e}"

def gather_sources(source_dict: Dict[str, str]) -> List[Dict[str, str]]:
    results = []
    for name, base in source_dict.items():
        # try fetch homepage summary
        title, text = fetch_page_text(base)
        results.append({"source_name": name, "url": base, "title": title, "text": text})
    return results

if __name__ == "__main__":
    print("Gathering a quick snapshot of configured reliable sources...\n")
    snapshot = gather_sources(RELIABLE_SOURCES)
    for r in snapshot:
        print(f"- {r['source_name']}: {r['title']} (len={len(r['text'])})")


def format_sources_for_answer(sources: List[Dict[str, str]], max_items: int = 3) -> str:
    lines = []
    for s in sources[:max_items]:
        lines.append(f"- {s['source_name']}: {s['url']}")
    return "\n".join(lines)


def print_warning_banner():
    banner_text = (
        "WARNING: MamaBot provides general informational support for new parents. "
        "It is NOT a substitute for professional medical, legal, or emergency advice. "
        "If you suspect an emergency, call your local emergency number or seek immediate medical care."
    )
    if Console and Panel and Text:
        console = Console()
        t = Text(banner_text)
        console.print(Panel(t, title="IMPORTANT", style="bold red"))
    else:
        # Fallback plain print with strong separators
        sep = "!" * 80
        print(sep)
        print(banner_text)
        print(sep)

# Print banner immediately when running the notebook interactively
if __name__ == "__main__":
    print_warning_banner()


def build_prompt(question: str, sources: List[Dict[str, str]]) -> str:
    # Use short excerpts from top N sources
    excerpts = []
    for s in sources[:3]:
        # keep a short excerpt from the fetched text
        excerpt = s.get("text", "")[:800]
        excerpts.append(f"Source: {s['source_name']} ({s['url']})\n{excerpt}\n")
    prompt = (
        "You are MamaBot, an assistant for new parents. Use the provided sources to answer the question. "
        "Be concise (3-6 sentences), mention if the evidence is uncertain, and list 2-3 short citations.\n\n"
        "SOURCES:\n"
        + "\n---\n".join(excerpts)
        + "\n\nQUESTION: \n" + question
        + "\n\nAnswer:" 
    )
    return prompt


def ask_llm_openai(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 300) -> str:
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    if not key:
        return "OPENAI_API_KEY not set in environment. Set it before using the LLM wrapper."
    try:
        resp = client.chat.completions.create(model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0)
        # this access path depends on client version
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"LLM_ERROR: {e}"


if __name__ == "__main__":
    print_warning_banner()
    q = "My 3-week-old baby is spitting up a lot — when is this normal and when should I worry?"
    print("Question:\n", q)
    sources = gather_sources(RELIABLE_SOURCES)
    prompt = build_prompt(q, sources)
    print("\nConstructed prompt preview (truncated):\n", prompt[:1000], "\n...\n")
    answer = ask_llm_openai(prompt)
    print("\nMamaBot answer:\n", answer)
    print("\nCitations:\n", format_sources_for_answer(sources))