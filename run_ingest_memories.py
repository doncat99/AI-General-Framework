#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import asyncio
import argparse
from typing import Optional

from loguru import logger

# import warnings
# # ---- optional: squelch known deprecation spam in 3.13 stacks ----
# warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^google\.adk\.runners$")
# warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^httpx\._models$")
# warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^aiohttp\.connector$")

# --------- LOGGING ---------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "[<level>{level}</level>] "
    "<cyan>{name}</cyan>: <level>{message}</level>"
)
logger.remove()
logger.add(lambda m: print(m, end=""), level=LOG_LEVEL, format=LOG_FORMAT, enqueue=False)


# ========== INGEST HELPERS ==========
def _build_ingest_pipeline(args):
    """
    Wire the ingest pipeline on demand. Requires your assistant ingest modules.
    """
    from agents.assistant.ingest.pipeline import IngestPipeline
    from agents.assistant.ingest.sources.json_roadmap import JSONRoadmapSource

    PDFSource = None
    if getattr(args, "pdf", None):
        try:
            from agents.assistant.ingest.sources.pdf_pymupdf import PDFPyMuPDFSource as _PDF
            PDFSource = _PDF
        except Exception as e:
            logger.error(f"PDF ingestion requested but pdf_pymupdf source not available: {e}")
            sys.exit(2)

    from agents.assistant.ingest.sinks.mem0_sink import Mem0Sink
    from agents.assistant.ingest.sinks.chroma_sink import ChromaSink
    from agents.assistant.ingest.sinks.neo4j_sink import Neo4jSink

    # sinks
    sinks = []
    if getattr(args, "mem0", False):
        sinks.append(Mem0Sink())
    if getattr(args, "chroma", False):
        sinks.append(ChromaSink())
    if getattr(args, "neo4j", False):
        sinks.append(Neo4jSink())
    if not sinks:
        logger.warning("No sinks selected; defaulting to mem0 only.")
        sinks = [Mem0Sink()]

    # source
    if getattr(args, "json_roadmap", None):
        source = JSONRoadmapSource(args.json_roadmap)
    elif getattr(args, "pdf", None):
        rng = None
        if getattr(args, "pages", None):
            try:
                a, b = args.pages.split("-", 1)
                rng = range(int(a), int(b) + 1)
            except Exception:
                logger.warning("Invalid --pages format (expected A-B); ignoring.")
        source = PDFSource(args.pdf, pages=rng)  # type: ignore[arg-type]
    else:
        logger.error("You must provide either --json-roadmap or --pdf")
        sys.exit(2)

    return IngestPipeline(source, sinks)


# ========== ASSISTANT ==========
async def _assistant_once(prompt: str, *, persona: str = "secretary", verbosity: str = "normal") -> str:
    """
    One-shot ask via AssistantAgent (centralizes prompts, tools, mem0, RAG).
    """
    from agents.assistant.agent import AssistantAgent
    # model = os.getenv("GOOGLE_MODEL", "google/gemini-2.5-flash-preview-09-2025")
    model = "openai/gpt-4.1-mini"
    agent = AssistantAgent(
        model_name=model,
        enable_rag=False,  # flip on if a retriever is available
        persona=persona,
        verbosity=verbosity,
    )
    return await agent.ainvoke(user_id="cli", user_message=prompt)


async def _assistant_chat(
    persona: str,
    verbosity: str,
    use_mem0_ctx: bool,
    save_mem0: bool,
    top_k: int,
) -> None:
    """
    Multi-turn REPL that can optionally include mem0 context and save QA to mem0.
    """
    from agents.assistant.agent import AssistantAgent
    # model = os.getenv("GOOGLE_MODEL", "google/gemini-2.5-flash-preview-09-2025")
    model = "openai/gpt-4.1-mini"
    agent = AssistantAgent(
        model_name=model,
        enable_rag=False,
        persona=persona,
        verbosity=verbosity,
    )

    print("chat mode. type ':q' or ':quit' to exit.")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if user in {":q", ":quit", ":exit"}:
            print("bye.")
            return

        try:
            answer = await agent.ainvoke(
                user_id="cli",
                user_message=user,
                use_mem0_ctx=use_mem0_ctx,
                save_mem0=save_mem0,
                mem_k=top_k,
            )
        except Exception as e:
            logger.error(f"Assistant run failed: {e}")
            answer = "Sorry — I had trouble answering that."

        print(f"Assistant: {answer}")


# ========== MENU (INTERACTIVE) ==========
def _input_nonempty(prompt: str) -> str:
    while True:
        s = input(prompt).strip()
        if s:
            return s
        print("Please enter a value.")

def _yes_no(prompt: str, default_no: bool = True) -> bool:
    s = input(f"{prompt} [{'y' if not default_no else 'Y'}/{'N' if not default_no else 'n'}]: ").strip().lower()
    if not s:
        return not default_no
    return s in {"y", "yes"}

def _menu_ingest():
    class Args:  # lightweight container compatible with _build_ingest_pipeline
        json_roadmap: Optional[str] = None
        pdf: Optional[str] = None
        pages: Optional[str] = None
        mem0: bool = False
        chroma: bool = False
        neo4j: bool = False

    args = Args()

    print("\n== Ingest ==")
    print("  1) JSON roadmap")
    print("  2) PDF (PyMuPDF)")
    print("  0) Back")
    choice = input("Select source: ").strip()

    if choice == "0":
        return
    elif choice == "1":
        args.json_roadmap = _input_nonempty("Path to JSON roadmap: ")
    elif choice == "2":
        args.pdf = _input_nonempty("Path to PDF: ")
        use_pages = _yes_no("Limit to a page range? e.g., 1-5", default_no=True)
        if use_pages:
            args.pages = _input_nonempty("Enter page range (A-B): ")
    else:
        print("Invalid choice.")
        return

    print("\nSelect sinks (enter numbers separated by comma):")
    print("  1) mem0")
    print("  2) chroma")
    print("  3) neo4j")
    sel = input("Your selection [default: 1]: ").strip() or "1"
    chosen = {s.strip() for s in sel.split(",")}
    args.mem0 = "1" in chosen
    args.chroma = "2" in chosen
    args.neo4j = "3" in chosen

    pipe = _build_ingest_pipeline(args)
    pipe.run()
    print("Ingest finished.\n")

def _menu_ask():
    print("\n== Ask (single turn) ==")
    prompt = _input_nonempty("Prompt: ")
    # Menu uses defaults (secretary/normal)
    txt = asyncio.run(_assistant_once(prompt=prompt))
    print(f"\nAssistant:\n{txt}\n")

def _menu_chat():
    print("\n== Chat (multi-turn) ==")
    use_ctx = _yes_no("Use mem0 context?", default_no=True)
    save = _yes_no("Save Q/A to mem0?", default_no=True)
    try:
        top_k = int(input("mem0 top_k [5]: ").strip() or "5")
    except ValueError:
        top_k = 5
    # Menu uses defaults (secretary/normal)
    asyncio.run(_assistant_chat("secretary", "normal", use_ctx, save, top_k))

def _menu_tools():
    print("\n== Tools ==")
    try:
        from agents.assistant.agent import AssistantAgent
        agent = AssistantAgent(enable_rag=False)
        names = agent.list_tools()
    except Exception as e:
        logger.warning(f"Failed to list tools: {e}")
        names = []

    if not names:
        print("No tools found.\n")
        return
    print("Available tools:")
    for n in names:
        print(f" - {n}")
    print("")


def interactive_menu():
    while True:
        print("\n========== Assistant ==========")
        print("  1) Ingest")
        print("  2) Ask (single turn)")
        print("  3) Chat (multi-turn)")
        print("  4) List tools")
        print("  0) Exit")
        sel = input("Choose an option: ").strip()
        if sel == "1":
            _menu_ingest()
        elif sel == "2":
            _menu_ask()
        elif sel == "3":
            _menu_chat()
        elif sel == "4":
            _menu_tools()
        elif sel == "0":
            print("bye.")
            return
        else:
            print("Invalid choice.")


# ========== CLI ==========
def main():
    p = argparse.ArgumentParser(prog="assistant", description="Project runner: ingest + chat assistant")
    p.add_argument("--menu", action="store_true", help="Launch interactive menu")

    sub = p.add_subparsers(dest="cmd")

    # ingest
    pi = sub.add_parser("ingest", help="Ingest data into mem0 / chroma / neo4j")
    src = pi.add_mutually_exclusive_group(required=True)
    src.add_argument("--json-roadmap", help="Path to JSON roadmap")
    src.add_argument("--pdf", help="Path to PDF (PyMuPDF)")
    pi.add_argument("--pages", help="1-based page range like 1-5", default=None)
    pi.add_argument("--mem0", action="store_true", help="Ingest to mem0")
    pi.add_argument("--chroma", action="store_true", help="Ingest to ChromaDB")
    pi.add_argument("--neo4j", action="store_true", help="Ingest to Neo4j")

    # ask (single-turn)
    pa = sub.add_parser("ask", help="Ask the assistant once")
    pa.add_argument("--prompt", required=True)
    pa.add_argument("--persona", choices=["secretary", "executive", "tutor"], default="secretary")
    pa.add_argument("--verbosity", choices=["brief", "normal", "thorough"], default="normal")

    # chat (multi-turn REPL)
    pc = sub.add_parser("chat", help="Interactive assistant chat")
    pc.add_argument("--mem0-context", action="store_true", help="Include top-k mem0 memory in the prompt")
    pc.add_argument("--mem0-save", action="store_true", help="Save Q/A to mem0")
    pc.add_argument("--top-k", type=int, default=5)
    pc.add_argument("--persona", choices=["secretary", "executive", "tutor"], default="secretary")
    pc.add_argument("--verbosity", choices=["brief", "normal", "thorough"], default="normal")

    # tools (list)
    sub.add_parser("tools", help="List available tools")

    args = p.parse_args()

    # menu if requested or no subcommand
    if args.menu or args.cmd is None:
        interactive_menu()
        return

    if args.cmd == "ingest":
        pipe = _build_ingest_pipeline(args)
        pipe.run()
        return

    if args.cmd == "ask":
        text = asyncio.run(_assistant_once(prompt=args.prompt, persona=args.persona, verbosity=args.verbosity))
        print(text)
        return

    if args.cmd == "chat":
        asyncio.run(_assistant_chat(args.persona, args.verbosity, args.mem0_context, args.mem0_save, args.top_k))
        return

    if args.cmd == "tools":
        _menu_tools()
        return


if __name__ == "__main__":
    main()
