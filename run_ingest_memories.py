#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import asyncio
import argparse
from typing import Dict, Any, Optional
from loguru import logger

# --------- LOGGING ---------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "[<level>{level}</level>] "
    "<cyan>{name}</cyan>: <level>{message}</level>"
)
logger.remove()
logger.add(lambda m: print(m, end=""), level=LOG_LEVEL, format=LOG_FORMAT, enqueue=False)


# ========== OPTIONAL MEM0 ==========
def _load_mem0_client():
    try:
        from mem0 import Memory  # type: ignore
        return Memory
    except Exception as e:
        logger.warning(f"mem0 not available: {e}")
        return None


# ========== INGEST HELPERS ==========
def _build_ingest_pipeline(args):
    from agents.assistant.ingest.pipeline import IngestPipeline
    from agents.assistant.ingest.sources.json_roadmap import JSONRoadmapSource

    PDFSource = None
    if args.pdf:
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


# ========== TOOL LOADING ==========
def _load_tools() -> Dict[str, Any]:
    tools: Dict[str, Any] = {}
    # builtin
    try:
        from agents.assistant.tools.builtin_tools import builtin_tools
        tools.update(builtin_tools())
    except Exception as e:
        logger.warning(f"builtin tools not available: {e}")

    # external (optional)
    try:
        from agents.assistant.tools.external_loader import load_external_tools  # if you created it
        tools.update(load_external_tools())
    except Exception:
        try:
            from agents.assistant.tools.external_loader import load_external_tools
            tools.update(load_external_tools())
        except Exception as e:
            logger.info(f"No external tools loaded: {e}")

    logger.info(f"Loaded {len(tools)} tool(s).")
    return tools


# ========== ADK ASSISTANT ==========
async def _assistant_once(prompt: str, system: Optional[str]) -> str:
    from agents.assistant.adk_client import ADKClient
    client = ADKClient(model_name="google/gemini-2.5-flash-preview-09-2025")  # ADK-only per your design
    for name, fn in _load_tools().items():
        client.register_tool(name, fn)
    return await client.run(prompt=prompt, system_prompt=system or "You are a helpful assistant.")


async def _assistant_chat(system: Optional[str], use_mem0_ctx: bool, save_mem0: bool, top_k: int) -> None:
    from agents.assistant.adk_client import ADKClient
    Memory = _load_mem0_client()
    mem = Memory() if (Memory and (use_mem0_ctx or save_mem0)) else None

    client = ADKClient()
    for name, fn in _load_tools().items():
        client.register_tool(name, fn)

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

        sys_msg = system or "You are a concise, helpful assistant."
        if mem and use_mem0_ctx:
            try:
                hits = mem.search(query=user, top_k=top_k) or []
                memory_lines = []
                for h in hits:
                    txt = h.get("memory") or h.get("text") or ""
                    if txt:
                        memory_lines.append(f"- {txt}")
                if memory_lines:
                    sys_msg += "\n\nRelevant user memory:\n" + "\n".join(memory_lines)
            except Exception as e:
                logger.warning(f"mem0 search failed: {e}")

        try:
            answer = await client.run(prompt=user, system_prompt=sys_msg)
        except Exception as e:
            logger.error(f"Assistant run failed: {e}")
            answer = "Sorry — I had trouble answering that."

        print(f"Assistant: {answer}")

        if mem and save_mem0:
            try:
                mem.add(f"Q: {user}\nA: {answer}")
            except Exception as e:
                logger.warning(f"mem0 save failed: {e}")


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
    system = input("System (optional): ").strip() or None
    txt = asyncio.run(_assistant_once(prompt=prompt, system=system))
    print(f"\nAssistant:\n{txt}\n")

def _menu_chat():
    print("\n== Chat (multi-turn) ==")
    system = input("System (optional): ").strip() or None
    use_ctx = _yes_no("Use mem0 context?", default_no=True)
    save = _yes_no("Save Q/A to mem0?", default_no=True)
    try:
        top_k = int(input("mem0 top_k [5]: ").strip() or "5")
    except ValueError:
        top_k = 5
    asyncio.run(_assistant_chat(system, use_ctx, save, top_k))

def _menu_tools():
    print("\n== Tools ==")
    tools = _load_tools()
    if not tools:
        print("No tools found.\n")
        return
    print("Available tools:")
    for n in sorted(tools.keys()):
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
    pa.add_argument("--system", default=None)

    # chat (multi-turn REPL)
    pc = sub.add_parser("chat", help="Interactive assistant chat")
    pc.add_argument("--system", default=None)
    pc.add_argument("--mem0-context", action="store_true", help="Include top-k mem0 memory in system prompt")
    pc.add_argument("--mem0-save", action="store_true", help="Save Q/A to mem0")
    pc.add_argument("--top-k", type=int, default=5)

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
        text = asyncio.run(_assistant_once(prompt=args.prompt, system=args.system))
        print(text)
        return

    if args.cmd == "chat":
        asyncio.run(_assistant_chat(args.system, args.mem0_context, args.mem0_save, args.top_k))
        return

    if args.cmd == "tools":
        _menu_tools()
        return


if __name__ == "__main__":
    main()
