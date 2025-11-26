#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import json
import asyncio
import argparse
from typing import Optional, List, Dict, Any

from loguru import logger

from config import config  # noqa: F401  (kept for env/config side-effects)

# --------- LOGGING ---------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "[<level>{level}</level>] "
    "<cyan>{name}</cyan>: <level>{message}</level>"
)
logger.remove()
logger.add(lambda m: print(m, end=""), level=LOG_LEVEL, format=LOG_FORMAT, enqueue=False)


# --------- ENV / URL HELPERS ---------
def _inside_container() -> bool:
    return bool(os.getenv("INSIDE_DOCKER") or os.getenv("KUBERNETES_SERVICE_HOST"))

def _resolve_mem0_url(fallback_host: str = "localhost") -> str:
    """
    Choose Mem0 base URL:
      1) MEM0_SERVER_URL if set
      2) If inside a container → http://mem0:8001
      3) Host default → http://localhost:${MEM0_PORT or 8001}
    """
    env_url = os.getenv("MEM0_SERVER_URL")
    if env_url:
        return env_url.rstrip("/")
    if _inside_container():
        return "http://mem0:8001"
    port = os.getenv("MEM0_PORT", "8001")
    return f"http://{fallback_host}:{port}"


# --------- MODELS ---------
class LLMModel(str):
    """Assistant runner model identifiers (OpenRouter-style; BaseAgent will normalize)."""
    GPT_4_1_MINI = "openai/gpt-4.1-mini"
    GPT_4O = "openai/gpt-4o"
    GPT_5 = "openai/gpt-5"
    CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4"
    CLAUDE_OPUS_4_1 = "anthropic/claude-opus-4.1"
    GOOGLE_GEMINI_2_5_FLASH = "google/gemini-2.5-flash-preview-09-2025"

    @classmethod
    def choices(cls) -> List[str]:
        return [
            cls.GPT_5,
            cls.GPT_4O,
            cls.GPT_4_1_MINI,
            cls.CLAUDE_SONNET_4,
            cls.CLAUDE_OPUS_4_1,
            cls.GOOGLE_GEMINI_2_5_FLASH,
        ]


# --------- Langfuse Prompt Mgmt helpers ---------
from utilities.base.base_llm import PromptVars  # dataclass used by AdkToolClient/AssistantAgent

def _parse_prompt_vars(
    prompt_name: Optional[str],
    prompt_label: Optional[str],
    prompt_version: Optional[str],
    prompt_type: Optional[str],
    vars_json: Optional[str],
    vars_file: Optional[str],
) -> Optional[PromptVars]:
    if PromptVars is None:
        if any([prompt_name, prompt_label, prompt_version, prompt_type, vars_json, vars_file]):
            logger.warning("Prompt management requested but PromptVars not available; ignoring prompt options.")
        return None

    if not any([prompt_name, prompt_label, prompt_version]):
        # Nothing to do
        return None

    variables: Dict[str, Any] = {}
    if vars_file:
        try:
            with open(vars_file, "r", encoding="utf-8") as fr:
                variables = json.load(fr) or {}
        except Exception as e:
            logger.warning(f"Failed to load --vars-file: {e}")
    if vars_json:
        try:
            variables.update(json.loads(vars_json) or {})
        except Exception as e:
            logger.warning(f"Failed to parse --vars-json: {e}")

    prompt_type = (prompt_type or "both").lower()
    if prompt_type not in {"system", "user", "both"}:
        logger.warning(f"Unknown --prompt-type '{prompt_type}', defaulting to 'both'.")
        prompt_type = "both"

    return PromptVars(  # type: ignore[call-arg]
        prompt_name=prompt_name,
        prompt_label=prompt_label,
        prompt_version=prompt_version,
        prompt_type=prompt_type,   # how to apply compiled templates
        variables=variables or {},
    )


# ========== INGEST HELPERS ==========
def _build_ingest_pipeline(args):
    """
    Wire the ingest pipeline on demand using your source classes.
    The pipeline itself hits the Mem0 server /sink/upsert endpoint.
    """
    from workshop.assistant.ingest.pipeline import IngestPipeline
    from workshop.assistant.ingest.sources.json_roadmap import JSONRoadmapSource

    PDFSource = None
    if getattr(args, "pdf", None):
        try:
            from workshop.assistant.ingest.sources.pdf_pymupdf import PDFPyMuPDFSource as _PDF
            PDFSource = _PDF
        except Exception as e:
            logger.error(f"PDF ingestion requested but pdf_pymupdf source not available: {e}")
            sys.exit(2)

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

    return IngestPipeline(source=source)


def _selected_sinks(args) -> List[str]:
    """Translate CLI flags to sink names expected by Mem0 server (/sink/upsert)."""
    sinks: List[str] = []
    if getattr(args, "mem0", False):
        sinks.append("mem0")
    if getattr(args, "chroma", False):
        sinks.append("chroma")
    if getattr(args, "neo4j", False):
        sinks.append("neo4j")
    if not sinks:
        logger.warning("No sinks selected; defaulting to mem0 only.")
        sinks = ["mem0"]
    return sinks


# ========== ASSISTANT ==========
async def _assistant_once(
    prompt: str,
    *,
    persona: str = "secretary",
    verbosity: str = "normal",
    model_name: str = LLMModel.GPT_5,
    prompt_vars: Optional[PromptVars] = None,
) -> str:
    """
    One-shot ask via AssistantAgent (centralizes prompts, tools, mem0, RAG).
    Supports Langfuse Prompt Management via `prompt_vars`.
    """
    from workshop.assistant.agent import AssistantAgent
    agent = AssistantAgent(
        model_name=model_name,
        enable_rag=False,  # flip on if a retriever is available
        persona=persona,
        verbosity=verbosity,
    )
    return await agent.ainvoke(user_id="cli", user_message=prompt, prompt_vars=prompt_vars)


async def _assistant_chat(
    persona: str,
    verbosity: str,
    use_mem0_ctx: bool,
    save_mem0: bool,
    top_k: int,
    model_name: str,
    prompt_vars: Optional[PromptVars],
) -> None:
    """
    Multi-turn REPL that can optionally include mem0 context and save QA to mem0.
    If `prompt_vars` is provided, it is applied on every turn.
    """
    from workshop.assistant.agent import AssistantAgent
    agent = AssistantAgent(
        model_name=model_name,
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
                prompt_vars=prompt_vars,
            )
        except Exception as e:
            logger.error(f"Assistant run failed: {e}")
            answer = "Sorry — I had trouble answering that."

        if not (answer or "").strip():
            print("Assistant: [no response — LLM runner failed; fallback may be misconfigured. Check logs.]")
        else:
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
    mem0_url = (os.getenv("MEM0_SERVER_URL") or _resolve_mem0_url()).rstrip("/")
    sinks = _selected_sinks(args)
    pipe.run(sinks=sinks, base_url=mem0_url)  # new pipeline signature
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
    asyncio.run(_assistant_chat("secretary", "normal", use_ctx, save, top_k, LLMModel.GPT_5, None))

def _menu_tools():
    print("\n== Tools ==")
    try:
        from workshop.assistant.agent import AssistantAgent
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
    pi = sub.add_parser("ingest", help="Ingest data into mem0 / chroma / neo4j via Mem0 server")
    src = pi.add_mutually_exclusive_group(required=True)
    src.add_argument("--json-roadmap", help="Path to JSON roadmap")
    src.add_argument("--pdf", help="Path to PDF (PyMuPDF)")
    pi.add_argument("--pages", help="1-based page range like 1-5", default=None)
    pi.add_argument("--mem0", action="store_true", help="Ingest to mem0 sink")
    pi.add_argument("--chroma", action="store_true", help="Ingest to ChromaDB sink")
    pi.add_argument("--neo4j", action="store_true", help="Ingest to Neo4j sink")
    pi.add_argument("--mem0-url", help="Override Mem0 server base URL (default: env MEM0_SERVER_URL or localhost:8001)")

    # ask (single-turn)
    pa = sub.add_parser("ask", help="Ask the assistant once")
    pa.add_argument("--prompt", required=True)
    pa.add_argument("--persona", choices=["secretary", "executive", "tutor"], default="secretary")
    pa.add_argument("--verbosity", choices=["brief", "normal", "thorough"], default="normal")
    pa.add_argument("--model", choices=LLMModel.choices(), default=LLMModel.GPT_5)

    # prompt mgmt (Langfuse)
    pa.add_argument("--prompt-name", help="Managed prompt name (Langfuse)")
    pa.add_argument("--prompt-label", help="Optional prompt label (Langfuse)")
    pa.add_argument("--prompt-version", help="Optional prompt version (Langfuse)")
    pa.add_argument("--prompt-type", choices=["system", "user", "both"], default="both",
                    help="Apply compiled system, user, or both templates")
    pa.add_argument("--vars-json", help="Inline JSON string for prompt variables (merged)")
    pa.add_argument("--vars-file", help="Path to JSON file for prompt variables (merged)")

    # chat (multi-turn REPL)
    pc = sub.add_parser("chat", help="Interactive assistant chat")
    pc.add_argument("--mem0-context", action="store_true", help="Include top-k mem0 memory in the prompt")
    pc.add_argument("--mem0-save", action="store_true", help="Save Q/A to mem0")
    pc.add_argument("--top-k", type=int, default=5)
    pc.add_argument("--persona", choices=["secretary", "executive", "tutor"], default="secretary")
    pc.add_argument("--verbosity", choices=["brief", "normal", "thorough"], default="normal")
    pc.add_argument("--model", choices=LLMModel.choices(), default=LLMModel.GPT_5)

    # prompt mgmt for chat as well (applied every turn)
    pc.add_argument("--prompt-name", help="Managed prompt name (Langfuse)")
    pc.add_argument("--prompt-label", help="Optional prompt label (Langfuse)")
    pc.add_argument("--prompt-version", help="Optional prompt version (Langfuse)")
    pc.add_argument("--prompt-type", choices=["system", "user", "both"], default="both")
    pc.add_argument("--vars-json", help="Inline JSON string for prompt variables (merged)")
    pc.add_argument("--vars-file", help="Path to JSON file for prompt variables (merged)")

    # tools (list)
    sub.add_parser("tools", help="List available tools")

    args = p.parse_args()

    # menu if requested or no subcommand
    if args.menu or args.cmd is None:
        logger.info(f"Mem0 base URL = {(os.getenv('MEM0_SERVER_URL') or _resolve_mem0_url())}")
        interactive_menu()
        return

    if args.cmd == "ingest":
        pipe = _build_ingest_pipeline(args)
        mem0_url = (args.mem0_url or os.getenv("MEM0_SERVER_URL") or _resolve_mem0_url()).rstrip("/")
        logger.info(f"Ingest → Mem0 base URL = {mem0_url}")
        sinks = _selected_sinks(args)
        pipe.run(sinks=sinks, base_url=mem0_url)  # new pipeline signature
        return

    if args.cmd == "ask":
        pv = _parse_prompt_vars(
            getattr(args, "prompt_name", None),
            getattr(args, "prompt_label", None),
            getattr(args, "prompt_version", None),
            getattr(args, "prompt_type", None),
            getattr(args, "vars_json", None),
            getattr(args, "vars_file", None),
        )
        text = asyncio.run(
            _assistant_once(
                prompt=args.prompt,
                persona=args.persona,
                verbosity=args.verbosity,
                model_name=args.model,
                prompt_vars=pv,
            )
        )
        print(text)
        return

    if args.cmd == "chat":
        logger.info(f"Chat → Mem0 base URL = {(os.getenv('MEM0_SERVER_URL') or _resolve_mem0_url())}")
        pv = _parse_prompt_vars(
            getattr(args, "prompt_name", None),
            getattr(args, "prompt_label", None),
            getattr(args, "prompt_version", None),
            getattr(args, "prompt_type", None),
            getattr(args, "vars_json", None),
            getattr(args, "vars_file", None),
        )
        asyncio.run(
            _assistant_chat(
                args.persona,
                args.verbosity,
                args.mem0_context,
                args.mem0_save,
                args.top_k,
                args.model,
                pv,
            )
        )
        return

    if args.cmd == "tools":
        _menu_tools()
        return


if __name__ == "__main__":
    main()
