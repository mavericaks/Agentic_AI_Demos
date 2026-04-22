"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 2B  —  RAG-Powered Agent (Semantic Memory)         ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Give the agent knowledge it was NEVER trained on.    ║
║  We ingest a "user_preferences.txt" file into a vector DB   ║
║  so the agent can look up YOUR personal rules before acting.║
║                                                             ║
║  Example: The preferences say "No meetings before 10 AM".   ║
║  If an email requests a 9 AM meeting, the agent will REFUSE ║
║  and suggest an alternative — because it checked the RAG!   ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_2_frameworks/demo_2b_rag_agent.py
"""

import os
import sys

# ── Shared bootstrap (warnings, encoding, sys.path, dns, dotenv) ─
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.bootstrap  # noqa: E402, F401

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent

from utils.tools import fetch_emails, schedule_meeting

# ═════════════════════════════════════════════════════════════
#  STEP 1 : BUILD THE VECTOR STORE (Data Ingestion)
#  This is the RAG pipeline: Load → Chunk → Embed → Store
# ═════════════════════════════════════════════════════════════

PREFS_FILE = os.path.join(os.path.dirname(__file__), "user_preferences.txt")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")

def build_vector_store():
    """
    Ingest user_preferences.txt into a ChromaDB vector store.
    This converts human text into numbers (vectors) so the AI can search it mathematically.
    """
    print("📚 [RAG] Building vector store from user_preferences.txt …")

    # 1. LOAD: Read the raw text file from the hard drive
    loader = TextLoader(PREFS_FILE, encoding="utf-8")
    documents = loader.load()

    # 2. CHUNK: AI models have context limits. We split the document into smaller pieces.
    # chunk_overlap=100 ensures that if a sentence spans two chunks, the context isn't lost.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"   → Split into {len(chunks)} chunks")

    # 3. EMBED + STORE: Convert the text chunks into mathematical vectors using an Embedding Model.
    # Then save those vectors to ChromaDB (a specialized database for vectors).
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"   → Stored in ChromaDB at {CHROMA_DIR}")
    return vectorstore


def load_vector_store():
    """Load an existing ChromaDB store from disk to save time, or build if missing."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return build_vector_store()


# ═════════════════════════════════════════════════════════════
#  STEP 2 : RAG-SPECIFIC TOOL (Retrieval)
# ═════════════════════════════════════════════════════════════

# Global reference to our database so the tool can access it
_vectorstore = None

@tool
def search_user_preferences(query: str) -> str:
    """Search the user's personal preferences and rules.

    Use this tool BEFORE scheduling meetings or drafting replies to check
    if the user has any relevant rules (e.g., time restrictions, priority
    settings, communication style preferences).

    Args:
        query: A natural language question about the user's preferences.
    """
    if _vectorstore is None:
        return "No preference database available."

    # 4. RETRIEVE: When the agent calls this tool, we perform a "similarity search".
    # This finds the top 3 chunks (k=3) in the database that mathematically match the query.
    docs = _vectorstore.similarity_search(query, k=3)

    if not docs:
        return "No relevant preferences found for this query."

    # We format the found documents and return them to the agent as text.
    result = "📋 USER PREFERENCES FOUND:\n"
    for i, doc in enumerate(docs, 1):
        result += f"\n--- Rule Set {i} ---\n"
        result += doc.page_content.strip()
        result += "\n"
    return result


# ═════════════════════════════════════════════════════════════
#  STEP 3 : BUILD AND RUN THE RAG-POWERED AGENT
# ═════════════════════════════════════════════════════════════

def run_rag_agent():
    global _vectorstore

    print("=" * 60)
    print("  SESSION 2B : RAG-Powered Agent (Semantic Memory)")
    print("=" * 60)

    # Initialize the database
    _vectorstore = load_vector_store()

    # Create the LLM using our Router (which handles rate limits gracefully)
    from utils.llm_router import get_routed_llm
    llm = get_routed_llm(role="worker_model")

    # The agent now has 3 tools: email fetching, calendar scheduling, AND preference searching.
    tools = [fetch_emails, search_user_preferences, schedule_meeting]

    # We use a System Message to strictly instruct the agent to use the search tool BEFORE acting.
    system_msg = SystemMessage(content=(
        "You are an intelligent email assistant with access to the user's "
        "personal preferences database. "
        "ALWAYS search the user's preferences BEFORE scheduling any meeting "
        "or drafting any reply. If a requested action violates a preference "
        "(e.g., a meeting before 10 AM), you MUST refuse and explain why, "
        "then suggest an alternative that complies with the rules."
    ))

    # Build the agent
    agent = create_react_agent(llm, tools)

    print("\n🤖 [Agent] Running RAG-powered agent …\n")

    result = agent.invoke({
        "messages": [
            system_msg,
            HumanMessage(content=(
                "Fetch my recent emails. For any meeting requests found, "
                "check my personal preferences first, then schedule the "
                "meetings ONLY if they comply with my rules. "
                "If they don't comply, explain the conflict and suggest "
                "an alternative. Give me a complete summary at the end."
            )),
        ]
    })

    # Print the conversation
    print("─" * 50)
    print("📜 FULL AGENT CONVERSATION:")
    print("─" * 50)
    for msg in result["messages"]:
        role = msg.type.upper()
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") if isinstance(c, dict) and "text" in c else str(c) for c in content)
        if content:
            print(f"\n[{role}]:")
            print(f"  {content}")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Tool Call: {tc['name']}({tc['args']})")

    print()
    print("─" * 50)
    print("🎓 KEY TAKEAWAY:")
    print("   The agent now has SEMANTIC MEMORY — knowledge it was")
    print("   never trained on (your personal rules in user_preferences.txt).")
    print()
    print("   RAG Pipeline:  Load → Chunk → Embed → Store → Retrieve")
    print()
    print("   The agent searched the vector DB BEFORE scheduling,")
    print("   ensuring it respects YOUR rules — not just generic logic.")
    print("=" * 60)


if __name__ == "__main__":
    run_rag_agent()
