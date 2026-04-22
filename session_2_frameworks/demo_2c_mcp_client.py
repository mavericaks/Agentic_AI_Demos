"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 2C (Part 2)  —  MCP Client Agent                   ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Show the "USB-C for AI" concept.                     ║
║                                                              ║
║  Instead of writing @tool decorators like in Demo 2A,        ║
║  we connect to an MCP Server and the agent AUTO-DISCOVERS    ║
║  all available tools at runtime.                             ║
║                                                              ║
║  Zero tool code in this file — the tools come from the       ║
║  MCP server automatically!                                   ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_2_frameworks/demo_2c_mcp_client.py
"""

import os
import sys
import asyncio
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

# Set stdout to utf-8 to prevent charmap UnicodeEncodeErrors in Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Make imports work from any sub-folder ────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.dns_patch
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Path to the MCP server script (the external "tool provider")
MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


async def run_mcp_agent():
    print("=" * 60)
    print("  SESSION 2C : MCP Client — Auto-Discovered Tools")
    print("=" * 60)

    # ── STEP 1: Connect to the MCP Server ────────────────────
    # In traditional development, if you want an AI to connect to Slack, Jira, and GitHub,
    # you have to write 3 custom API adapters using LangChain @tools.
    # With Model Context Protocol (MCP), you simply connect to an MCP server over stdio.
    # It acts like a "USB-C" port for AI.
    print("\n🔌 [MCP] Connecting to the Inbox Intelligence MCP Server …")

    # MultiServerMCPClient spins up the server script as a background process
    # and communicates with it using standard input/output streams (stdio).
    client = MultiServerMCPClient(
        {
            "inbox_server": {
                "command": sys.executable,       # Run using the current python interpreter
                "args": [MCP_SERVER_SCRIPT],     # The target server script
                "transport": "stdio",            # Use standard I/O for communication
            }
        }
    )
    
    # ── STEP 2: Auto-discover tools ──────────────────────
    # THIS is the magic of MCP. Notice how there are no @tool functions defined in this file!
    # The client asks the server: "What tools do you have?" and the server sends back the list.
    tools = await client.get_tools()

    print(f"\n🔍 [MCP] Auto-discovered {len(tools)} tools:")
    for t in tools:
        print(f"   • {t.name}: {t.description[:80]}…")

    # ── STEP 3: Create the agent with discovered tools ───
    # We pass the dynamically discovered tools directly into LangGraph.
    from utils.llm_router import get_routed_llm
    llm = get_routed_llm(role="worker_model")
    agent = create_react_agent(llm, tools)

    print("\n🤖 [Agent] Running with MCP-discovered tools …\n")

    # Execute the agent normally. It will use the tools via the MCP server connection!
    result = await agent.ainvoke({
        "messages": [
            HumanMessage(content=(
                "First get my inbox statistics, then fetch my recent "
                "emails and analyze them. If you find any meeting "
                "requests, schedule them on my calendar. "
                "Give me a complete summary at the end."
            ))
        ]
    })

    # ── Print the conversation ───────────────────────────
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
    print("   We wrote ZERO tool definitions in this file!")
    print("   The MCP server exposed tools, and the agent")
    print("   auto-discovered them at runtime via the protocol.")
    print()
    print("   MCP = 'USB-C for AI'")
    print("   • Traditional: Write a custom adapter per tool (days)")
    print("   • MCP:         Connect to a server (minutes)")
    print()
    print("   Imagine connecting to Slack, GitHub, or Database MCP")
    print("   servers — your agent gains new abilities instantly.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
