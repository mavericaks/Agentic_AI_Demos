"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 2A  —  LangChain Agent                             ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Replace the messy while-loop from Session 1B with    ║
║  LangChain's clean agent framework.                         ║
║                                                             ║
║  Compare:  ~120 lines of manual JSON parsing in 1B          ║
║            → ~40 lines of clean LangChain code here         ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_2_frameworks/demo_2a_langchain_agent.py
"""

import os
import sys

# ── Shared bootstrap (warnings, encoding, sys.path, dns, dotenv) ─
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.bootstrap  # noqa: E402, F401

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# LangGraph's prebuilt ReAct agent — it handles the while-loop for us!
from langgraph.prebuilt import create_react_agent

# ── Shared tools (no more copy-paste!) ───────────────────────
# Under the hood, these functions use LangChain's @tool decorator,
# which automatically extracts their docstrings and parameters to feed to the LLM.
from utils.tools import fetch_emails, schedule_meeting


# ═════════════════════════════════════════════════════════════
#  AGENT SETUP  —  Compare this to the 120-line while-loop!
# ═════════════════════════════════════════════════════════════

def run_langchain_agent():
    print("=" * 60)
    print("  SESSION 2A : LangChain Agent")
    print("=" * 60)

    # 1. Create the LLM using LangChain's unified wrapper.
    # We set temperature=0. This makes the LLM deterministic and precise.
    # We don't want the agent to be "creative" or hallucinate when deciding which tool to call.
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
    )

    # 2. List all tools the agent is allowed to use.
    tools = [fetch_emails, schedule_meeting]

    # 3. Create the agent — ONE LINE replaces the entire while-loop from Demo 1B!
    # Under the hood, this function sets up the System Prompt, creates the JSON parser,
    # and builds the loop that feeds function results back to the LLM.
    agent = create_react_agent(llm, tools)

    # 4. Run the agent using .invoke()
    print("\n🤖 [Agent] Running LangChain ReAct agent …\n")

    result = agent.invoke({
        "messages": [
            HumanMessage(content=(
                "Fetch my recent emails, analyze them, categorize each as "
                "urgent/meeting/task/info, and if you find any meeting "
                "requests, schedule them on my Google Calendar. "
                "Give me a full summary at the end."
            ))
        ]
    })

    # 5. Print the conversation history
    # The result contains the entire memory of the loop. We iterate through it to see what happened.
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

        # Show tool calls if the agent decided to use one
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Tool Call: {tc['name']}({tc['args']})")

    print()
    print("─" * 50)
    print("🎓 KEY TAKEAWAY:")
    print("   We got the SAME result as Session 1B's manual agent,")
    print("   but LangChain handled ALL the complexity:")
    print("   • JSON parsing")
    print("   • Tool routing")
    print("   • Conversation management")
    print("   • Error handling")
    print()
    print("   The @tool decorator + create_react_agent() = Production-ready agent.")
    print("=" * 60)


if __name__ == "__main__":
    run_langchain_agent()
