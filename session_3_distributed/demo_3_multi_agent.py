"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 3  —  Multi-Agent System (LangGraph)               ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Break the single monolithic agent into a TEAM of     ║
║  specialised agents that route work to each other.          ║
║                                                             ║
║  The Graph:                                                 ║
║                                                             ║
║    ┌──────────┐    meeting    ┌────────────┐                ║
║    │  TRIAGE  │──────────────▶│ SCHEDULER  │                ║
║    │  AGENT   │               └────────────┘                ║
║    │          │    task       ┌────────────┐  ┌──────────┐  ║
║    │          │──────────────▶│  DRAFTER   │─▶│  HUMAN   │  ║
║    │          │               │   AGENT    │  │  REVIEW  │  ║
║    └──────────┘               └────────────┘  └──────────┘  ║
║         │ info                                              ║
║         ▼                                                   ║
║    ┌──────────┐                                             ║
║    │   END    │                                             ║
║    └──────────┘                                             ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_3_distributed/demo_3_multi_agent.py
"""

import os
import sys
from typing import Literal

# ── Shared bootstrap (warnings, encoding, sys.path, dns, dotenv) ─
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.bootstrap  # noqa: E402, F401

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LangGraph is the orchestration library that lets us build the flow chart above
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from utils.tools import fetch_emails, schedule_meeting, draft_email_reply


# ═════════════════════════════════════════════════════════════
#  SHARED LLM (WITH FALLBACK ROUTER)
# ═════════════════════════════════════════════════════════════
from utils.llm_router import get_routed_llm
llm = get_routed_llm(role="master_model")


# ═════════════════════════════════════════════════════════════
#  SPECIALISED SUB-AGENTS (The Team)
# ═════════════════════════════════════════════════════════════
# Unlike Session 2 where ONE agent had all tools, we now create 3 separate
# agents. Each agent only has the tools and instructions it absolutely needs.
# This makes them much less likely to hallucinate or get confused.

triage_agent = create_react_agent(
    llm,
    tools=[fetch_emails], # Triage can ONLY fetch emails
    prompt=SystemMessage(content=(
        "You are the TRIAGE AGENT. Your ONLY job is to:\n"
        "1. Fetch the user's recent emails.\n"
        "2. Categorize EACH email as exactly one of: 'meeting', 'task', or 'info'.\n"
        "3. After analyzing, respond with a JSON summary like:\n"
        '   {"emails": [{"subject": "...", "category": "meeting|task|info", '
        '"from": "...", "snippet": "...", "time_mentioned": "..." }]}\n'
        "Be precise. Include any time mentioned in meeting emails."
    )),
)

scheduler_agent = create_react_agent(
    llm,
    tools=[schedule_meeting], # Scheduler can ONLY schedule
    prompt=SystemMessage(content=(
        "You are the SCHEDULER AGENT. You receive meeting-related emails.\n"
        "Your job is to extract the meeting time and attendees, then\n"
        "call the schedule_meeting tool to create the calendar event.\n"
        "If no clear time is found, suggest a reasonable time."
    )),
)

drafter_agent = create_react_agent(
    llm,
    tools=[draft_email_reply], # Drafter can ONLY draft replies
    prompt=SystemMessage(content=(
        "You are the DRAFTER AGENT. You receive task-related emails.\n"
        "Your job is to draft professional, concise replies (under 100 words).\n"
        "Use the draft_email_reply tool to create each draft."
    )),
)


# ═════════════════════════════════════════════════════════════
#  GRAPH NODES (The Execution Blocks)
# ═════════════════════════════════════════════════════════════

def triage_node(state: MessagesState):
    """The Triage Agent reads emails and categorises them."""
    print("\n🔀 [TRIAGE AGENT] Analyzing inbox …")
    # We pass the current graph 'state' (which holds the message history) into the agent
    result = triage_agent.invoke(state)
    return {"messages": result["messages"]}


def router(state: MessagesState) -> Command[Literal["scheduler_node", "drafter_node", "__end__"]]:
    """
    This is the CONDITIONAL EDGE. It decides where to send the workflow next.
    It reads the output of the Triage agent and routes accordingly.
    """
    last_msg = state["messages"][-1]
    content = last_msg.content.lower() if hasattr(last_msg, "content") else ""

    has_meetings = "meeting" in content
    has_tasks = "task" in content

    if has_meetings:
        print("   📌 Route → SCHEDULER (meeting emails found)")
        return Command(goto="scheduler_node")
    elif has_tasks:
        print("   📌 Route → DRAFTER (task emails found)")
        return Command(goto="drafter_node")
    else:
        print("   📌 Route → END (only informational emails)")
        return Command(goto="__end__")


def scheduler_node(state: MessagesState):
    """The Scheduler Agent handles meeting requests."""
    print("\n📅 [SCHEDULER AGENT] Processing meeting requests …")
    result = scheduler_agent.invoke({
        "messages": state["messages"] + [
            HumanMessage(content=(
                "Based on the triage analysis above, schedule all "
                "detected meetings on Google Calendar."
            ))
        ]
    })
    return {"messages": result["messages"]}


def drafter_node(state: MessagesState):
    """The Drafter Agent creates reply drafts."""
    print("\n✍️  [DRAFTER AGENT] Drafting replies …")
    result = drafter_agent.invoke({
        "messages": state["messages"] + [
            HumanMessage(content=(
                "Based on the triage analysis above, draft professional "
                "replies for all task-related emails."
            ))
        ]
    })
    return {"messages": result["messages"]}


def human_review_node(state: MessagesState):
    """
    HUMAN-IN-THE-LOOP (HITL): High-stakes actions shouldn't happen automatically.
    This node pauses the entire system to ask for your permission.
    """
    print("\n🛑 [HUMAN REVIEW] The agent wants to send the following:")

    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    print(content[:800])

    # The interrupt() function literally pauses the Python script and saves the graph 
    # state to disk. It waits until a human resumes it.
    human_decision = interrupt("Do you approve this action? (yes/no): ")

    if human_decision.lower().strip() in ("yes", "y"):
        return {
            "messages": state["messages"] + [AIMessage(content="✅ Human approved. Action finalized.")]
        }
    else:
        return {
            "messages": state["messages"] + [AIMessage(content="❌ Human rejected this action. Discarding.")]
        }


# ═════════════════════════════════════════════════════════════
#  BUILD THE GRAPH (Wiring the flowchart)
# ═════════════════════════════════════════════════════════════

def build_graph():
    # We initialize a graph that uses MessagesState to track history
    graph = StateGraph(MessagesState)

    # 1. Add all our functions as nodes
    graph.add_node("triage_node", triage_node)
    graph.add_node("router", router)
    graph.add_node("scheduler_node", scheduler_node)
    graph.add_node("drafter_node", drafter_node)
    graph.add_node("human_review_node", human_review_node)

    # 2. Draw the lines (edges) between them
    # START -> Triage
    graph.add_edge(START, "triage_node")
    # Triage -> Router
    graph.add_edge("triage_node", "router")

    # After the specialist is done, force it to go to Human Review
    graph.add_edge("scheduler_node", "human_review_node")
    graph.add_edge("drafter_node", "human_review_node")
    # Human Review -> END
    graph.add_edge("human_review_node", END)

    # 3. Compile the graph with a Checkpointer
    # MemorySaver() allows the graph to be paused and resumed later!
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ═════════════════════════════════════════════════════════════
#  RUN THE GRAPH
# ═════════════════════════════════════════════════════════════

def run_multi_agent():
    print("=" * 60)
    print("  SESSION 3 : Multi-Agent System (LangGraph)")
    print("=" * 60)

    app = build_graph()
    
    # We need a Thread ID so the checkpointer knows which conversation to save/resume
    config = {"configurable": {"thread_id": "hackathon-demo-1"}}

    print("\n🚀 [System] Starting the multi-agent graph …")
    print("   Nodes: Triage → Router → Scheduler/Drafter → Human Review")
    print()

    # FIRST RUN: The graph will run from START until it hits the `interrupt()` in human_review
    result = app.invoke(
        {"messages": [HumanMessage(content="Analyze my inbox and handle everything appropriately.")]},
        config=config,
    )

    print("\n" + "─" * 50)
    print("📜 AGENT CONVERSATION (paused for review):")
    print("─" * 50)
    for msg in result["messages"]:
        role = msg.type.upper()
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") if isinstance(c, dict) and "text" in c else str(c) for c in content)
        if content:
            print(f"\n[{role}]: {content}")

    # Ask the user in the terminal
    print("\n" + "─" * 50)
    human_input = input("🛑 HUMAN-IN-THE-LOOP: Do you approve? (yes/no): ").strip()

    # SECOND RUN: We resume the exact same thread, injecting the human's answer
    result = app.invoke(Command(resume=human_input), config=config)

    print("\n" + "─" * 50)
    print("📜 FINAL RESULT:")
    print("─" * 50)
    last_msg = result["messages"][-1]
    print(f"  {last_msg.content if hasattr(last_msg, 'content') else last_msg}")

    print()
    print("─" * 50)
    print("🎓 KEY TAKEAWAYS:")
    print("   1. SPECIALISATION — Each agent has its own tools and prompt.")
    print("      The triage agent CANNOT schedule; the scheduler CANNOT draft.")
    print("   2. ROUTING — The graph decides which agent handles what.")
    print("   3. HUMAN-IN-THE-LOOP — The graph PAUSED and waited for you!")
    print("   4. CHECKPOINTING — If the server crashed after triage,")
    print("      it would resume from the checkpoint, not start over.")
    print("=" * 60)


if __name__ == "__main__":
    run_multi_agent()
