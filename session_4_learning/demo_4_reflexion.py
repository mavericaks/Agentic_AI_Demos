"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 4  —  Reflexion (Learning Agents)                  ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Build an agent that grades its own work and saves    ║
║  the lessons learned to "Memory" so it never makes the      ║
║  same mistake twice.                                        ║
║                                                             ║
║  The Loop:                                                  ║
║  1. ACTOR drafts an email.                                  ║
║  2. EVALUATOR critiques it (Pass/Fail) based on a rubric.   ║
║  3. If Fail → Actor tries again using the critique.         ║
║  4. If Pass → Save the task + final output to MEMORY.       ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_4_learning/demo_4_reflexion.py
"""

import os
import sys
import json

# ── Shared bootstrap (warnings, encoding, sys.path, dns, dotenv) ─
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.bootstrap  # noqa: E402, F401

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
import operator

from utils.tools import fetch_emails, draft_email_reply
from utils.llm_router import get_routed_llm

# ═════════════════════════════════════════════════════════════
#  EPISODIC MEMORY (The "Brain" File)
# ═════════════════════════════════════════════════════════════
# Instead of fine-tuning the model's weights (which is expensive),
# we save successful workflows to a local JSON file. 
# On future runs, the agent reads this file to remember how it succeeded in the past.

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "episodic_memory.json")

def load_memory() -> list:
    """Read past successful experiences from disk."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(task: str, successful_result: str, critique: str):
    """Save a new successful experience to disk."""
    memory = load_memory()
    memory.append({
        "task": task,
        "successful_result": successful_result,
        "lesson_learned": critique
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# ═════════════════════════════════════════════════════════════
#  GRAPH STATE
# ═════════════════════════════════════════════════════════════
# In LangGraph, the 'State' is the data that flows between nodes.
# Here, we need to track the current draft, the critique, and how many times we've tried.

class ReflexionState(TypedDict):
    task: str
    current_draft: str
    critique: str
    is_passing: bool
    iterations: int


# ═════════════════════════════════════════════════════════════
#  THE AGENTS (Actor vs Evaluator)
# ═════════════════════════════════════════════════════════════

llm = get_routed_llm(role="master_model")

# 1. The ACTOR is the worker. It drafts the email.
actor_agent = create_react_agent(
    llm,
    tools=[fetch_emails, draft_email_reply],
    prompt=SystemMessage(content=(
        "You are the DRAFTER AGENT.\n"
        "Your job is to read the task and output a professional email draft.\n"
        "If you are given a critique of a previous draft, you MUST rewrite the draft to address the critique."
    )),
)

# 2. The EVALUATOR is the boss. It grades the Actor's work using a rubric.
# It does NOT have any tools. It only analyzes text.
evaluator_agent = create_react_agent(
    llm,
    tools=[], 
    prompt=SystemMessage(content=(
        "You are the EVALUATOR AGENT.\n"
        "Analyze the provided email draft against this strict rubric:\n"
        "  1. Tone: Must be exceptionally professional and polite.\n"
        "  2. Length: Must be under 100 words.\n"
        "  3. Structure: Must have a clear greeting, body, and sign-off.\n\n"
        "Respond with EXACTLY this JSON format:\n"
        '{"pass": true/false, "critique": "detailed feedback on what is wrong and how to fix it"}'
    )),
)


# ═════════════════════════════════════════════════════════════
#  GRAPH NODES
# ═════════════════════════════════════════════════════════════

def actor_node(state: ReflexionState):
    """The Actor drafts an email, injecting past memories if available."""
    print(f"\n✍️  [ACTOR] Drafting (Iteration {state['iterations'] + 1}) …")
    
    # Check if we have past memories to help us
    memories = load_memory()
    memory_context = ""
    if memories:
        memory_context = "PAST SUCCESSFUL EXAMPLES TO LEARN FROM:\n"
        for i, m in enumerate(memories[-3:]): # Only load the 3 most recent to save tokens
            memory_context += f"Example {i+1}: {m['successful_result']}\n"
    
    prompt = f"Task: {state['task']}\n\n{memory_context}"
    
    # If the Evaluator rejected the last draft, we feed the critique back to the Actor
    if state['critique']:
        prompt += f"\n\nYour last draft was REJECTED. Critique: {state['critique']}\nRewrite it to be better."
        
    result = actor_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    last_msg = result["messages"][-1].content
    
    return {
        "current_draft": str(last_msg),
        "iterations": state['iterations'] + 1
    }


def evaluator_node(state: ReflexionState):
    """The Evaluator grades the draft using the JSON rubric."""
    print("\n🧐 [EVALUATOR] Grading draft …")
    
    prompt = f"Task: {state['task']}\n\nDraft to evaluate:\n{state['current_draft']}"
    result = evaluator_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    # Extract the JSON response from the Evaluator
    raw_response = result["messages"][-1].content
    
    try:
        # Strip markdown fences if present
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0]
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0]
        else:
            json_str = raw_response
            
        evaluation = json.loads(json_str.strip())
        is_pass = evaluation.get("pass", False)
        critique = evaluation.get("critique", "No critique provided.")
    except Exception as e:
        print(f"   [Warning] Failed to parse evaluator JSON: {e}")
        is_pass = False
        critique = "Failed to evaluate properly. Please try again."

    print(f"   Result: {'✅ PASS' if is_pass else '❌ FAIL'}")
    print(f"   Critique: {critique}")
    
    return {
        "is_passing": is_pass,
        "critique": critique
    }


def route_evaluation(state: ReflexionState):
    """
    The Conditional Edge.
    If Pass -> We are done.
    If Fail -> Send back to the Actor to try again (max 3 tries).
    """
    if state["is_passing"]:
        print("   📌 Route → MEMORY SAVER (Draft passed!)")
        return "save_memory_node"
    elif state["iterations"] >= 3:
        print("   📌 Route → END (Max iterations reached, giving up)")
        return END
    else:
        print("   📌 Route → ACTOR (Draft failed, try again)")
        return "actor_node"


def save_memory_node(state: ReflexionState):
    """If the draft passed, we save it to Episodic Memory so we never forget it."""
    print("\n💾 [MEMORY] Saving successful draft to episodic_memory.json …")
    save_memory(
        task=state["task"],
        successful_result=state["current_draft"],
        critique=state["critique"]
    )
    return state


# ═════════════════════════════════════════════════════════════
#  BUILD & RUN THE GRAPH
# ═════════════════════════════════════════════════════════════

def build_reflexion_graph():
    graph = StateGraph(ReflexionState)
    
    graph.add_node("actor_node", actor_node)
    graph.add_node("evaluator_node", evaluator_node)
    graph.add_node("save_memory_node", save_memory_node)
    
    graph.add_edge(START, "actor_node")
    graph.add_edge("actor_node", "evaluator_node")
    
    # Conditional routing based on pass/fail
    graph.add_conditional_edges("evaluator_node", route_evaluation)
    
    graph.add_edge("save_memory_node", END)
    
    return graph.compile()


def run_reflexion():
    print("=" * 60)
    print("  SESSION 4 : Reflexion (Learning Agents)")
    print("=" * 60)

    app = build_reflexion_graph()
    
    # A hard task that typically fails on the first try
    task = (
        "Draft an email to a highly demanding client apologizing for a 2-week delay "
        "on the Q3 report. It must be extremely professional, under 100 words, "
        "and have a clear greeting, body, and sign-off."
    )
    
    initial_state = {
        "task": task,
        "current_draft": "",
        "critique": "",
        "is_passing": False,
        "iterations": 0
    }
    
    print("\n🚀 [System] Starting the Reflexion loop …")
    print(f"   Task: {task}")
    
    # Run the graph
    app.invoke(initial_state)

    print("\n" + "─" * 50)
    print("🎓 KEY TAKEAWAYS:")
    print("   1. ACTOR-EVALUATOR PARADIGM — The agent critiques its own work.")
    print("   2. RETRY LOOP — It doesn't give up. It uses the critique to rewrite.")
    print("   3. EPISODIC MEMORY — It saved the final passing draft to disk.")
    print()
    print("   👉 RUN THIS SCRIPT AGAIN! Watch it load the past memory.")
    print("      It will likely PASS on the very first try this time,")
    print("      because it 'learned' without us changing the model weights!")
    print("=" * 60)

if __name__ == "__main__":
    run_reflexion()
