"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 1B  —  The Vanilla Agent (ReAct Loop)              ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Build a REAL AI Agent from scratch.                  ║
║  No LangChain, no LangGraph — just a Python while-loop     ║
║  that lets the LLM call actual Python functions.            ║
║                                                             ║
║  The ReAct pattern:  Reason → Act → Observe → Repeat       ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_1_vanilla/demo_1b_vanilla_agent.py
"""

import os
import sys
import json

# ── Shared bootstrap (warnings, encoding, sys.path, dns, dotenv) ─
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.bootstrap  # noqa: E402, F401

import google.generativeai as genai
from utils.gmail_utils import fetch_recent_emails
from utils.calendar_utils import create_calendar_event
from utils.tools import format_emails

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ═════════════════════════════════════════════════════════════
#  TOOL REGISTRY
#  These are the real Python functions the agent can call.
#  NOTE: We intentionally avoid LangChain here to teach
#  the raw mechanics of how an agent actually works.
# ═════════════════════════════════════════════════════════════

def tool_fetch_emails(**kwargs) -> str:
    """Fetch recent emails from Gmail and return them as formatted text."""
    limit = int(kwargs.get("limit", 5))
    emails = fetch_recent_emails(limit=limit)
    return format_emails(emails)


def tool_schedule_meeting(**kwargs) -> str:
    """Schedule a meeting on Google Calendar with attendees and a Meet link."""
    time = kwargs.get("time", "")
    attendees_raw = kwargs.get("attendees", [])
    title = kwargs.get("title", "AI Scheduled Meeting")

    # Handle both comma-separated string and list types passed by the LLM
    if isinstance(attendees_raw, str):
        attendees = [a.strip() for a in attendees_raw.split(",") if a.strip()]
    else:
        attendees = list(attendees_raw)

    try:
        link = create_calendar_event(time, attendees, title)
        if "already scheduled" in link:
            return link
        return f"✅ Meeting '{title}' scheduled at {time}. Calendar link: {link}"
    except ValueError as e:
        return f"❌ {str(e)}"


# ── The Tool Map ──
# We create a dictionary so we can easily look up the function based on the string name the LLM provides.
TOOLS = {
    "fetch_emails": {
        "function": tool_fetch_emails,
        "description": (
            "Fetch recent emails from the user's Gmail inbox. "
            "Optional arg: limit (int, default 5)."
        ),
    },
    "schedule_meeting": {
        "function": tool_schedule_meeting,
        "description": (
            "Schedule a meeting on Google Calendar and send invites. "
            "Required args: time (str, format 'YYYY-MM-DD HH:MM'), "
            "attendees (list of email strings), title (str)."
        ),
    },
}


# ═════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  —  tells the LLM how to use tools via JSON
# ═════════════════════════════════════════════════════════════
# This is the "brain conditioning". We tell the LLM exactly how we expect it to behave.
# Crucially, we force it to reply in JSON so our Python script can easily parse its decisions.

SYSTEM_PROMPT = """You are an AI email assistant.
You have access to these tools:

1. fetch_emails(limit)
   → Fetches recent emails from the user's Gmail inbox.

2. schedule_meeting(time, attendees, title)
   → Schedules a Google Calendar event with a Meet link.
   → time must be in 'YYYY-MM-DD HH:MM' format.
   → attendees is a list of email addresses.

RULES (follow these EXACTLY):
• To call a tool, output EXACTLY ONE JSON object on a single line:
  {"tool": "tool_name", "args": {"arg1": "value1"}}
• After receiving tool results, decide your next action.
• When your task is COMPLETE, output:
  {"tool": "DONE", "summary": "Your final summary here"}
• NEVER include markdown fences, explanations, or anything outside the JSON.
"""


# ═════════════════════════════════════════════════════════════
#  THE REACT LOOP  —  the core of every AI agent
# ═════════════════════════════════════════════════════════════

def extract_json_from_response(text: str) -> dict | None:
    """Try to parse JSON from the LLM's response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if the LLM hallucinated them
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def run_vanilla_agent():
    print("=" * 60)
    print("  SESSION 1B : The Vanilla Agent (ReAct Loop)")
    print("=" * 60)

    model = genai.GenerativeModel("gemini-flash-latest")

    # ── CONVERSATION HISTORY (The Agent's Memory) ──
    # Every time the LLM acts or sees a result, we append it to this list.
    # This prevents the LLM from forgetting what it did 2 steps ago.
    conversation = [
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ['{"tool": "acknowledge", "status": "ready"}']},
        {
            "role": "user",
            "parts": [
                "Please fetch my recent emails, analyze them, "
                "and if you find any meeting requests, schedule them on my calendar."
            ],
        },
    ]

    print("\n🔄 [Agent] Starting the ReAct loop …\n")

    # ── THE LOOP ──
    # This loop is what turns a passive LLM into an active Agent.
    MAX_ITERATIONS = 10
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"── Iteration {i} {'─' * 40}")

        # ── REASON: Ask the LLM what to do next ─────────────
        # We pass the ENTIRE conversation history so it knows the current state.
        response = model.generate_content(conversation)
        raw = response.text.strip()
        print(f"   🧠 LLM says: {raw[:200]}")

        # Convert the LLM's text output into a Python Dictionary
        action = extract_json_from_response(raw)

        if action is None:
            print("   ⚠  Non-JSON response. Nudging LLM …")
            conversation.append({"role": "model", "parts": [raw]})
            conversation.append({
                "role": "user",
                "parts": ["Please respond with ONLY a JSON object. No other text."],
            })
            continue

        tool_name = action.get("tool", "")

        # ── CHECK: Is the agent done? ────────────────────────
        if tool_name == "DONE":
            print(f"\n✅ [Agent] FINISHED!")
            print(f"   📋 Summary: {action.get('summary', 'No summary.')}")
            break

        # ── ACT: Execute the tool ────────────────────────────
        # Here is where Python bridges the gap between text and action.
        if tool_name in TOOLS:
            args = action.get("args", {})
            print(f"   🔧 Calling tool: {tool_name}({args})")

            try:
                # Actually execute the Python function mapping to the requested tool
                result = TOOLS[tool_name]["function"](**args)
                print(f"   📦 Result: {str(result)[:300]}")
            except Exception as e:
                result = f"❌ ERROR: {e}"
                print(f"   {result}")

            # ── OBSERVE: Feed result back to the LLM ────────
            # We append what the tool returned so the LLM can read it on the next loop iteration.
            conversation.append({"role": "model", "parts": [raw]})
            conversation.append({
                "role": "user",
                "parts": [
                    f"Tool '{tool_name}' returned:\n{result}\n\n"
                    f"What is your next action? Reply with JSON only."
                ],
            })
        else:
            print(f"   ❓ Unknown tool: {tool_name}")
            conversation.append({"role": "model", "parts": [raw]})
            conversation.append({
                "role": "user",
                "parts": [
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {list(TOOLS.keys())}. Try again."
                ],
            })

    print()
    print("🎓 KEY TAKEAWAY:")
    print("   Compare this to Demo 1A (the passive LLM).")
    print("   The LLM's 'brain' is the same — but now it lives inside")
    print("   a while-loop that lets it CALL REAL FUNCTIONS.")
    print("   That loop is what makes it an AGENT.")
    print("=" * 60)


if __name__ == "__main__":
    run_vanilla_agent()
