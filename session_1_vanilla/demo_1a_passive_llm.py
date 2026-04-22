"""
╔══════════════════════════════════════════════════════════════╗
║  SESSION 1A  —  The Passive LLM (The Baseline)             ║
╠══════════════════════════════════════════════════════════════╣
║  GOAL: Prove that a standard LLM is ONLY a text predictor. ║
║  It can read and summarize emails, but it CANNOT actually   ║
║  schedule a meeting or take any real-world action.          ║
╚══════════════════════════════════════════════════════════════╝

Run:   python session_1_vanilla/demo_1a_passive_llm.py
"""

import os
import sys

# ── Shared bootstrap (warnings, encoding, sys.path, dns, dotenv) ─
# This ensures our API keys from the .env file are loaded into os.environ
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.bootstrap  # noqa: E402, F401

import google.generativeai as genai
from utils.gmail_utils import fetch_recent_emails
from utils.tools import format_emails

# Configure the Gemini library with the API key loaded from our .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def run_passive_llm():
    print("=" * 60)
    print("  SESSION 1A : The Passive LLM (Text Predictor)")
    print("=" * 60)

    # ── STEP 1: Fetch real emails from your Gmail ────────────
    # We use our custom Gmail utility to securely connect to your inbox.
    # limit=5 ensures we only pull the 5 most recent threads to save token space.
    print("\n📥 [Step 1] Fetching your 5 most recent emails …")
    emails = fetch_recent_emails(limit=5)

    if not emails:
        print("   ⚠  No emails found. Make sure credentials.json is set up.")
        return

    # ── STEP 2: Format emails as plain text for the LLM ─────
    # The Gmail API returns complex JSON with headers and metadata.
    # LLMs perform much better when data is formatted as clean, readable text.
    email_text = format_emails(emails, include_date=False)

    # ── STEP 3: Ask the LLM to analyse AND schedule ─────────
    # We construct a prompt string. Notice how we are explicitly commanding it
    # to perform an action: "Schedule the meeting on my Google Calendar right now."
    prompt = f"""Here are my recent emails:
{email_text}

Please do the following:
1. Categorize each email (urgent / meeting / task / informational).
2. Identify any meeting requests and extract the proposed time.
3. **Schedule the meeting on my Google Calendar right now.**
4. Give me a summary of what you did.
"""

    print("\n🧠 [Step 2] Sending emails to Gemini LLM …")
    
    # Initialize the Google Gemini model. "gemini-flash-latest" is fast and lightweight.
    model = genai.GenerativeModel("gemini-flash-latest")
    
    # We send the text prompt to the model and wait for it to generate a text response.
    response = model.generate_content(prompt)

    print("\n💬 [Step 3] LLM Response:")
    print("-" * 50)
    print(response.text)
    print("-" * 50)

    # ── THE KEY LESSON ───────────────────────────────────────
    # Read the output above. The LLM will boldly claim "I have scheduled the meeting."
    # But if you check your actual Google Calendar, it is empty.
    # Why? Because standard LLMs are trapped in a text-generation box. They have no arms or legs.
    print()
    print("⚠️  IMPORTANT OBSERVATION:")
    print("   The LLM *says* it scheduled the meeting …")
    print("   But open your Google Calendar — NOTHING was created!")
    print()
    print("   WHY?  Because a standard LLM is a TEXT PREDICTOR.")
    print("   It has NO ability to call APIs or execute code.")
    print("   It can only generate text that *looks* like an answer.")
    print()
    print("   👉 THIS is why we need AGENTS (see Demo 1B).")
    print("=" * 60)


if __name__ == "__main__":
    run_passive_llm()
