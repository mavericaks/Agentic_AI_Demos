# 🤖 Inbox Intelligence: Agentic AI Framework

Welcome to the **Agentic AI Hackathon** repository! This project serves as a comprehensive, hands-on masterclass in building advanced AI agents. 

Unlike standard text-generation bots, **Agentic AI** systems have the ability to *reason*, *use tools*, *interact with external APIs*, and *self-correct*. This repository takes you step-by-step from a traditional LLM script all the way to a Multi-Agent, self-learning Reflection loop, using a practical **Inbox Intelligence** scenario: an agent that reads your emails, categorizes them, drafts replies, and schedules calendar events autonomously.

---

## 🎯 What You Will Learn

This repository is split into progressive sessions. By running each demo, you will understand how AI architectures evolve:

- **Session 1: The Vanilla Agent** — Understand the core ReAct (Reason + Act) loop from scratch.
- **Session 2: Frameworks & RAG** — Scale up with LangChain, inject personal knowledge via ChromaDB (RAG), and magically discover tools using the Model Context Protocol (MCP).
- **Session 3: Distributed Multi-Agent Systems** — Use LangGraph to build a team of specialized agents (Triage, Drafter, Scheduler) that collaborate and pause for Human-in-the-Loop approval.
- **Session 4: Learning Agents** — Implement the Reflexion paradigm, where the agent critiques its own work and saves successful strategies to an Episodic Memory JSON file, improving over time without model retraining.

---

## 📁 Repository Structure

```text
├── .env.example                          # Template for API keys
├── requirements.txt                      # Python dependencies
├── main.py                               # Web Dashboard (FastAPI baseline UI)
│
├── utils/                                # Shared utilities
│   ├── auth.py                           #   Google OAuth Flow
│   ├── gmail_utils.py                    #   Fetch inbox messages
│   ├── calendar_utils.py                 #   Schedule meetings
│   ├── dns_patch.py                      #   Network stability patch
│   └── llm_router.py                     #   Multi-provider LLM Load Balancer
│
├── session_1_vanilla/                    # Session 1: No frameworks
│   ├── demo_1a_passive_llm.py            #   Demo: LLM can only talk
│   └── demo_1b_vanilla_agent.py          #   Demo: Agent can ACT (while loop)
│
├── session_2_frameworks/                 # Session 2: LangChain + RAG + MCP
│   ├── demo_2a_langchain_agent.py        #   Demo: Clean agent with LangChain
│   ├── demo_2b_rag_agent.py              #   Demo: Agent with personal knowledge
│   ├── user_preferences.txt              #   Private data for RAG DB
│   ├── mcp_server.py                     #   MCP tool provider
│   └── demo_2c_mcp_client.py             #   Demo: Auto-discovered tools
│
├── session_3_distributed/                # Session 3: Multi-agent
│   └── demo_3_multi_agent.py             #   Demo: Team of agents + Human-approval
│
├── session_4_learning/                   # Session 4: Reflexion
│   └── demo_4_reflexion.py               #   Demo: Self-correcting & learning agent
```

---

## 🔧 Setup & Execution: Getting Started

Follow these steps carefully to configure your environment before running the demos. 

> [!WARNING]
> **Read the instructions carefully.** Missing a step in the Google Cloud setup or forgetting to activate your virtual environment are the #1 reasons students get stuck!

### Step 1: Setup Environment & Install Dependencies
We provide automated setup scripts to create an isolated Python virtual environment (`venv`) and install all dependencies safely.

**On Windows:**
```cmd
setup.bat
```

**On Linux/macOS:**
```bash
bash setup.sh
```

> [!IMPORTANT]
> **VITAL STEP:** Every time you open a new terminal, you MUST activate the virtual environment before running the code!
> - Windows: `venv\Scripts\activate`
> - Linux/Mac: `source venv/bin/activate`

### Step 2: Configure Environment Variables (.env)
This repository uses a high-throughput multiplexing router that load-balances across different AI providers. You don't need all of them, but you need at least one API key.

1. **Open the `.env` file** (created automatically by the setup script in Step 1).
2. **Add your API Keys** inside the `.env` file without quotes. You can generate free API keys from:
   - [Google AI Studio (Gemini)](https://aistudio.google.com/)
   - [Groq Console (Llama 3)](https://console.groq.com/keys)
   - [GitHub Models](https://github.com/marketplace/models)

### Step 3: Google Workspace API Setup (credentials.json)
This project actually reads your Gmail and can insert events into your Google Calendar. 
*Note: Your data stays entirely local to your machine. The `.gitignore` ensures secrets are never pushed to GitHub.*

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
2. Enable **BOTH** the **Gmail API** AND **Google Calendar API**. *(Do not skip the Calendar API!)*
3. Go to **OAuth consent screen**, set up an **External** app.
4. **CRITICAL:** On the consent screen setup, you MUST add your own email address to the **"Test Users"** section. If you don't do this, Google will block your login later.
5. Go to **Credentials**, click **Create Credentials → OAuth client ID** (Choose "Desktop app").
6. Click **Download JSON**, rename the downloaded file to **exactly** `credentials.json`, and place it in the **root** of this project folder (not inside a subfolder).

### Step 4: First-Time Authentication
Run this one-liner to securely authenticate. **Make sure your venv is activated first!**

```bash
python -c "from utils.auth import get_credentials; get_credentials()"
```

> [!TIP]
> A browser window will pop up asking you to sign in. 
> You will see a scary screen that says **"Google hasn't verified this app"**. This is normal because you just created it!
> Click **"Advanced"** at the bottom, then click **"Go to Inbox Intelligence Agent (unsafe)"** to proceed.

*This will generate a local `token.json` file so you don't have to log in every time.*

---

## 🚀 Execution Guide: Running the Demos

You have two options for running the demos: **Jupyter Notebooks (Recommended)** or **Python Scripts**. Ensure your `venv` is activated!

### Option A: Jupyter Notebooks (Cell-by-Cell Execution)
Perfect for beginners. You can run the code block-by-block and inspect the variables.
1. Start Jupyter Lab from your terminal:
   ```bash
   jupyter lab
   ```
2. A browser window will open. Navigate to the `session_*` folders and double-click the `.ipynb` files!

### Option B: Python Scripts (Terminal Execution)
Run the full scripts directly from the terminal.

#### Session 1: The Paradigm Shift
*Prove that an LLM alone cannot take actions, then build the ReAct agent loop manually.*
```bash
python session_1_vanilla/demo_1a_passive_llm.py
python session_1_vanilla/demo_1b_vanilla_agent.py
```

#### Session 2: Frameworks, Memory & Tools
*Inject private rules via Vector Databases (RAG) and dynamically load tool definitions (MCP).*
```bash
python session_2_frameworks/demo_2a_langchain_agent.py
python session_2_frameworks/demo_2b_rag_agent.py
python session_2_frameworks/demo_2c_mcp_client.py
```

#### Session 3: Distributed Agent Teams
*Watch a LangGraph system route tasks between a Triage Agent, a Drafter, and a Scheduler, pausing for your manual approval.*
```bash
python session_3_distributed/demo_3_multi_agent.py
```

#### Session 4: Experiential Learning
*Run this twice! The agent drafts an email, an Evaluator agent critiques it (Fail/Pass), and the final result is saved format to Episodic Memory. On the second run, watch the agent recall its past mistakes to draft a better email instantly.*
```bash
python session_4_learning/demo_4_reflexion.py
```

### Optional: Start the Web UI
There is a baseline FastAPI Web Dashboard included to visualize the inbox flow.
```bash
uvicorn main:app --reload
```

---

## 🆘 Troubleshooting Common Student Errors

| Error | Cause & Solution |
|-------|------------------|
| `ModuleNotFoundError: No module named 'langchain'` | **Cause:** You forgot to activate your virtual environment.<br>**Solution:** Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) in your terminal, then try again. |
| Browser says "Access Blocked: Authorization Error" | **Cause:** You didn't add your email as a Test User in Google Cloud.<br>**Solution:** Go to Google Cloud Console > OAuth Consent Screen > scroll down to "Test Users" and add your exact email address. |
| Agent crashes when trying to schedule a meeting | **Cause:** You forgot to enable the Calendar API.<br>**Solution:** Go to Google Cloud Console > APIs & Services > Library > Search "Google Calendar API" and click Enable. |
| "Token has been expired or revoked" | **Cause:** You changed your OAuth scopes or your token got stale.<br>**Solution:** Delete the `token.json` file from your project folder and re-run the script to log in again. |

---

## 🤝 Contribution & License
Feel free to fork this project to experiment with your own `@tool` injections or LangGraph routing node architectures! Make sure to NEVER commit your `.env` or `.json` secrets to source control. Happy Hacking!
