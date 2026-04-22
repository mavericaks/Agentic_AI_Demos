import json
import os

def write_nb(path, cells):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
    for ctype, src in cells:
        cell = {"cell_type": ctype, "metadata": {}, "source": [line + "\n" for line in src.split("\n")]}
        if cell["source"] and cell["source"][-1].endswith("\n"): 
            cell["source"][-1] = cell["source"][-1].rstrip("\n")
        if ctype == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb["cells"].append(cell)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

# ==========================================
# DEMO 1A
# ==========================================
cells_1a = [
    ("markdown", "# Session 1A: The Passive LLM\n**Goal**: Prove that a standard LLM is ONLY a text predictor. It cannot take actions."),
    ("code", "import os, sys\n# Ensure root path is accessible\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\n\nimport google.generativeai as genai\nfrom utils.gmail_utils import fetch_recent_emails\nfrom utils.tools import format_emails\n\ngenai.configure(api_key=os.getenv('GOOGLE_API_KEY'))"),
    ("markdown", "### Step 1: Fetch Real Emails"),
    ("code", "print('Fetching your 5 most recent emails...')\nemails = fetch_recent_emails(limit=5)\nif not emails:\n    raise RuntimeError('No emails found. Make sure credentials.json is set up.')"),
    ("markdown", "### Step 2: Format & Send to LLM"),
    ("code", "email_text = format_emails(emails, include_date=False)\nprompt = f\"\"\"Here are my recent emails:\\n{email_text}\\n\\nPlease do the following:\\n1. Categorize each email.\\n2. Schedule the meeting on my Google Calendar right now.\"\"\"\n\nmodel = genai.GenerativeModel('gemini-flash-latest')\nresponse = model.generate_content(prompt)\nprint(response.text)"),
    ("markdown", "### The Lesson\nThe LLM *says* it scheduled the meeting... but check your calendar. Nothing is there!\nAn LLM is just a text predictor. It needs tools to become an agent.")
]
write_nb("session_1_vanilla/demo_1a_passive_llm.ipynb", cells_1a)

# ==========================================
# DEMO 1B
# ==========================================
cells_1b = [
    ("markdown", "# Session 1B: The Vanilla Agent\n**Goal**: Build a ReAct (Reason+Act) loop from scratch to let the LLM call real functions."),
    ("code", "import os, sys, json\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\nimport google.generativeai as genai\nfrom utils.gmail_utils import fetch_recent_emails\nfrom utils.calendar_utils import create_calendar_event\nfrom utils.tools import format_emails\n\ngenai.configure(api_key=os.getenv('GOOGLE_API_KEY'))"),
    ("markdown", "### Step 1: Define the Tools (Real Python Functions)"),
    ("code", "def tool_fetch_emails(**kwargs):\n    limit = int(kwargs.get('limit', 5))\n    return format_emails(fetch_recent_emails(limit=limit))\n\ndef tool_schedule_meeting(**kwargs):\n    time = kwargs.get('time', '')\n    attendees_raw = kwargs.get('attendees', [])\n    title = kwargs.get('title', 'AI Scheduled Meeting')\n    attendees = [a.strip() for a in attendees_raw.split(',')] if isinstance(attendees_raw, str) else list(attendees_raw)\n    try:\n        link = create_calendar_event(time, attendees, title)\n        if 'already scheduled' in link: return link\n        return f\"Meeting '{title}' scheduled at {time}. Link: {link}\"\n    except ValueError as e:\n        return f\"Error: {str(e)}\"\n\nTOOLS = {\n    'fetch_emails': {'function': tool_fetch_emails},\n    'schedule_meeting': {'function': tool_schedule_meeting}\n}"),
    ("markdown", "### Step 2: Set the System Prompt & Init Loop"),
    ("code", "SYSTEM_PROMPT = \"\"\"You are an AI email assistant.\\nYou have access to these tools:\\n1. fetch_emails(limit)\\n2. schedule_meeting(time, attendees, title)\\n\\nRULES:\\nTo call a tool, output EXACTLY ONE JSON object: {\"tool\": \"name\", \"args\": {}}\\nWhen done, output: {\"tool\": \"DONE\", \"summary\": \"...\"}\"\"\"\n\nmodel = genai.GenerativeModel('gemini-flash-latest')\nconversation = [\n    {\"role\": \"user\", \"parts\": [SYSTEM_PROMPT]},\n    {\"role\": \"model\", \"parts\": ['{\"tool\": \"acknowledge\", \"status\": \"ready\"}']},\n    {\"role\": \"user\", \"parts\": [\"Fetch my emails, analyze them, and schedule any meetings.\"]}\n]"),
    ("markdown", "### Step 3: Run the Agent Loop"),
    ("code", "import re\nfor i in range(1, 6):\n    print(f\"\\n--- Iteration {i} ---\")\n    response = model.generate_content(conversation)\n    raw = response.text.strip()\n    print(f\"LLM says: {raw[:200]}...\")\n    \n    # Parse JSON\n    match = re.search(r'\\{[^{}]*\\}', raw)\n    if not match: break\n    action = json.loads(match.group())\n    \n    tool_name = action.get('tool')\n    if tool_name == 'DONE':\n        print(\"\\n\\u2705 FINISHED!\", action.get('summary'))\n        break\n        \n    print(f\"\\ud83d\\udd27 Calling {tool_name}...\")\n    args = action.get('args', {})\n    result = TOOLS[tool_name]['function'](**args)\n    print(f\"\\ud83d\\udce6 Result: {str(result)[:100]}...\")\n    \n    # Feedback loop\n    conversation.append({\"role\": \"model\", \"parts\": [raw]})\n    conversation.append({\"role\": \"user\", \"parts\": [f\"Tool returned: {result}\\nNext action? (Reply in JSON)\"]})")
]
write_nb("session_1_vanilla/demo_1b_vanilla_agent.ipynb", cells_1b)

# ==========================================
# DEMO 2A
# ==========================================
cells_2a = [
    ("markdown", "# Session 2A: LangChain Agent\n**Goal**: Replace the messy while-loop with LangChain's clean framework."),
    ("code", "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\n\nfrom langchain_google_genai import ChatGoogleGenerativeAI\nfrom langchain_core.messages import HumanMessage\nfrom langgraph.prebuilt import create_react_agent\nfrom utils.tools import fetch_emails, schedule_meeting"),
    ("markdown", "### Setup and Run Agent"),
    ("code", "llm = ChatGoogleGenerativeAI(model=\"gemini-flash-latest\", temperature=0)\ntools = [fetch_emails, schedule_meeting]\nagent = create_react_agent(llm, tools)\n\nprint(\"\\ud83e\\udd16 Running LangChain Agent...\")\nresult = agent.invoke({\"messages\": [HumanMessage(content=\"Fetch my recent emails, analyze them, and schedule any meeting requests.\")]})\n\nfor msg in result['messages']:\n    print(f\"\\n[{msg.type.upper()}]: {str(msg.content)[:300]}\")\n    if hasattr(msg, 'tool_calls') and msg.tool_calls:\n        for tc in msg.tool_calls:\n            print(f\"  \\ud83d\\udd27 Tool Call: {tc['name']}({tc['args']})\")")
]
write_nb("session_2_frameworks/demo_2a_langchain_agent.ipynb", cells_2a)

# ==========================================
# DEMO 2B
# ==========================================
cells_2b = [
    ("markdown", "# Session 2B: RAG-Powered Agent\n**Goal**: Give the agent semantic memory so it reads your rules BEFORE scheduling."),
    ("code", "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\n\nfrom langchain_google_genai import GoogleGenerativeAIEmbeddings\nfrom langchain_core.tools import tool\nfrom langchain_core.messages import HumanMessage, SystemMessage\nfrom langchain_chroma import Chroma\nfrom langgraph.prebuilt import create_react_agent\nfrom utils.tools import fetch_emails, schedule_meeting\nfrom utils.llm_router import get_routed_llm"),
    ("markdown", "### Load Vector Store (RAG)"),
    ("code", "CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(''))), 'session_2_frameworks', '.chroma_db')\nembeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')\nvectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)\n\n@tool\ndef search_user_preferences(query: str) -> str:\n    \"\"\"Search the user's personal preferences and rules.\"\"\"\n    docs = vectorstore.similarity_search(query, k=3)\n    return '\\n'.join([d.page_content for d in docs]) if docs else 'No rules found.'"),
    ("markdown", "### Run the Agent"),
    ("code", "llm = get_routed_llm(role='worker_model')\ntools = [fetch_emails, search_user_preferences, schedule_meeting]\n\nsystem_msg = SystemMessage(content=\"ALWAYS search preferences BEFORE scheduling. Refuse and suggest alternatives if it violates rules.\")\nagent = create_react_agent(llm, tools)\n\nprint(\"\\ud83e\\udd16 Running RAG Agent...\")\nresult = agent.invoke({\"messages\": [system_msg, HumanMessage(content=\"Fetch emails. Check my preferences before scheduling any meetings.\")]})\n\nfor msg in result['messages']:\n    print(f\"\\n[{msg.type.upper()}]: {str(msg.content)[:300]}\")\n    if hasattr(msg, 'tool_calls') and msg.tool_calls:\n        for tc in msg.tool_calls:\n            print(f\"  \\ud83d\\udd27 Tool Call: {tc['name']}({tc['args']})\")")
]
write_nb("session_2_frameworks/demo_2b_rag_agent.ipynb", cells_2b)

# ==========================================
# DEMO 2C
# ==========================================
cells_2c = [
    ("markdown", "# Session 2C: MCP Client\n**Goal**: Show 'USB-C for AI'. Tools are auto-discovered from a server without writing @tool locally."),
    ("code", "import os, sys, asyncio\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\n\nfrom langchain_core.messages import HumanMessage\nfrom langchain_mcp_adapters.client import MultiServerMCPClient\nfrom langgraph.prebuilt import create_react_agent\nfrom utils.llm_router import get_routed_llm"),
    ("markdown", "### Discover Tools & Run Agent (Async)"),
    ("code", "async def main():\n    MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(''))), 'session_2_frameworks', 'mcp_server.py')\n    client = MultiServerMCPClient({'inbox_server': {'command': sys.executable, 'args': [MCP_SERVER_SCRIPT], 'transport': 'stdio'}})\n    tools = await client.get_tools()\n    print(f\"\\ud83d\\udd0d Discovered {len(tools)} tools via MCP!\")\n    \n    llm = get_routed_llm(role='worker_model')\n    agent = create_react_agent(llm, tools)\n    \n    result = await agent.ainvoke({\"messages\": [HumanMessage(content=\"Fetch my emails, analyze them, and schedule any meetings.\")]})\n    for msg in result['messages']:\n        print(f\"\\n[{msg.type.upper()}]: {str(msg.content)[:300]}\")\n        if hasattr(msg, 'tool_calls') and msg.tool_calls:\n            for tc in msg.tool_calls:\n                print(f\"  \\ud83d\\udd27 Tool Call: {tc['name']}({tc['args']})\")\n\nawait main()")
]
write_nb("session_2_frameworks/demo_2c_mcp_client.ipynb", cells_2c)

# ==========================================
# DEMO 3
# ==========================================
cells_3 = [
    ("markdown", "# Session 3: Multi-Agent System (LangGraph)\n**Goal**: Break monolithic agents into a TEAM of specialized agents with a conditional router and Human-In-The-Loop."),
    ("code", "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\n\nfrom langchain_core.messages import HumanMessage, SystemMessage, AIMessage\nfrom langgraph.graph import StateGraph, MessagesState, START, END\nfrom langgraph.prebuilt import create_react_agent\nfrom langgraph.checkpoint.memory import MemorySaver\nfrom langgraph.types import interrupt, Command\nfrom typing import Literal\n\nfrom utils.tools import fetch_emails, schedule_meeting, draft_email_reply\nfrom utils.llm_router import get_routed_llm"),
    ("markdown", "### Define Specialized Sub-Agents"),
    ("code", "llm = get_routed_llm(role='master_model')\ntriage_agent = create_react_agent(llm, tools=[fetch_emails], prompt=SystemMessage(content=\"TRIAGE AGENT: Fetch emails and categorize as 'meeting', 'task', or 'info'.\"))\nscheduler_agent = create_react_agent(llm, tools=[schedule_meeting], prompt=SystemMessage(content=\"SCHEDULER AGENT: Schedule the requested meetings.\"))\ndrafter_agent = create_react_agent(llm, tools=[draft_email_reply], prompt=SystemMessage(content=\"DRAFTER AGENT: Draft replies for tasks.\"))"),
    ("markdown", "### Define Graph Nodes & Router"),
    ("code", "def triage_node(state: MessagesState):\n    print(\"\\n\\ud83d\\udd00 [TRIAGE] Analyzing...\")\n    return {\"messages\": triage_agent.invoke(state)[\"messages\"]}\n\ndef router(state: MessagesState) -> Command[Literal[\"scheduler_node\", \"drafter_node\", \"__end__\"]]:\n    content = state[\"messages\"][-1].content.lower()\n    if \"meeting\" in content: return Command(goto=\"scheduler_node\")\n    elif \"task\" in content: return Command(goto=\"drafter_node\")\n    return Command(goto=\"__end__\")\n\ndef scheduler_node(state: MessagesState):\n    print(\"\\n\\ud83d\\udcc5 [SCHEDULER] Booking...\")\n    return {\"messages\": scheduler_agent.invoke({\"messages\": state[\"messages\"] + [HumanMessage(content=\"Schedule the meetings\")]})[\"messages\"]}\n\ndef drafter_node(state: MessagesState):\n    print(\"\\n\\u270d\\ufe0f [DRAFTER] Drafting...\")\n    return {\"messages\": drafter_agent.invoke({\"messages\": state[\"messages\"] + [HumanMessage(content=\"Draft the replies\")]})[\"messages\"]}\n\ndef human_review_node(state: MessagesState):\n    print(\"\\n\\ud83d\\uded1 [HUMAN REVIEW] Requested action:\")\n    print(str(state[\"messages\"][-1].content)[:200])\n    decision = interrupt(\"Approve? (yes/no): \")\n    return {\"messages\": state[\"messages\"] + [AIMessage(content=f\"Human decision: {decision}\")]}"),
    ("markdown", "### Build & Run Graph"),
    ("code", "builder = StateGraph(MessagesState)\nbuilder.add_node(\"triage_node\", triage_node)\nbuilder.add_node(\"router\", router)\nbuilder.add_node(\"scheduler_node\", scheduler_node)\nbuilder.add_node(\"drafter_node\", drafter_node)\nbuilder.add_node(\"human_review_node\", human_review_node)\n\nbuilder.add_edge(START, \"triage_node\")\nbuilder.add_edge(\"triage_node\", \"router\")\nbuilder.add_edge(\"scheduler_node\", \"human_review_node\")\nbuilder.add_edge(\"drafter_node\", \"human_review_node\")\nbuilder.add_edge(\"human_review_node\", END)\n\napp = builder.compile(checkpointer=MemorySaver())\nconfig = {\"configurable\": {\"thread_id\": \"1\"}}\n\n# FIRST RUN (Will pause at interrupt)\nprint(\"\\n\\ud83d\\ude80 Starting Graph...\")\nres = app.invoke({\"messages\": [HumanMessage(content=\"Analyze inbox.\")]}, config=config)"),
    ("markdown", "### Resume Graph After Human Input"),
    ("code", "# SECOND RUN (Resuming)\nhuman_input = input(\"\\ud83d\\uded1 Type 'yes' to approve: \")\nres = app.invoke(Command(resume=human_input), config=config)\nprint(\"\\n\\u2728 FINAL:\", res[\"messages\"][-1].content)")
]
write_nb("session_3_distributed/demo_3_multi_agent.ipynb", cells_3)

# ==========================================
# DEMO 4
# ==========================================
cells_4 = [
    ("markdown", "# Session 4: Reflexion (Learning Agents)\n**Goal**: Agent critiques its own work and saves lessons to Episodic Memory."),
    ("code", "import os, sys, json\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))\nimport utils.bootstrap\n\nfrom langchain_core.messages import HumanMessage, SystemMessage\nfrom langgraph.graph import StateGraph, START, END\nfrom langgraph.prebuilt import create_react_agent\nfrom typing import TypedDict\nfrom utils.tools import draft_email_reply\nfrom utils.llm_router import get_routed_llm"),
    ("markdown", "### Setup State & Memory"),
    ("code", "class ReflexionState(TypedDict):\n    task: str\n    current_draft: str\n    critique: str\n    is_passing: bool\n    iterations: int\n\nMEMORY_FILE = 'episodic_memory.json'\n\ndef load_memory():\n    return json.load(open(MEMORY_FILE)) if os.path.exists(MEMORY_FILE) else []\n\ndef save_memory(task, draft, critique):\n    mem = load_memory()\n    mem.append({\"task\": task, \"draft\": draft, \"lesson\": critique})\n    json.dump(mem, open(MEMORY_FILE, 'w'))"),
    ("markdown", "### Setup Agents"),
    ("code", "llm = get_routed_llm(role='master_model')\nactor_agent = create_react_agent(llm, tools=[draft_email_reply], prompt=SystemMessage(content=\"DRAFTER: Output a professional email draft. Address any critiques.\"))\nevaluator_agent = create_react_agent(llm, tools=[], prompt=SystemMessage(content=\"EVALUATOR: Grade the draft. It must be < 100 words and polite. Respond ONLY with JSON: {'pass': bool, 'critique': '...'}\"))"),
    ("markdown", "### Define Graph Nodes"),
    ("code", "def actor_node(state: ReflexionState):\n    print(f\"\\n\\u270d\\ufe0f [ACTOR] Drafting (Iter {state['iterations'] + 1}) ...\")\n    mem_ctx = \"\\n\".join([m['draft'] for m in load_memory()[-2:]])\n    prompt = f\"Task: {state['task']}\\nPast Good Examples: {mem_ctx}\\nCritique: {state['critique']}\"\n    res = actor_agent.invoke({\"messages\": [HumanMessage(content=prompt)]})\n    return {\"current_draft\": str(res['messages'][-1].content), \"iterations\": state['iterations'] + 1}\n\ndef evaluator_node(state: ReflexionState):\n    print(\"\\n\\ud83e\\uddd0 [EVALUATOR] Grading...\")\n    res = evaluator_agent.invoke({\"messages\": [HumanMessage(content=f\"Draft: {state['current_draft']}\")]})\n    try:\n        text = res['messages'][-1].content\n        if '```json' in text: text = text.split('```json')[1].split('```')[0]\n        elif '```' in text: text = text.split('```')[1].split('```')[0]\n        ev = json.loads(text.strip())\n        return {\"is_passing\": ev.get('pass', False), \"critique\": ev.get('critique', 'Failed.')}\n    except: return {\"is_passing\": False, \"critique\": \"JSON Parse Error\"}\n\ndef route(state: ReflexionState):\n    if state['is_passing']: return \"save_node\"\n    if state['iterations'] >= 3: return END\n    return \"actor_node\"\n\ndef save_node(state: ReflexionState):\n    print(\"\\n\\ud83d\\udcbe [MEMORY] Saving to disk...\")\n    save_memory(state['task'], state['current_draft'], state['critique'])\n    return state"),
    ("markdown", "### Build & Run Graph"),
    ("code", "builder = StateGraph(ReflexionState)\nbuilder.add_node(\"actor_node\", actor_node)\nbuilder.add_node(\"evaluator_node\", evaluator_node)\nbuilder.add_node(\"save_node\", save_node)\nbuilder.add_edge(START, \"actor_node\")\nbuilder.add_edge(\"actor_node\", \"evaluator_node\")\nbuilder.add_conditional_edges(\"evaluator_node\", route)\nbuilder.add_edge(\"save_node\", END)\napp = builder.compile()\n\ntask = \"Apologize to a VIP client for a 2-week delay. Extremely polite, under 100 words.\"\nres = app.invoke({\"task\": task, \"current_draft\": \"\", \"critique\": \"\", \"is_passing\": False, \"iterations\": 0})\nprint(\"\\n\\u2728 Final Draft:\\n\", res['current_draft'])")
]
write_nb("session_4_learning/demo_4_reflexion.ipynb", cells_4)
