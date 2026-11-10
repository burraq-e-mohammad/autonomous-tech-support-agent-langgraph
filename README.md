# Autonomous IT Tech Support Agent with LangGraph & Multi-Tool RAG

In this project, I built an autonomous IT tech support agent using **LangGraph**, **LangChain**, and **Google Gemini**. It troubleshoots corporate VPN issues using dense vector retrieval (RAG) over internal PDF documentation and executes deterministic actions against a local SQLite database through a stateful LangGraph workflow.

I configured this project to run on the **Google Gemini** free tier using `ChatGoogleGenerativeAI` (configurable via `GEMINI_MODEL`, defaulting to `gemini-1.5-flash`) and `models/gemini-embedding-001`.

---

## Key Capabilities I Implemented
- **Multi-Tool Agentic Reasoning:** The agent decides whether to answer questions from the knowledge base or invoke operational tools to create tickets, check VPN account permissions, reactivate inactive accounts, provision new users, or update ticket statuses.
- **Dense Vector Search (RAG):** Uses FAISS and Google Gemini embeddings (`models/gemini-embedding-001`) to index and retrieve answers from `vpn_user_guide.pdf`.
- **Deterministic Action Layer:** Executes structured SQLite CRUD operations (`support_demo.db`) for support tickets and VPN user access records.
- **LangGraph State Machine:** Cyclical graph using `MessagesState`, a `chatbot` reasoning node, dynamic `ToolNode`, and `tools_condition` routing.
- **Free-Tier Ready:** Powered by Google AI Studio (`GOOGLE_API_KEY` / `GEMINI_API_KEY`) with no paid API subscription required.

---

## Project Structure
```
01-langgraph-autonomous-tech-support-agent/
│
├── Autonomous_Tech_Support_Agent_LangGraph.ipynb   # Master engineering notebook
├── tech_support_agent.py                           # Standalone CLI agent executable
├── build_knowledge_base.py                         # FAISS vector store build script
├── vpn_user_guide.pdf                              # Corporate VPN documentation corpus
├── requirements.txt                                # Dependencies (langgraph, langchain-google-genai, faiss-cpu)
├── .env.example                                    # Environment variable template
├── .gitignore                                      # Ignored local secrets, cache, and runtime artifacts
└── README.md                                       # Project documentation
```

---

## Getting Started

### 1. Environment Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Free Google Gemini API Key
1. Get a free API key from [Google AI Studio](https://aistudio.google.com).
2. Create a `.env` file in this directory (or copy `.env.example`):
```bash
GOOGLE_API_KEY="your-gemini-api-key"
```

### 3. Build Knowledge Base & Launch
- **Run the Standalone CLI Agent:**
  ```bash
  python tech_support_agent.py
  ```
- **Or Run the Master Notebook:**
  ```bash
  jupyter lab Autonomous_Tech_Support_Agent_LangGraph.ipynb
  ```
