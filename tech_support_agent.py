import os
import sys
import sqlite3
import random
import string
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SUPPORT_DB_PATH", os.path.join(SCRIPT_DIR, "support_demo.db"))

DEMO_VPN_USERS = [
    ("Alice Smith", "active"),
    ("Bob Jones", "inactive"),
    ("Charlie Brown", "active"),
]

VALID_TICKET_STATUSES = ("open", "in_progress", "resolved", "closed", "pending")


def load_environment() -> None:
    """Load .env variables and normalize GEMINI_API_KEY to GOOGLE_API_KEY."""
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
    load_dotenv()
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


load_environment()


# -------------------------------------------------------------
# 0. Database Configuration & State Helpers
# -------------------------------------------------------------
def _connect_db(db_path: str | None = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or DB_PATH)


def init_database(db_path: str | None = None) -> None:
    """Create database tables if needed and seed initial demo VPN users."""
    conn = _connect_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        employee_name TEXT,
        issue_type TEXT,
        description TEXT,
        reported_time TEXT,
        status TEXT,
        updated_at TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vpn_user_access (
        employee_name TEXT PRIMARY KEY,
        status TEXT
    )
    """)
    cursor.execute("SELECT count(*) FROM vpn_user_access")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO vpn_user_access (employee_name, status) VALUES (?, ?)",
            DEMO_VPN_USERS,
        )
    conn.commit()
    conn.close()


def reset_database(db_path: str | None = None) -> None:
    """Clear tickets and restore the initial seeded VPN user access records."""
    init_database(db_path)
    conn = _connect_db(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets")
    cursor.execute("DELETE FROM vpn_user_access")
    cursor.executemany(
        "INSERT INTO vpn_user_access (employee_name, status) VALUES (?, ?)",
        DEMO_VPN_USERS,
    )
    conn.commit()
    conn.close()


def get_database_snapshot(db_path: str | None = None) -> dict[str, list[dict]]:
    """Return current tickets and VPN access rows for UI inspection."""
    init_database(db_path)
    conn = _connect_db(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT ticket_id, employee_name, issue_type, status, description, reported_time, updated_at
        FROM tickets
        ORDER BY reported_time DESC
        """
    )
    tickets = [dict(row) for row in cursor.fetchall()]

    cursor.execute(
        """
        SELECT employee_name, status
        FROM vpn_user_access
        ORDER BY employee_name ASC
        """
    )
    vpn_users = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {"tickets": tickets, "vpn_users": vpn_users}


# -------------------------------------------------------------
# 1. Action Tools: Tickets & Access Management
# -------------------------------------------------------------
@tool
def create_support_ticket(employee_name: str, issue_type: str, description: str) -> str:
    """Create a new support ticket when the knowledge base cannot resolve the
    employee's issue. Only call this after you have collected the employee's
    name, the issue type, and a description of the problem."""
    init_database()
    conn = _connect_db()
    cursor = conn.cursor()

    ticket_id = "T-" + "".join(random.choices(string.digits, k=6))
    reported_time = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        "INSERT INTO tickets (ticket_id, employee_name, issue_type, description, "
        "reported_time, status) VALUES (?, ?, ?, ?, ?, ?)",
        (ticket_id, employee_name, issue_type, description, reported_time, "open"),
    )
    conn.commit()
    conn.close()

    return (
        f"Support ticket {ticket_id} has been created for {employee_name}. "
        f"Issue type: {issue_type}. Status: open."
    )


@tool
def check_vpn_user(employee_name: str) -> str:
    """Check whether an employee currently has VPN access, and automatically
    reactivate them if their access is inactive. Only call this after a support
    ticket has been created and the employee has agreed to a status check."""
    init_database()
    conn = _connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status FROM vpn_user_access WHERE employee_name = ?",
        (employee_name,),
    )
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return (
            f"employee_name={employee_name}, status=inactive, "
            f"was_reactivated=False, vpn_access=False (employee not found in VPN system)"
        )

    status = row[0]
    was_reactivated = False
    if status == "inactive":
        cursor.execute(
            "UPDATE vpn_user_access SET status = ? WHERE employee_name = ?",
            ("active", employee_name),
        )
        conn.commit()
        status = "active"
        was_reactivated = True

    conn.close()
    return (
        f"employee_name={employee_name}, status={status}, "
        f"was_reactivated={was_reactivated}, vpn_access={status == 'active'}"
    )


@tool
def update_ticket(ticket_id: str, status: str, description: str) -> str:
    """Update an existing support ticket's status and append a short note to
    its description. status must be one of: open, in_progress, resolved,
    closed, pending."""
    if status.lower() not in VALID_TICKET_STATUSES:
        return (
            f"Invalid status '{status}'. "
            f"Must be one of: {', '.join(VALID_TICKET_STATUSES)}."
        )

    init_database()
    conn = _connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM tickets WHERE ticket_id = ?", (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return f"No ticket found with ID {ticket_id}."

    existing_description = row[0] or ""
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    appended_description = f"{existing_description}\n\n[UPDATE {updated_at}]: {description}"

    cursor.execute(
        "UPDATE tickets SET status = ?, description = ?, updated_at = ? WHERE ticket_id = ?",
        (status.lower(), appended_description, updated_at, ticket_id),
    )
    conn.commit()
    conn.close()

    return f"Ticket {ticket_id} updated - status: {status.lower()}, note added."


@tool
def provision_vpn_user(employee_name: str) -> str:
    """Provision VPN access for an employee who was not found in the VPN
    system. Only call this after the employee has agreed to have an account
    created."""
    init_database()
    conn = _connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status FROM vpn_user_access WHERE employee_name = ?",
        (employee_name,),
    )
    row = cursor.fetchone()

    if row is not None:
        cursor.execute(
            "UPDATE vpn_user_access SET status = ? WHERE employee_name = ?",
            ("active", employee_name),
        )
        conn.commit()
        conn.close()
        return f"employee_name={employee_name}, status=active, user_already_existed=True"

    cursor.execute(
        "INSERT INTO vpn_user_access (employee_name, status) VALUES (?, ?)",
        (employee_name, "active"),
    )
    conn.commit()
    conn.close()

    return f"employee_name={employee_name}, status=active, user_already_existed=False"


# -------------------------------------------------------------
# 2. Vector Retrieval & LangGraph Agent Factory
# -------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful IT support assistant for Acme Corp. You assist employees with VPN-related issues.

Follow this process in order:
1. Always search the VPN knowledge base first for setup, troubleshooting, or configuration questions.
2. If the knowledge base cannot resolve the issue, collect the employee's name, issue type, and a description, then create a support ticket. Never create ticket without asking the employee name first. Never invent, guess, or use a placeholder for the employee's name.
3. Immediately after creating a ticket, ask the employee if they'd like you to check their VPN account status to help resolve things faster.
4. If they agree, call check_vpn_user with their employee name:
   - status=active, was_reactivated=False: tell them access is active and ask to retry.
   - status=active, was_reactivated=True: tell them account was inactive and has been reactivated.
   - status=inactive, was_reactivated=False: ask if they want a new VPN account created.
5. If they agree to account creation, call provision_vpn_user with their name.
6. After telling them to try again, wait for their response:
   - If fixed, call update_ticket to close the ticket with a note.
   - If still not working, call update_ticket to keep ticket open with a note.
7. If they decline the status check, leave the ticket open for IT.

Whenever a tool returns a ticket ID or status, repeat that exact value back to the user.
Always be polite, concise, and stay focused on VPN-related topics only."""


def load_or_build_vectorstore(
    embeddings: GoogleGenerativeAIEmbeddings,
    base_dir: str = SCRIPT_DIR,
) -> FAISS:
    """Load an existing FAISS index or build one from vpn_user_guide.pdf."""
    faiss_dir = os.path.join(base_dir, "faiss_index")
    index_file = os.path.join(faiss_dir, "index.faiss")

    if not os.path.exists(index_file):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pdf_path = os.path.join(base_dir, "vpn_user_guide.pdf")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_dir)
        return vectorstore

    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)


def resolve_api_key(api_key: str | None = None) -> str:
    """Resolve Google Gemini API key from explicit input or environment."""
    resolved = (
        (api_key or "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not resolved:
        raise ValueError(
            "Missing GOOGLE_API_KEY. Set it in .env or provide a Google Gemini API key."
        )
    return resolved


def build_agent(
    api_key: str | None = None,
    model_name: str | None = None,
    db_path: str | None = None,
):
    """Build and compile the stateful LangGraph IT support agent."""
    resolved_key = resolve_api_key(api_key)
    init_database(db_path)

    resolved_model = model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    llm = ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=0,
        google_api_key=resolved_key,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=resolved_key,
    )
    vectorstore = load_or_build_vectorstore(embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    kb_tool = create_retriever_tool(
        retriever,
        name="search_vpn_knowledge_base",
        description=(
            "Search Acme Corp's internal VPN user guide for installation steps, "
            "troubleshooting, and configuration details."
        ),
    )

    tools = [
        kb_tool,
        create_support_ticket,
        check_vpn_user,
        update_ticket,
        provision_vpn_user,
    ]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile()


# -------------------------------------------------------------
# 3. Conversation & Tool-Trace Helpers
# -------------------------------------------------------------
def extract_text_content(content) -> str:
    """Normalize LangChain message content (string or structured parts) into text."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None:
        return ""
    return str(content)


def extract_tool_events(messages: list) -> list[dict]:
    """Pair AIMessage tool calls with their ToolMessage outputs for UI tracing."""
    events: list[dict] = []
    by_call_id: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                call_id = call.get("id") or f"call_{len(events)}"
                event = {
                    "id": call_id,
                    "name": call.get("name", "unknown_tool"),
                    "args": call.get("args", {}),
                    "output": None,
                }
                events.append(event)
                by_call_id[call_id] = event
        elif isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            output_text = extract_text_content(msg.content)
            if call_id and call_id in by_call_id:
                by_call_id[call_id]["output"] = output_text
            else:
                events.append(
                    {
                        "id": call_id or f"tool_{len(events)}",
                        "name": getattr(msg, "name", "tool"),
                        "args": {},
                        "output": output_text,
                    }
                )

    return events


def run_agent_turn(agent, conversation: list, user_input: str) -> dict:
    """Execute one user turn through the compiled LangGraph agent."""
    updated_input = list(conversation) + [{"role": "user", "content": user_input}]
    result = agent.invoke({"messages": updated_input})
    all_messages = result["messages"]
    turn_messages = all_messages[len(updated_input):]
    reply = extract_text_content(all_messages[-1].content) if all_messages else ""
    tool_events = extract_tool_events(turn_messages)

    return {
        "messages": all_messages,
        "reply": reply,
        "tool_events": tool_events,
    }


def main() -> None:
    try:
        agent = build_agent()
    except ValueError:
        print("\n" + "=" * 60)
        print("⚠️  MISSING GOOGLE_API_KEY")
        print("To run this agent 100% free with Google Gemini:")
        print("1. Get a free API key at: https://aistudio.google.com")
        print("2. Create a .env file in this directory with:")
        print("   GOOGLE_API_KEY=your_gemini_api_key_here")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("Autonomous Tech Support Agent (Powered by Google Gemini - Free Tier) online.")
    print("Type 'quit' or 'exit' to stop.\n")
    conversation = []
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                break
            if not user_input.strip():
                continue
            turn = run_agent_turn(agent, conversation, user_input)
            conversation = turn["messages"]
            print("Agent:", turn["reply"], "\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
