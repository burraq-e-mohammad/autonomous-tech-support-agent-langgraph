import os
import sys
import sqlite3
import random
import string
from datetime import datetime, timezone
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))
load_dotenv(os.path.join(script_dir, ",env"))
load_dotenv()

# Normalize Google API key environment variables
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
    print("\n" + "=" * 60)
    print("⚠️  MISSING GOOGLE_API_KEY")
    print("To run this agent 100% free with Google Gemini:")
    print("1. Get a free API key at: https://aistudio.google.com")
    print("2. Create a .env file in this directory with:")
    print("   GOOGLE_API_KEY=your_gemini_api_key_here")
    print("=" * 60 + "\n")
    sys.exit(1)

# -------------------------------------------------------------
# 0. Database Configuration & Automatic Initialization
# -------------------------------------------------------------
DB_PATH = os.environ.get("SUPPORT_DB_PATH", os.path.join(os.path.dirname(__file__), "support_demo.db"))

def init_database(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
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
    # Seed sample corporate accounts if empty
    cursor.execute("SELECT count(*) FROM vpn_user_access")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO vpn_user_access (employee_name, status) VALUES (?, ?)",
            [("Alice Smith", "active"), ("Bob Jones", "inactive"), ("Charlie Brown", "active")]
        )
    conn.commit()
    conn.close()

init_database()

# -------------------------------------------------------------
# 1. LLM Initialization (Google Gemini - Free Tier)
# -------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)

# -------------------------------------------------------------
# 2. Knowledge Base & Vector Retrieval Tool (Gemini Embeddings + FAISS)
# -------------------------------------------------------------
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool

embedding_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
faiss_dir = os.path.join(os.path.dirname(__file__), "faiss_index")

if not os.path.exists(faiss_dir):
    print("Building FAISS index from vpn_user_guide.pdf with Gemini embeddings...")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    loader = PyPDFLoader(os.path.join(os.path.dirname(__file__), "vpn_user_guide.pdf"))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_dir)
else:
    vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
kb_tool = create_retriever_tool(
    retriever,
    name="search_vpn_knowledge_base",
    description=(
        "Search Acme Corp's internal VPN user guide for installation steps, "
        "troubleshooting, and configuration details."
    ),
)

# -------------------------------------------------------------
# 3. Action Tools: Tickets & Access Management
# -------------------------------------------------------------
from langchain_core.tools import tool

@tool
def create_support_ticket(employee_name: str, issue_type: str, description: str) -> str:
    """Create a new support ticket when the knowledge base cannot resolve the
    employee's issue. Only call this after you have collected the employee's
    name, the issue type, and a description of the problem."""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    valid_statuses = ["open", "in_progress", "resolved", "closed", "pending"]
    if status.lower() not in valid_statuses:
        return f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}."

    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
# 4. Tool Binding & LangGraph Architecture
# -------------------------------------------------------------
tools = [kb_tool, create_support_ticket, check_vpn_user, update_ticket, provision_vpn_user]
llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

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

agent = graph_builder.compile()

if __name__ == "__main__":
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
            conversation.append({"role": "user", "content": user_input})
            result = agent.invoke({"messages": conversation})
            conversation = result["messages"]
            agent_response = conversation[-1].content
            if isinstance(agent_response, list):
                agent_response = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in agent_response
                )
            print("Agent:", agent_response, "\n")
        except (KeyboardInterrupt, EOFError):
            break
