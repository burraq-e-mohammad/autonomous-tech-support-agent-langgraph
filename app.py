import importlib.util
import json
import os
import sys
import streamlit as st

from tech_support_agent import (
    build_agent,
    get_database_snapshot,
    load_environment,
    reset_database,
    run_agent_turn,
)

load_environment()

SUGGESTED_PROMPTS = [
    ("📘 VPN Setup Guide", "How do I install and connect to the Acme Corp VPN?"),
    (
        "🔧 Inactive User (Bob Jones)",
        "My VPN isn't working. My name is Bob Jones and I keep getting a connection denied error.",
    ),
    (
        "🆕 New User (Sarah Connor)",
        "Hi, I'm Sarah Connor and I need VPN access set up for my laptop.",
    ),
]

DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


def _init_session_state() -> None:
    """Initialize Streamlit session state keys on first load."""
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []
    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages = []
    if "tool_trace" not in st.session_state:
        st.session_state.tool_trace = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "turn_counter" not in st.session_state:
        st.session_state.turn_counter = 0


def _clear_chat_state() -> None:
    """Reset conversation history and tool execution trace."""
    st.session_state.ui_messages = []
    st.session_state.lc_messages = []
    st.session_state.tool_trace = []
    st.session_state.pending_prompt = None
    st.session_state.turn_counter = 0


@st.cache_resource(show_spinner="Initializing LangGraph agent & FAISS knowledge base...")
def _get_cached_agent(api_key: str, model_name: str):
    """Build and cache the compiled LangGraph agent per API key and model."""
    return build_agent(api_key=api_key, model_name=model_name)


def _render_sidebar(env_api_key: str, env_model: str) -> tuple[str, str]:
    """Render configuration, demo controls, and architecture reference."""
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
                padding-top: 1.25rem;
            }
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.8rem;
            }
            [data-testid="stSidebar"] hr {
                margin: 0.7rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("## 🛡️ Acme Corp IT Helpdesk")
        st.caption("LangGraph • Google Gemini • FAISS RAG • SQLite")
        st.divider()

        st.markdown("### ⚙️ Configuration")
        model_options = [env_model] + [m for m in DEFAULT_MODELS if m != env_model]
        selected_model = st.selectbox("Gemini Model", options=model_options, index=0)

        if env_api_key:
            st.success("API key loaded from environment (`.env`)", icon="✅")
            with st.expander("Override API Key"):
                override_key = st.text_input(
                    "Custom Google Gemini API Key",
                    type="password",
                    placeholder="Leave empty to use .env key",
                )
            active_api_key = override_key.strip() or env_api_key
        else:
            st.warning("No `GOOGLE_API_KEY` found in environment.", icon="⚠️")
            active_api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                placeholder="Paste your free AI Studio key...",
            ).strip()
            st.caption("Get a free key at [Google AI Studio](https://aistudio.google.com).")

        st.divider()
        st.markdown("### 🧪 Demo Controls")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear Chat", width="stretch"):
                _clear_chat_state()
                st.toast("Conversation cleared.", icon="💬")
        with col_b:
            if st.button("Reset DB", width="stretch"):
                reset_database()
                st.toast("Demo database reset to initial state.", icon="🗄️")

        st.divider()
        st.markdown("### 🧠 LangGraph Architecture")
        st.markdown(
            """
            **State Machine:** `START → chatbot ⇄ tools → END`

            **Available Tools:**
            - `search_vpn_knowledge_base` (FAISS RAG)
            - `create_support_ticket` (SQLite)
            - `check_vpn_user` (SQLite)
            - `provision_vpn_user` (SQLite)
            - `update_ticket` (SQLite)
            """
        )
        st.caption(f"Python `{sys.version.split()[0]}` (`{sys.executable}`)")

    return active_api_key, selected_model


def _handle_prompt(prompt: str, active_api_key: str, selected_model: str) -> None:
    """Invoke the LangGraph agent for a single user turn and store trace events."""
    st.session_state.ui_messages.append({"role": "user", "content": prompt})

    try:
        agent = _get_cached_agent(active_api_key, selected_model)
        turn_result = run_agent_turn(agent, st.session_state.lc_messages, prompt)
        st.session_state.lc_messages = turn_result["messages"]
        st.session_state.turn_counter += 1

        tool_events = turn_result["tool_events"]
        st.session_state.ui_messages.append(
            {
                "role": "assistant",
                "content": turn_result["reply"],
                "tool_events": tool_events,
            }
        )
        for event in tool_events:
            st.session_state.tool_trace.append(
                {
                    "turn": st.session_state.turn_counter,
                    **event,
                }
            )
    except Exception as exc:
        st.session_state.ui_messages.append(
            {
                "role": "assistant",
                "content": (
                    f"⚠️ Error communicating with the agent: `{exc}`\n\n"
                    f"Active Python interpreter: `{sys.executable}`"
                ),
                "tool_events": [],
                "is_error": True,
            }
        )


def _render_chat_panel(active_api_key: str, selected_model: str) -> None:
    """Render the interactive chat column, starter prompts, and input box."""
    st.markdown("#### 💬 Helpdesk Conversation")

    if not st.session_state.ui_messages:
        prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
        for idx, (label, prompt_text) in enumerate(SUGGESTED_PROMPTS):
            with prompt_cols[idx]:
                if st.button(
                    label,
                    key=f"starter_{idx}",
                    width="stretch",
                    disabled=not bool(active_api_key),
                    help=prompt_text,
                ):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()

    chat_box = st.container(height=620, border=True, autoscroll=True)
    with chat_box:
        if not st.session_state.ui_messages:
            with st.chat_message("assistant"):
                st.markdown(
                    "Hi! I'm Acme Corp's autonomous VPN support assistant. "
                    "Ask a setup or troubleshooting question from the VPN guide, "
                    "or share your issue and name to open a ticket and verify your access."
                )

        for msg in st.session_state.ui_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                tool_events = msg.get("tool_events") or []
                if tool_events:
                    tool_names = ", ".join(f"`{e['name']}`" for e in tool_events)
                    st.caption(f"🔧 Tools executed: {tool_names}")

        pending = st.session_state.pending_prompt
        if pending and active_api_key:
            st.session_state.pending_prompt = None
            with st.chat_message("user"):
                st.markdown(pending)
            with st.chat_message("assistant"):
                with st.spinner("Reasoning & executing LangGraph tools..."):
                    _handle_prompt(pending, active_api_key, selected_model)
            st.rerun()

    user_input = st.chat_input(
        "Ask a VPN question or report an issue...",
        disabled=not bool(active_api_key),
    )
    if user_input and active_api_key:
        st.session_state.pending_prompt = user_input
        st.rerun()
    elif not active_api_key:
        st.info("Enter a Google Gemini API key in the sidebar to start chatting.")


def _format_vpn_status(status: str) -> str:
    return f"🟢 {status}" if status == "active" else f"🔴 {status}"


def _format_ticket_status(status: str) -> str:
    icons = {
        "open": "🟡 open",
        "in_progress": "🔵 in_progress",
        "pending": "🟠 pending",
        "resolved": "🟢 resolved",
        "closed": "⚫ closed",
    }
    return icons.get(status, status)


def _render_state_panel() -> None:
    """Render live SQLite database tables and LangGraph tool execution trace."""
    st.markdown("#### 🔍 Live Backend State & Agent Trace")
    snapshot = get_database_snapshot()
    tickets = snapshot["tickets"]
    vpn_users = snapshot["vpn_users"]

    open_tickets = sum(
        1 for t in tickets if t.get("status") in ("open", "in_progress", "pending")
    )
    active_users = sum(1 for u in vpn_users if u.get("status") == "active")

    m1, m2, m3 = st.columns(3)
    m1.metric("Open Tickets", open_tickets, f"{len(tickets)} total")
    m2.metric("Active VPN Users", f"{active_users}/{len(vpn_users)}")
    m3.metric("Tool Calls", len(st.session_state.tool_trace))

    st.markdown("**VPN User Access (`vpn_user_access`)**")
    formatted_users = [
        {
            "Employee": u["employee_name"],
            "Status": _format_vpn_status(u["status"]),
        }
        for u in vpn_users
    ]
    st.dataframe(formatted_users, width="stretch", height=145, hide_index=True)

    st.markdown("**Support Tickets (`tickets`)**")
    if tickets:
        formatted_tickets = [
            {
                "Ticket ID": t["ticket_id"],
                "Employee": t["employee_name"],
                "Issue": t["issue_type"],
                "Status": _format_ticket_status(t["status"]),
                "Description": t["description"],
                "Updated": t["updated_at"] or "—",
            }
            for t in tickets
        ]
        st.dataframe(formatted_tickets, width="stretch", height=165, hide_index=True)
    else:
        st.info("No tickets created yet. Ask the agent to open one if troubleshooting fails.")

    st.markdown("**LangGraph Tool Execution Trace**")
    trace_box = st.container(height=280, border=True)
    with trace_box:
        if not st.session_state.tool_trace:
            st.caption("Tool invocations, arguments, and outputs will appear here in real time.")
        else:
            for idx, event in enumerate(reversed(st.session_state.tool_trace), start=1):
                label = f"Turn {event['turn']} • `{event['name']}`"
                with st.expander(label, expanded=(idx == 1)):
                    st.markdown("**Arguments**")
                    st.code(json.dumps(event.get("args", {}), indent=2), language="json")
                    st.markdown("**Tool Output**")
                    st.text(event.get("output") or "(no output)")


def main() -> None:
    st.set_page_config(
        page_title="Autonomous IT Support Agent",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    env_api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    env_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

    active_api_key, selected_model = _render_sidebar(env_api_key, env_model)

    st.markdown("## 🛡️ Autonomous IT Tech Support Agent")
    st.caption(
        "Troubleshoots corporate VPN issues with FAISS RAG over `vpn_user_guide.pdf` "
        "and executes live SQLite ticket & access operations via LangGraph."
    )

    if importlib.util.find_spec("faiss") is None:
        st.error(
            f"`faiss` is not installed in the active Python environment (`{sys.executable}`). "
            "Launch Streamlit with the Python interpreter where dependencies were installed "
            "(`python -m streamlit run app.py`), or install them into this interpreter with: "
            f"`\"{sys.executable}\" -m pip install -r requirements.txt`."
        )

    chat_col, state_col = st.columns([1.3, 1.0], gap="large")
    with chat_col:
        _render_chat_panel(active_api_key, selected_model)
    with state_col:
        _render_state_panel()


if __name__ == "__main__":
    main()

