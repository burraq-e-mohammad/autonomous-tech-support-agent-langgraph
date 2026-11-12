import os
import tempfile
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import tech_support_agent as agent_mod


class _StubAgent:
    """Minimal runnable double returning predetermined LangGraph state output."""

    def __init__(self, new_messages):
        self._new_messages = new_messages

    def invoke(self, payload):
        return {"messages": list(payload["messages"]) + list(self._new_messages)}


class TestDatabaseAndTools(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._orig_db = agent_mod.DB_PATH
        agent_mod.DB_PATH = os.path.join(self._tmp_dir.name, "test_support.db")
        agent_mod.init_database()

    def tearDown(self):
        agent_mod.DB_PATH = self._orig_db
        self._tmp_dir.cleanup()

    def test_seed_and_snapshot(self):
        snapshot = agent_mod.get_database_snapshot()
        users = {u["employee_name"]: u["status"] for u in snapshot["vpn_users"]}
        self.assertEqual(
            users,
            {
                "Alice Smith": "active",
                "Bob Jones": "inactive",
                "Charlie Brown": "active",
            },
        )
        self.assertEqual(snapshot["tickets"], [])

    def test_ticket_lifecycle_and_inactive_reactivation(self):
        created = agent_mod.create_support_ticket.invoke(
            {
                "employee_name": "Bob Jones",
                "issue_type": "Authentication Failure",
                "description": "Connection denied",
            }
        )
        self.assertIn("T-", created)

        snapshot = agent_mod.get_database_snapshot()
        self.assertEqual(len(snapshot["tickets"]), 1)
        ticket_id = snapshot["tickets"][0]["ticket_id"]
        self.assertEqual(snapshot["tickets"][0]["status"], "open")

        check_res = agent_mod.check_vpn_user.invoke({"employee_name": "Bob Jones"})
        self.assertIn("status=active", check_res)
        self.assertIn("was_reactivated=True", check_res)

        update_res = agent_mod.update_ticket.invoke(
            {
                "ticket_id": ticket_id,
                "status": "resolved",
                "description": "Reactivated VPN account",
            }
        )
        self.assertIn("status: resolved", update_res)

        updated_snapshot = agent_mod.get_database_snapshot()
        self.assertEqual(updated_snapshot["tickets"][0]["status"], "resolved")
        self.assertIn("Reactivated VPN account", updated_snapshot["tickets"][0]["description"])

    def test_unknown_user_and_provisioning(self):
        missing = agent_mod.check_vpn_user.invoke({"employee_name": "Sarah Connor"})
        self.assertIn("employee not found", missing)

        provisioned = agent_mod.provision_vpn_user.invoke({"employee_name": "Sarah Connor"})
        self.assertIn("user_already_existed=False", provisioned)

        snapshot = agent_mod.get_database_snapshot()
        users = {u["employee_name"]: u["status"] for u in snapshot["vpn_users"]}
        self.assertEqual(users["Sarah Connor"], "active")

    def test_invalid_ticket_status_rejected(self):
        res = agent_mod.update_ticket.invoke(
            {
                "ticket_id": "T-000000",
                "status": "archived",
                "description": "Invalid",
            }
        )
        self.assertIn("Invalid status", res)

    def test_reset_database_restores_seed_state(self):
        agent_mod.create_support_ticket.invoke(
            {
                "employee_name": "Alice Smith",
                "issue_type": "Timeout",
                "description": "Drops frequently",
            }
        )
        agent_mod.provision_vpn_user.invoke({"employee_name": "Sarah Connor"})
        agent_mod.check_vpn_user.invoke({"employee_name": "Bob Jones"})

        agent_mod.reset_database()
        snapshot = agent_mod.get_database_snapshot()
        self.assertEqual(snapshot["tickets"], [])
        users = {u["employee_name"]: u["status"] for u in snapshot["vpn_users"]}
        self.assertEqual(len(users), 3)
        self.assertEqual(users["Bob Jones"], "inactive")


class TestAgentHelpers(unittest.TestCase):
    def test_extract_text_content_handles_parts_and_strings(self):
        self.assertEqual(agent_mod.extract_text_content("hello"), "hello")
        self.assertEqual(agent_mod.extract_text_content(None), "")
        self.assertEqual(
            agent_mod.extract_text_content(
                [{"type": "text", "text": "Part 1 "}, {"type": "text", "text": "Part 2"}]
            ),
            "Part 1 Part 2",
        )

    def test_run_agent_turn_extracts_tool_events(self):
        prior = [HumanMessage(content="Hi"), AIMessage(content="Hello!")]
        stub = _StubAgent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "check_vpn_user",
                            "args": {"employee_name": "Bob Jones"},
                        }
                    ],
                ),
                ToolMessage(
                    content="employee_name=Bob Jones, status=active, was_reactivated=True",
                    tool_call_id="call_1",
                    name="check_vpn_user",
                ),
                AIMessage(content=[{"type": "text", "text": "Your VPN account is active now."}]),
            ]
        )

        turn = agent_mod.run_agent_turn(stub, prior, "Check Bob Jones")
        self.assertEqual(turn["reply"], "Your VPN account is active now.")
        self.assertEqual(len(turn["tool_events"]), 1)
        self.assertEqual(turn["tool_events"][0]["name"], "check_vpn_user")
        self.assertEqual(turn["tool_events"][0]["args"], {"employee_name": "Bob Jones"})
        self.assertIn("was_reactivated=True", turn["tool_events"][0]["output"])

    def test_resolve_api_key_raises_when_missing(self):
        orig_google = os.environ.pop("GOOGLE_API_KEY", None)
        orig_gemini = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                agent_mod.resolve_api_key(None)
            self.assertEqual(agent_mod.resolve_api_key(" test-key "), "test-key")
        finally:
            if orig_google is not None:
                os.environ["GOOGLE_API_KEY"] = orig_google
            if orig_gemini is not None:
                os.environ["GEMINI_API_KEY"] = orig_gemini


if __name__ == "__main__":
    unittest.main()
