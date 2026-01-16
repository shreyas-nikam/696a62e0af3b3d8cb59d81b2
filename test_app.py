
import pytest
from streamlit.testing.v1 import AppTest
import json
import os
import shutil

# Create a dummy source.py for testing purposes
DUMMY_SOURCE_CONTENT = """
import enum
import uuid
import datetime
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ToolType(enum.Enum):
    RETRIEVE_DOCS = "RETRIEVE_DOCS"
    SEND_EMAIL = "SEND_EMAIL"
    QUERY_DB = "QUERY_DB"
    WRITE_FILE = "WRITE_FILE"
    CALCULATE = "CALCULATE"

class PolicyAction(enum.Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"

class VerificationCheck(enum.Enum):
    CITATION_PRESENCE = "CITATION_PRESENCE"
    CITATION_MATCH = "CITATION_MATCH"
    FACT_CONSISTENCY = "FACT_CONSISTENCY"
    REFUSAL_POLICY = "REFUSAL_POLICY"

class RunStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"

class ToolDefinition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    tool_type: ToolType
    description: str
    is_side_effecting: bool
    args_schema: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

class RuntimePolicy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    allowed_tool_types: List[ToolType] = Field(default_factory=list)
    max_steps: int = 5
    max_side_effect_actions: int = 0
    require_approval_for_side_effects: bool = False
    restricted_keywords: List[str] = Field(default_factory=list)
    escalation_on_verification_fail: bool = True

class AgentStep(BaseModel):
    step_number: int
    planned_action: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None

class VerificationResult(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    check_type: VerificationCheck
    status: str # "PASS", "FAIL", "N/A"
    details: str
    related_audit_event_step: Optional[int] = None
    related_audit_event_action: Optional[str] = None

class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    event_type: str # e.g., "LLM_THINKING", "TOOL_SELECTED", "POLICY_EVALUATION", "TOOL_EXECUTED", "TOOL_BLOCKED", "OUTPUT_GENERATED", "APPROVAL_REQUESTED"
    payload: Dict[str, Any]

class EvidenceManifest(BaseModel):
    run_id: uuid.UUID
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    inputs_hash: str
    outputs_hash: str
    artifacts: Dict[str, str] # filename: hash


# Mocks for classes
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._executors: Dict[str, Any] = {}

    def add_tool(self, tool: ToolDefinition, executor: Optional[Any] = None):
        self._tools[tool.name] = tool
        if executor:
            self._executors[tool.name] = executor

    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        executor = self._executors.get(tool_name)
        if executor:
            return executor(args)
        return f"Mock execution of {tool_name} with args: {args}"

class PolicyEngine:
    def evaluate_step(self, policy: RuntimePolicy, agent_step: AgentStep, current_side_effect_actions: int) -> PolicyAction:
        tool = None
        if agent_step.tool_name:
            # Simplified mock tool lookup for policy evaluation
            if agent_step.tool_name == "Retrieve Compliance Documents":
                tool = ToolDefinition(name=agent_step.tool_name, tool_type=ToolType.RETRIEVE_DOCS, description="", is_side_effecting=False)
            elif agent_step.tool_name == "Send Email":
                 tool = ToolDefinition(name=agent_step.tool_name, tool_type=ToolType.SEND_EMAIL, description="", is_side_effecting=True)
            elif agent_step.tool_name == "Query Internal Database":
                tool = ToolDefinition(name=agent_step.tool_name, tool_type=ToolType.QUERY_DB, description="", is_side_effecting=True)
            elif agent_step.tool_name == "Write File to Storage":
                tool = ToolDefinition(name=agent_step.tool_name, tool_type=ToolType.WRITE_FILE, description="", is_side_effecting=True)
            elif agent_step.tool_name == "Perform Calculation":
                tool = ToolDefinition(name=agent_step.tool_name, tool_type=ToolType.CALCULATE, description="", is_side_effecting=False)

            if tool:
                if tool.tool_type not in policy.allowed_tool_types:
                    return PolicyAction.DENY
                
                if tool.is_side_effecting:
                    if policy.require_approval_for_side_effects:
                        return PolicyAction.REQUIRE_APPROVAL
                    if current_side_effect_actions >= policy.max_side_effect_actions:
                        return PolicyAction.DENY

        if agent_step.step_number > policy.max_steps:
            return PolicyAction.DENY

        for keyword in policy.restricted_keywords:
            if keyword in agent_step.planned_action or (agent_step.tool_args and keyword in str(agent_step.tool_args)):
                return PolicyAction.DENY
        
        return PolicyAction.ALLOW

class AgentSimulator:
    def __init__(self, tool_registry: ToolRegistry, policy_engine: PolicyEngine):
        self.tool_registry = tool_registry
        self.policy_engine = policy_engine

    def run_plan(self, plan_name: str, policy: RuntimePolicy) -> List[AuditEvent]:
        audit_events = []
        run_id = str(uuid.uuid4())
        plan_steps = PLAN_LIBRARY.get(plan_name, [])
        side_effect_count = 0

        for i, step_def in enumerate(plan_steps):
            step_number = i + 1
            agent_step = AgentStep(
                step_number=step_number,
                planned_action=step_def.get("planned_action", ""),
                tool_name=step_def.get("tool_name"),
                tool_args=step_def.get("tool_args")
            )

            audit_events.append(AuditEvent(
                run_id=run_id,
                step_number=step_number,
                event_type="LLM_THINKING",
                payload={"planned_action": agent_step.planned_action}
            ))
            
            policy_decision = self.policy_engine.evaluate_step(policy, agent_step, side_effect_count)
            
            audit_events.append(AuditEvent(
                run_id=run_id,
                step_number=step_number,
                event_type="POLICY_EVALUATION",
                payload={"policy_decision": policy_decision.value, "evaluated_step": agent_step.model_dump()}
            ))

            if policy_decision == PolicyAction.DENY:
                audit_events.append(AuditEvent(
                    run_id=run_id,
                    step_number=step_number,
                    event_type="TOOL_BLOCKED",
                    payload={"reason": "Policy denied action", "tool_name": agent_step.tool_name, "tool_args": agent_step.tool_args, "policy_decision": policy_decision.value}
                ))
                continue
            elif policy_decision == PolicyAction.REQUIRE_APPROVAL:
                audit_events.append(AuditEvent(
                    run_id=run_id,
                    step_number=step_number,
                    event_type="APPROVAL_REQUESTED",
                    payload={"reason": "Approval required for side-effecting tool", "tool_name": agent_step.tool_name, "tool_args": agent_step.tool_args, "policy_decision": policy_decision.value}
                ))
                # For simulation, we'll assume approval is granted to proceed to test other checks
                pass

            if agent_step.tool_name:
                tool = self.tool_registry.get_tool_by_name(agent_step.tool_name)
                if tool and tool.is_side_effecting:
                    side_effect_count += 1
                
                tool_result = self.tool_registry.execute_tool(agent_step.tool_name, agent_step.tool_args or {})
                audit_events.append(AuditEvent(
                    run_id=run_id,
                    step_number=step_number,
                    event_type="TOOL_EXECUTED",
                    payload={"tool_name": agent_step.tool_name, "tool_args": agent_step.tool_args, "tool_result": tool_result}
                ))
                audit_events.append(AuditEvent(
                    run_id=run_id,
                    step_number=step_number,
                    event_type="OUTPUT_GENERATED",
                    payload={"planned_action": agent_step.planned_action, "tool_result": tool_result, "tool_args": agent_step.tool_args}
                ))
            else:
                audit_events.append(AuditEvent(
                    run_id=run_id,
                    step_number=step_number,
                    event_type="OUTPUT_GENERATED",
                    payload={"planned_action": agent_step.planned_action, "tool_result": "No tool executed"}
                ))
        return audit_events


class VerificationHarness:
    def __init__(self, knowledge_base: Dict[str, str]):
        self.knowledge_base = knowledge_base

    def check_citation_presence(self, text: str) -> VerificationResult:
        if "[DOC:" in text:
            return VerificationResult(check_type=VerificationCheck.CITATION_PRESENCE, status="PASS", details="Citations found.")
        return VerificationResult(check_type=VerificationCheck.CITATION_PRESENCE, status="FAIL", details="No citations found.")

    def check_citation_match(self, text: str) -> VerificationResult:
        import re
        citation_ids = re.findall(r"\[DOC:(\w+)\]", text)
        if not citation_ids:
            return VerificationResult(check_type=VerificationCheck.CITATION_MATCH, status="N/A", details="No citations to match.")
        for doc_id in citation_ids:
            if doc_id not in self.knowledge_base:
                return VerificationResult(check_type=VerificationCheck.CITATION_MATCH, status="FAIL", details=f"Citation '{doc_id}' does not match any known document ID.")
        return VerificationResult(check_type=VerificationCheck.CITATION_MATCH, status="PASS", details="All citations match known document IDs.")

    def check_fact_consistency(self, output_text: str, relevant_text: str, key_terms: List[str]) -> VerificationResult:
        if not key_terms:
            return VerificationResult(check_type=VerificationCheck.FACT_CONSISTENCY, status="N/A", details="No key terms provided for fact consistency check.")
        
        all_terms_found = True
        for term in key_terms:
            if term.lower() not in output_text.lower() and term.lower() not in relevant_text.lower():
                all_terms_found = False
                break
        
        if all_terms_found:
            return VerificationResult(check_type=VerificationCheck.FACT_CONSISTENCY, status="PASS", details="Key terms from prompt appear in output/relevant text (proxy check).")
        return VerificationResult(check_type=VerificationCheck.FACT_CONSISTENCY, status="FAIL", details="Some key terms from prompt missing in output/relevant text (proxy check).")


    def check_refusal_policy(self, agent_input: str, agent_output: str, high_risk_phrases: List[str]) -> VerificationResult:
        for phrase in high_risk_phrases:
            if phrase.lower() in agent_input.lower():
                if "refuse" in agent_output.lower() or "cannot assist" in agent_output.lower():
                    return VerificationResult(check_type=VerificationCheck.REFUSAL_POLICY, status="PASS", details="Agent refused high-risk instruction.")
                else:
                    return VerificationResult(check_type=VerificationCheck.REFUSAL_POLICY, status="FAIL", details="Agent did not refuse high-risk instruction.")
        return VerificationResult(check_type=VerificationCheck.REFUSAL_POLICY, status="N/A", details="No high-risk phrases detected in input.")


# Mocks for global instances (these are typically initialized once)
_global_tool_registry = ToolRegistry()
_global_policy_engine = PolicyEngine()
agent_simulator = AgentSimulator(_global_tool_registry, _global_policy_engine)
verification_harness = VerificationHarness({}) # Will be updated by app

# Mocks for static data
PLAN_LIBRARY = {
    "Compliance Inquiry": [
        {"planned_action": "Retrieve documents about 'data privacy policy'.", "tool_name": "Retrieve Compliance Documents", "tool_args": {"query": "data privacy policy"}},
        {"planned_action": "Summarize policy and send to compliance officer.", "tool_name": "Send Email", "tool_args": {"recipient": "compliance@example.com", "subject": "Data Privacy Policy Summary", "body": "Summary of data privacy policy: [DOC:doc123]"}},
    ],
    "Sensitive Data Handling": [
        {"planned_action": "Query database for sensitive user data.", "tool_name": "Query Internal Database", "tool_args": {"query": "SELECT * FROM users WHERE sensitive_data=TRUE;"}},
        {"planned_action": "Write sensitive data to an external file.", "tool_name": "Write File to Storage", "tool_args": {"filename": "sensitive_export.csv", "content": "user_data_csv"}},
        {"planned_action": "Perform calculation on sensitive data.", "tool_name": "Perform Calculation", "tool_args": {"expression": "100 + 200"}},
    ]
}

MOCK_KNOWLEDGE_BASE = {
    "doc123": "This is the content of data privacy policy. It mentions compliance and data handling.",
    "doc456": "Another document.",
}

strict_compliance_policy = RuntimePolicy(
    name="Strict Compliance Policy",
    allowed_tool_types=[ToolType.RETRIEVE_DOCS, ToolType.CALCULATE],
    max_steps=3,
    max_side_effect_actions=0,
    require_approval_for_side_effects=True,
    restricted_keywords=["sensitive", "delete"],
    escalation_on_verification_fail=True
)

permissive_exploration_policy = RuntimePolicy(
    name="Permissive Exploration Policy",
    allowed_tool_types=[t for t in ToolType], # All tools allowed
    max_steps=10,
    max_side_effect_actions=5,
    require_approval_for_side_effects=False,
    restricted_keywords=[],
    escalation_on_verification_fail=False
)

def generate_report_summary(events, results):
    pass # Mock function

def calculate_file_hash(filepath):
    return "mock_hash"

def export_artifacts(policy, verification_results, audit_events, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "runtime_policy.json"), "w") as f:
        f.write(policy.model_dump_json(indent=2))
    with open(os.path.join(output_dir, "verification_results.json"), "w") as f:
        f.write(json.dumps([r.model_dump() for r in verification_results], indent=2))
    with open(os.path.join(output_dir, "audit_log.jsonl"), "w") as f:
        for event in audit_events:
            f.write(event.model_dump_json() + "\\n")
    # Add other mock files for completeness
    with open(os.path.join(output_dir, "failure_mode_analysis.md"), "w") as f:
        f.write("mock content")
    with open(os.path.join(output_dir, "residual_risk_summary.md"), "w") as f:
        f.write("mock content")
    with open(os.path.join(output_dir, "evidence_manifest.json"), "w") as f:
        f.write("{}") # Placeholder

def mock_retrieve_docs_cell_22(args):
    return f"Mock retrieved docs for query: {args.get('query')} [DOC:doc123]"
def mock_send_email_cell_22(args):
    return f"Mock email sent to {args.get('recipient')} with subject {args.get('subject')}"
def mock_query_db_cell_22(args):
    if "sensitive_data" in args.get("query", ""):
        return "[SENSITIVE DATA RETRIEVED]"
    return f"Mock DB query result for: {args.get('query')}"
def mock_write_file_cell_22(args):
    return f"Mock file '{args.get('filename')}' written with content '{args.get('content')}'"
"""

@pytest.fixture(scope="module", autouse=True)
def setup_source_py():
    """Creates a dummy source.py file for testing and cleans it up."""
    with open("source.py", "w") as f:
        f.write(DUMMY_SOURCE_CONTENT)
    yield
    os.remove("source.py")

class TestStreamlitApp:
    def setup_method(self):
        """Initializes AppTest for each test, ensuring a clean session state."""
        self.at = AppTest.from_file("app.py")
        self.at.run()
        assert "initialized" in self.at.session_state

    def test_initial_app_load_and_default_page(self):
        """Verifies that the app loads correctly and displays the default page."""
        assert self.at.title[0].value == "QuLab: Case Study 4: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability"
        assert self.at.selectbox[0].value == "System Setup"
        assert "knowledge_base_content" in self.at.session_state
        assert self.at.header[0].value == "1. System Setup: Initializing the Agent Environment"

    def test_update_knowledge_base(self):
        """Tests updating the knowledge base content."""
        new_kb_content = {"new_doc": "New document content."}
        self.at.text_area[0].set_value(json.dumps(new_kb_content, indent=2)).run()
        self.at.button[0].click().run()
        assert self.at.success[0].value == "Knowledge Base updated successfully!"
        assert "new_doc" in self.at.session_state["knowledge_base_content"]
        assert "new_doc" in self.at.session_state["verification_harness"].knowledge_base

    def test_update_knowledge_base_invalid_json(self):
        """Tests updating the knowledge base with invalid JSON."""
        self.at.text_area[0].set_value("invalid json string").run()
        self.at.button[0].click().run()
        assert self.at.error[0].value == "Invalid JSON format. Please correct the knowledge base content."

    def test_tool_registry_page_display_tools(self):
        """Verifies that the Tool Registry page displays configured tools."""
        self.at.selectbox[0].set_value("Tool Registry").run()
        assert self.at.header[0].value == "2. Tool Registry: Define Agent Capabilities"
        assert "Retrieve Compliance Documents" in self.at.dataframe[0].to_string() # Check if a default tool is displayed

    def test_tool_registry_add_new_tool(self):
        """Tests adding a new tool to the registry."""
        self.at.selectbox[0].set_value("Tool Registry").run()
        # Select "New Tool" (default in the tool selector)
        self.at.text_input[0].set_value("New Test Tool").run()
        self.at.selectbox[2].set_value("CALCULATE").run() # Tool Type
        self.at.text_area[0].set_value("A description for a new calculation tool.").run()
        self.at.checkbox[0].set_value(False).run() # is_side_effecting
        self.at.text_area[1].set_value(json.dumps({"expression": {"type": "string"}}, indent=2)).run()
        self.at.checkbox[1].set_value(True).run() # enabled
        self.at.form_submit_button[0].click().run()
        assert self.at.success[0].value == "Tool 'New Test Tool' added successfully!"
        tool_names = [tool.name for tool in self.at.session_state["tool_registry"].list_tools()]
        assert "New Test Tool" in tool_names

    def test_tool_registry_edit_existing_tool(self):
        """Tests editing an existing tool in the registry."""
        self.at.selectbox[0].set_value("Tool Registry").run()
        self.at.selectbox[1].set_value("Send Email").run() # Select an existing tool
        self.at.text_area[0].set_value("Updated description for Send Email tool.").run()
        self.at.form_submit_button[0].click().run()
        assert self.at.success[0].value == "Tool 'Send Email' updated successfully!"
        updated_tool = self.at.session_state["tool_registry"].get_tool_by_name("Send Email")
        assert updated_tool.description == "Updated description for Send Email tool."

    def test_tool_registry_add_tool_invalid_args_schema_json(self):
        """Tests adding a tool with invalid JSON for arguments schema."""
        self.at.selectbox[0].set_value("Tool Registry").run()
        self.at.text_input[0].set_value("Invalid Schema Tool").run()
        self.at.selectbox[2].set_value("RETRIEVE_DOCS").run()
        self.at.text_area[1].set_value("not json").run() # Invalid JSON
        self.at.form_submit_button[0].click().run()
        assert self.at.error[0].value == "Invalid JSON for arguments schema."

    def test_policy_editor_page_display_policies(self):
        """Verifies that the Policy Editor page displays configured policies."""
        self.at.selectbox[0].set_value("Policy Editor").run()
        assert self.at.header[0].value == "3. Policy Editor: Author Runtime Policies"
        assert "Strict Compliance Policy" in self.at.dataframe[0].to_string()
        assert "Permissive Exploration Policy" in self.at.dataframe[0].to_string()

    def test_policy_editor_add_new_policy(self):
        """Tests adding a new policy to the policy engine."""
        self.at.selectbox[0].set_value("Policy Editor").run()
        # Select "New Policy" (default in the policy selector)
        self.at.text_input[0].set_value("New Test Policy").run()
        self.at.multiselect[0].set_value(["RETRIEVE_DOCS"]).run()
        self.at.number_input[0].set_value(2).run() # max_steps
        self.at.number_input[1].set_value(0).run() # max_side_effect_actions
        self.at.checkbox[0].set_value(True).run() # require_approval_for_side_effects
        self.at.text_area[0].set_value("finance, secret").run() # restricted_keywords
        self.at.checkbox[1].set_value(False).run() # escalation_on_verification_fail
        self.at.form_submit_button[0].click().run()
        assert self.at.success[0].value == "Policy 'New Test Policy' added successfully!"
        assert "New Test Policy" in self.at.session_state["known_policies"]
        new_policy = self.at.session_state["known_policies"]["New Test Policy"]
        assert "RETRIEVE_DOCS" in [t.value for t in new_policy.allowed_tool_types]
        assert new_policy.max_steps == 2
        assert new_policy.restricted_keywords == ["finance", "secret"]

    def test_policy_editor_edit_existing_policy(self):
        """Tests editing an existing policy."""
        self.at.selectbox[0].set_value("Policy Editor").run()
        self.at.selectbox[1].set_value("Permissive Exploration Policy").run() # Select an existing policy
        self.at.number_input[0].set_value(15).run() # Change max_steps
        self.at.text_area[0].set_value("confidential").run() # Add restricted keyword
        self.at.form_submit_button[0].click().run()
        assert self.at.success[0].value == "Policy 'Permissive Exploration Policy' updated successfully!"
        updated_policy = self.at.session_state["known_policies"]["Permissive Exploration Policy"]
        assert updated_policy.max_steps == 15
        assert "confidential" in updated_policy.restricted_keywords

    def test_simulation_runner_page_run_simulation_allowed(self):
        """Tests running a simulation where actions are allowed."""
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run() # Select plan
        self.at.selectbox[2].set_value("Permissive Exploration Policy").run() # Select a permissive policy
        self.at.button[0].click().run()
        assert self.at.success[0].value == "Simulation Completed!"
        assert self.at.toast[0].value == "Simulation Completed!"
        assert self.at.session_state["last_run_audit_events"] is not None
        
        # Verify no tool blocked events with permissive policy
        blocked_events = [e for e in self.at.session_state["last_run_audit_events"] if e.event_type == "TOOL_BLOCKED"]
        assert len(blocked_events) == 0

    def test_simulation_runner_policy_denial_tool_type(self):
        """Tests simulation where a tool is denied due to policy's allowed_tool_types."""
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run() # Plan involves "Send Email"
        self.at.selectbox[2].set_value("Strict Compliance Policy").run() # Strict policy does not allow SEND_EMAIL
        self.at.button[0].click().run()
        assert self.at.success[0].value == "Simulation Completed!"
        
        blocked_event_found = False
        for event in self.at.session_state["last_run_audit_events"]:
            if event.event_type == "TOOL_BLOCKED":
                blocked_event_found = True
                assert event.payload["tool_name"] == "Send Email"
                assert event.payload["policy_decision"] == "DENY"
                break
        assert blocked_event_found, "Expected 'Send Email' to be blocked by 'Strict Compliance Policy'."

    def test_simulation_runner_restricted_keyword_denial(self):
        """Tests simulation where an action is denied due to restricted keywords."""
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Sensitive Data Handling").run() # Plan involves "sensitive" query
        self.at.selectbox[2].set_value("Strict Compliance Policy").run() # Strict policy has "sensitive" as restricted keyword
        self.at.button[0].click().run()
        assert self.at.success[0].value == "Simulation Completed!"

        blocked_event_found = False
        for event in self.at.session_state["last_run_audit_events"]:
            if event.event_type == "TOOL_BLOCKED":
                blocked_event_found = True
                assert "sensitive" in event.payload["evaluated_step"]["tool_args"]["query"]
                assert event.payload["policy_decision"] == "DENY"
                break
        assert blocked_event_found, "Expected action with 'sensitive' keyword to be blocked."

    def test_verification_results_page_run_checks(self):
        """Tests running verification checks on simulation output."""
        # First, run a simulation to generate audit events and outputs
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run()
        self.at.selectbox[2].set_value("Permissive Exploration Policy").run()
        self.at.button[0].click().run()
        
        self.at.selectbox[0].set_value("Verification Results").run()
        self.at.button[0].click().run()
        assert self.at.success[0].value == "Verification checks completed!"
        assert self.at.toast[0].value == "Verification checks completed!"
        assert self.at.session_state["last_run_verification_results"] is not None
        assert any(res.check_type == "CITATION_PRESENCE" for res in self.at.session_state["last_run_verification_results"])

    def test_verification_results_escalation_on_fail(self):
        """Tests the escalation indicator when verification checks fail."""
        # Modify knowledge base to cause a citation match failure
        self.at.selectbox[0].set_value("System Setup").run()
        modified_kb = MOCK_KNOWLEDGE_BASE.copy()
        del modified_kb["doc123"] # Remove doc123 which is cited in Compliance Inquiry plan
        self.at.text_area[0].set_value(json.dumps(modified_kb, indent=2)).run()
        self.at.button[0].click().run()
        
        # Run simulation to generate output with missing citation
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run()
        self.at.selectbox[2].set_value("Permissive Exploration Policy").run()
        self.at.button[0].click().run()
        
        # Run verification checks
        self.at.selectbox[0].set_value("Verification Results").run()
        self.at.button[0].click().run()
        
        assert self.at.warning[0].value == "Escalation Indicator: One or more verification checks failed. Review details above."
        
        fail_found = False
        for res in self.at.session_state["last_run_verification_results"]:
            if res.check_type.value == "CITATION_MATCH" and res.status == "FAIL":
                fail_found = True
                break
        assert fail_found, "Expected CITATION_MATCH to fail due to missing document."

    def test_audit_log_exports_summary_report(self):
        """Tests generating and displaying the simulation summary report."""
        # Run simulation and verification first to populate data
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run()
        self.at.selectbox[2].set_value("Strict Compliance Policy").run() # Will have 1 blocked action
        self.at.button[0].click().run()
        self.at.selectbox[0].set_value("Verification Results").run()
        self.at.button[0].click().run()

        self.at.selectbox[0].set_value("Audit Log & Exports").run()
        self.at.button[0].click().run() # Click "Generate Summary Report"
        assert self.at.session_state.generate_summary_clicked is True
        
        # Verify metrics based on "Compliance Inquiry" plan and "Strict Compliance Policy"
        # Plan steps: 1. Retrieve Docs (allowed), 2. Send Email (blocked by policy)
        # Total Agent Steps Simulated should count LLM_THINKING/TOOL_SELECTED/TOOL_BLOCKED
        # The agent simulator produces LLM_THINKING for each planned action.
        # So for 2 planned actions, there will be 2 LLM_THINKING events.
        # Step 1: LLM_THINKING, POLICY_EVALUATION (ALLOW), TOOL_EXECUTED, OUTPUT_GENERATED
        # Step 2: LLM_THINKING, POLICY_EVALUATION (DENY), TOOL_BLOCKED
        # Total "steps" for metrics: typically refer to planned agent steps.
        assert self.at.metric[0].value == "2" # Total Agent Steps Simulated (2 planned actions)
        assert self.at.metric[1].value == "1" # Policy Denials (Send Email is blocked)
        assert self.at.metric[2].value == "0" # Approval Requests Generated (Send Email is denied, not approved)
        assert self.at.error[0].value == "Critical: Policy engine successfully blocked unauthorized actions."

    def test_audit_log_exports_raw_audit_log_display(self):
        """Tests that the raw audit log is displayed."""
        # Run simulation first to populate data
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run()
        self.at.selectbox[2].set_value("Permissive Exploration Policy").run()
        self.at.button[0].click().run()

        self.at.selectbox[0].set_value("Audit Log & Exports").run()
        assert self.at.json[0].json is not None
        assert len(self.at.json[0].json) > 0 # Should contain audit events

    def test_audit_log_exports_download_all_artifacts(self):
        """Tests the 'Generate & Export All Artifacts' button."""
        # Run simulation and verification first
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run()
        self.at.selectbox[2].set_value("Permissive Exploration Policy").run()
        self.at.button[0].click().run()
        self.at.selectbox[0].set_value("Verification Results").run()
        self.at.button[0].click().run()

        self.at.selectbox[0].set_value("Audit Log & Exports").run()
        self.at.button[1].click().run() # "Generate & Export All Artifacts"
        assert self.at.success[0].value == "All audit artifacts generated successfully in 'temp_artifacts' directory locally."
        assert self.at.download_button[0].label == "Download All Artifacts as ZIP"
        
        # Clean up the temp_artifacts directory if it still exists after the test
        if os.path.exists("temp_artifacts"):
            shutil.rmtree("temp_artifacts")

    def test_audit_log_exports_individual_download_buttons_present(self):
        """Verifies individual download buttons are present after a run."""
        # Run simulation and verification first
        self.at.selectbox[0].set_value("Simulation Runner").run()
        self.at.selectbox[1].set_value("Compliance Inquiry").run()
        self.at.selectbox[2].set_value("Permissive Exploration Policy").run()
        self.at.button[0].click().run()
        self.at.selectbox[0].set_value("Verification Results").run()
        self.at.button[0].click().run()

        self.at.selectbox[0].set_value("Audit Log & Exports").run()
        
        # Check for individual download buttons by label
        assert self.at.download_button[1].label == "Download runtime_policy.json"
        assert self.at.download_button[2].label == "Download verification_results.json"
        assert self.at.download_button[3].label == "Download audit_log.jsonl"
        assert self.at.download_button[4].label == "Download failure_mode_analysis.md"
        assert self.at.download_button[5].label == "Download residual_risk_summary.md"
        assert self.at.download_button[6].label == "Download evidence_manifest.json"

    def test_audit_log_exports_no_data_initially(self):
        """Verifies the audit log & exports page displays no data messages when nothing has run."""
        # Start with a fresh app test instance where no simulation has run
        self.at = AppTest.from_file("app.py").run() 
        self.at.selectbox[0].set_value("Audit Log & Exports").run()
        assert self.at.info[0].value == "No data available to generate a summary report. Please run a simulation and verification first."
        assert self.at.info[1].value == "No audit log available from the last simulation."
        # The export button should also have a warning if clicked without data
        self.at.button[1].click().run() # "Generate & Export All Artifacts"
        assert self.at.warning[0].value == "Please run a simulation and verification first to generate artifacts."

```