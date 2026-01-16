id: 696a62e0af3b3d8cb59d81b2_documentation
summary: Case Study 4: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability

## 1. Introduction and System Overview
Duration: 00:05

<aside class="positive">
This step provides critical context for the codelab. Understanding the 'why' behind agent risk controls is paramount for an AI Safety Engineer. We'll outline the problem, the solution, and the core components that make up this robust system.
</aside>

In today's rapidly evolving landscape of Large Language Models (LLMs) and autonomous agents, ensuring their safe, compliant, and responsible operation is not just a best practice—it's a critical necessity. This codelab focuses on **QuLab Case Study 4**, which addresses the crucial challenge of implementing robust risk controls for LLM-powered agentic systems. Following a hypothetical critical production incident, this system has been designed to provide comprehensive tools for defining and enforcing agent runtime policies, verifying outputs, and maintaining an immutable audit trail.

As an AI Safety Engineer, your role involves mitigating risks associated with agent autonomy, preventing misuse, ensuring data privacy, and maintaining regulatory compliance. This application provides a hands-on environment to achieve these objectives by building guardrails around agent behavior.

### Why is this important?

*   **Preventing Harm:** Agents, especially those with access to tools (e.g., sending emails, writing files), can cause real-world harm if unchecked.
*   **Ensuring Compliance:** Regulatory frameworks often require strict adherence to policies, which agents must also follow.
*   **Building Trust:** Demonstrating control and auditability is essential for user and stakeholder trust in AI systems.
*   **Debugging and Post-Mortem Analysis:** A comprehensive audit trail is vital for understanding agent behavior, especially when things go wrong.

### Core Concepts Explained

This codelab will guide you through the implementation of several key components:

1.  **Tool Registry:** A centralized catalog of all tools an agent can use, along with their capabilities and potential side-effects.
2.  **Runtime Policies:** Dynamic rules that govern an agent's behavior at each step, enforcing constraints like allowed tools, action limits, and restricted keywords.
3.  **Agent Simulation:** A mechanism to test agent plans against defined policies and observe their behavior in a controlled environment.
4.  **Verification Harness:** A system to validate the quality, accuracy, and compliance of an agent's outputs, independent of runtime enforcement.
5.  **Audit Log and Evidence Manifest:** An immutable record of all agent actions, policy decisions, and verification outcomes, crucial for accountability and governance.

### System Architecture Overview

The system architecture is designed to enforce safety and compliance throughout the agent's lifecycle, from capability definition to execution and post-run analysis.

```mermaid
graph TD
    A[Agent Simulator] -->|Uses| B(Predefined Plans)
    A -->|Interacts with| C(Tool Registry)
    A -->|Consults| D(Policy Engine)
    C -->|Executes mock actions via| E(Mock Tool Executors)
    D -->|Evaluates actions against| F(Runtime Policies)
    A -->|Generates| G[Audit Events]

    G -->|Inputs to| H(Verification Harness)
    H -->|Checks against| I(Knowledge Base)
    H -->|Generates| J[Verification Results]

    G & J & F --> K[Audit Log & Export System]
    K --> L[Comprehensive Reports & Evidence Manifest]

    subgraph User Interface (Streamlit)
        M[System Setup] --> I
        N[Tool Registry] --> C
        O[Policy Editor] --> F
        P[Simulation Runner] --> A
        Q[Verification Results] --> H
        R[Audit Log & Exports] --> K
    end
```

<aside class="positive">
<b>Figure 1:</b> QuLab Agent Risk Control System Architecture. This diagram illustrates how the `AgentSimulator` leverages `ToolRegistry` and `PolicyEngine` to execute agent plans, generating `AuditEvents`. The `VerificationHarness` then analyzes these events and the `KnowledgeBase` to produce `VerificationResults`. All these components feed into a comprehensive `Audit Log & Export System` for governance.
</aside>

### Initial System Setup

The first step is to configure the `Knowledge Base` which acts as a source of truth for the `RETRIEVE_DOCS` tool and for verification checks.

1.  Navigate to the `System Setup` page using the sidebar.
2.  Review the pre-populated mock knowledge base.
    ```json
    {
      "doc_id_1": "This document outlines the strict data privacy policy for handling customer PII. Anonymization is mandatory for all aggregated data used in analytics. Sharing raw customer data externally is strictly forbidden.",
      "doc_id_2": "The internal compliance policy mandates that all emails containing sensitive financial information must be encrypted and sent only to approved recipients. Data minimization principles apply.",
      "doc_id_3": "Operational guidelines for database access: Only authorized personnel can execute write operations. Read-only access for analytics requires prior approval and must use anonymized datasets."
    }
    ```
3.  You can modify this JSON content to add or change documents that the agent might "retrieve" or that are used for factual verification.
4.  Click **"Update Knowledge Base"** to save your changes.

## 2. Defining Agent Capabilities with the Tool Registry
Duration: 00:10

<aside class="positive">
The `Tool Registry` is the foundational layer. By precisely defining each tool's characteristics, you establish clear boundaries for what an agent can and cannot do, enabling granular control through policies.
</aside>

As an AI Safety Engineer, your next task is to define the specific capabilities (tools) that your compliance assistant agent can utilize. This involves meticulously classifying each tool by its type, determining if it has side effects, and specifying its argument schema. This detailed definition forms the bedrock for applying granular runtime policies.

The core data model for a tool is `ToolDefinition`:

```python
class ToolDefinition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    tool_type: ToolType
    description: str
    is_side_effecting: bool
    args_schema: Dict[str, Any]
    enabled: bool = True
```

Here, `is_side_effecting` is a crucial boolean flag indicating whether the tool modifies external state (e.g., sending an email, writing to a database) or merely retrieves information (e.g., searching a document).

Mathematically, a tool $T_i$ can be represented as a tuple of its attributes:
$$ T_i = ( \text{id}_i, \text{name}_i, \text{type}_i, \text{description}_i, \text{is\_side\_effecting}_i, \text{args\_schema}_i, \text{enabled}_i ) $$
where $\text{is\_side\_effecting}_i \in \{ \text{True, False} \}$ reflects its impact on the external environment.

### Exploring and Managing Tools

1.  Navigate to the `Tool Registry` page using the sidebar.
2.  You will see a table listing all pre-configured tools. Observe their `tool_type`, `is_side_effecting` status, and `args_schema`.

    <aside class="positive">
    <b>Example:</b> The `Send Email` tool is marked `is_side_effecting=True` because it modifies the external world by sending an email. The `Retrieve Compliance Documents` tool is `is_side_effecting=False` as it only fetches information.
    </aside>

3.  **Add a New Tool:**
    *   In the "Add/Edit Tool" section, select "New Tool" from the dropdown.
    *   Fill in the details for a new tool. For instance, let's create a "Data Anonymization Service":
        *   **Tool Name:** `Anonymize Data`
        *   **Tool Type:** `CALCULATE` (as a proxy for a data processing service)
        *   **Description:** `Anonymizes sensitive data fields.`
        *   **Is Side-Effecting?:** `True` (as it modifies data)
        *   **Arguments Schema (JSON):**
            ```json
            {
              "data": {"type": "string", "description": "The raw data to anonymize."},
              "fields": {"type": "array", "items": {"type": "string"}, "description": "List of fields to anonymize."}
            }
            ```
        *   **Enabled:** `True`
    *   Click **"Save Tool"**. You should see it appear in the "Configured Tools" table.

4.  **Edit an Existing Tool:**
    *   Select an existing tool from the dropdown (e.g., `Perform Calculation`).
    *   Modify its properties, such as changing its `description` or `args_schema`.
    *   Click **"Save Tool"**.

Here's how these tools are added to the `ToolRegistry` in the application's backend:

```python
# From app() initialization:
st.session_state.tool_registry = ToolRegistry()

retrieve_docs_tool = ToolDefinition(
    name="Retrieve Compliance Documents",
    tool_type=ToolType.RETRIEVE_DOCS,
    description="Retrieves internal compliance documents based on a query.",
    is_side_effecting=False,
    args_schema={"query": {"type": "string", "description": "The search query for documents."}}
)
send_email_tool = ToolDefinition(
    name="Send Email",
    tool_type=ToolType.SEND_EMAIL,
    description="Sends an email to specified recipients.",
    is_side_effecting=True,
    args_schema={"recipient": {"type": "string", "description": "Email recipient"}, "subject": {"type": "string", "description": "Email subject"}, "body": {"type": "string", "description": "Email body"}}
)
# ... other tool definitions ...

st.session_state.tool_registry.add_tool(retrieve_docs_tool, mock_retrieve_docs_cell_22)
st.session_state.tool_registry.add_tool(send_email_tool, mock_send_email_cell_22)
# Note: For 'Perform Calculation', no explicit mock executor is provided as it's a non-side-effecting placeholder.
```

## 3. Authoring Runtime Policies
Duration: 00:15

<aside class="negative">
Careless policy definition can either overly restrict an agent (making it useless) or insufficiently restrict it (leading to safety incidents). Pay close attention to how each parameter contributes to the overall risk posture.
</aside>

With the agent's capabilities (tools) defined, the next crucial step for the AI Safety Engineer is to translate the organization's safety and compliance requirements into concrete `RuntimePolicy` instances. These policies set limits on agent autonomy, control tool access, and define how to handle sensitive operations or keywords. We will define both strict and permissive policies to observe their effects.

The `RuntimePolicy` data model captures various constraints:

```python
class RuntimePolicy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    allowed_tool_types: List[ToolType]
    max_steps: int
    max_side_effect_actions: int
    require_approval_for_side_effects: bool
    restricted_keywords: List[str]
    escalation_on_verification_fail: bool
```

Mathematically, a `RuntimePolicy` $P$ is a set of constraints that an agent's actions must satisfy:
$$ P = \{ \text{name}, \text{AllowedToolTypes}, \text{MaxSteps}, \text{MaxSideEffectActions}, \text{RequireApproval}, \text{RestrictedKeywords}, \text{EscalateOnFail} \} $$
where $\text{AllowedToolTypes}$ is a subset of all available $\text{ToolType}$, $\text{MaxSteps}$ and $\text{MaxSideEffectActions}$ are positive integers, $\text{RequireApproval} \in \{ \text{True, False} \}$, $\text{RestrictedKeywords}$ is a list of strings, and $\text{EscalateOnFail} \in \{ \text{True, False} \}$.

### Managing Runtime Policies

1.  Navigate to the `Policy Editor` page using the sidebar.
2.  You will see a table of pre-defined policies: `Strict Compliance Policy` and `Permissive Exploration Policy`.
    *   **Strict Compliance Policy:** Designed for high-assurance scenarios. It typically restricts side-effecting tools, limits steps, requires approval for sensitive actions, and blocks certain keywords.
    *   **Permissive Exploration Policy:** Allows for more flexibility, suitable for development or low-risk exploration, with fewer restrictions.

3.  **Create a New Policy:**
    *   In the "Create/Edit Policy" section, select "New Policy" from the dropdown.
    *   **Policy Name:** `Analytics Data Policy`
    *   **Allowed Tool Types:** Select `RETRIEVE_DOCS`, `CALCULATE`, `ANONYMIZE_DATA` (if you added it in the previous step).
    *   **Maximum Steps:** `7` (allowing a few more steps for analysis)
    *   **Maximum Side-Effecting Actions:** `1` (allowing anonymization but not many other side-effects without explicit approval)
    *   **Require Approval for Side-Effects:** `True`
    *   **Restricted Keywords (comma-separated):** `customer PII, raw data, delete records`
    *   **Escalate on Verification Fail:** `True`
    *   Click **"Save Policy"**.

4.  **Edit an Existing Policy:**
    *   Select `Permissive Exploration Policy` from the dropdown.
    *   Change `Maximum Steps` to `10`.
    *   Add `internal only, confidential` to `Restricted Keywords`.
    *   Click **"Save Policy"**.

Here's how the example policies are defined in the backend:

```python
# From source.py or app() initialization:
strict_compliance_policy = RuntimePolicy(
    name="Strict Compliance Policy",
    allowed_tool_types=[ToolType.RETRIEVE_DOCS, ToolType.CALCULATE],
    max_steps=5,
    max_side_effect_actions=0, # No side effects allowed
    require_approval_for_side_effects=True,
    restricted_keywords=["customer PII", "financial data", "delete records"],
    escalation_on_verification_fail=True
)

permissive_exploration_policy = RuntimePolicy(
    name="Permissive Exploration Policy",
    allowed_tool_types=[t for t in ToolType], # All tool types
    max_steps=10,
    max_side_effect_actions=3,
    require_approval_for_side_effects=False, # No approval needed
    restricted_keywords=["confidential passwords"], # Fewer restrictions
    escalation_on_verification_fail=False # Less strict on verification fails
)

st.session_state.known_policies = {
    strict_compliance_policy.name: strict_compliance_policy,
    permissive_exploration_policy.name: permissive_exploration_policy
}
```

## 4. Simulation Runner: Test Policy Enforcement
Duration: 00:20

<aside class="positive">
This is where theory meets practice. Simulating agent behavior under different policies allows you to observe policy enforcement in action and understand its impact on agent autonomy and safety.
</aside>

With tools and policies defined, the `AgentSimulator` and `PolicyEngine` become central. The `PolicyEngine` evaluates each agent action against the active policy, deciding whether to allow, deny, or require approval. The `AgentSimulator` then mimics the LLM's multi-step decision-making process based on predefined plans, interacting with mocked tools and logging every action and policy decision.

### Policy Engine Evaluation Logic

Let $P$ be the active `RuntimePolicy` and $A_t$ be the `AgentStep` at time $t$. The `PolicyEngine` determines the `PolicyAction` (ALLOWED, DENIED, REQUIRES_APPROVAL) based on the following logic:

1.  **Tool Type Check**: If $A_t.\text{selected\_tool}$ is not `None` and $A_t.\text{selected\_tool.tool\_type} \notin P.\text{allowed\_tool\_types}$, then the action is **DENIED**.
2.  **Step Limit Check**: If $A_t.\text{step\_number} > P.\text{max\_steps}$, then the action is **DENIED**.
3.  **Restricted Keywords Check**: If any keyword $k \in P.\text{restricted\_keywords}$ is found in $A_t.\text{planned\_action}$ or $A_t.\text{tool\_args}$ (if applicable), then the action is **DENIED**.
4.  **Side-Effect Approval Check**: If $A_t.\text{selected\_tool}$ is not `None`, $A_t.\text{selected\_tool.is\_side\_effecting}$ is `True`, and $P.\text{require\_approval\_for\_side\_effects}$ is `True`, then the action **REQUIRES\_APPROVAL**.
5.  **Side-Effect Count Check**: If $A_t.\text{selected\_tool}$ is not `None`, $A_t.\text{selected\_tool.is\_side\_effecting}$ is `True`, and the count of previous side-effecting actions in the current run exceeds $P.\text{max\_side\_effect\_actions}$, then the action is **DENIED**.
6.  Otherwise, the action is **ALLOWED**.

Every decision and action is recorded as an `AuditEvent`.

### Running a Simulation

1.  Navigate to the `Simulation Runner` page using the sidebar.
2.  **Select Plan Template:** The `PLAN_LIBRARY` contains predefined sequences of `AgentStep`s.
    *   **"Compliance Report Generation (Sensitive Data)"**: This plan involves retrieving documents, querying a database (potentially sensitive), and sending an email.
    *   **"Document Retrieval and Summary"**: A simpler plan focusing on information retrieval.
    *   **"Financial Transaction Analysis"**: Involves calculations and potentially querying a database.

    Choose `Compliance Report Generation (Sensitive Data)`.

3.  **Select Runtime Policy:**
    *   First, select `Strict Compliance Policy`.
    *   Click **"Run Simulation"**.

    Observe the `Simulation Trace`. You should see several `TOOL_BLOCKED` events, particularly for `Query Internal Database` and `Send Email`, as the `Strict Compliance Policy` has `max_side_effect_actions=0` and restricts tool types. You might also see `APPROVAL_REQUESTED` if `require_approval_for_side_effects` is true and a side-effecting tool is selected.

    ```console
    # Example Audit Event (Blocked)
    {
      "run_id": "...",
      "timestamp": "...",
      "step_number": 2,
      "event_type": "TOOL_BLOCKED",
      "payload": {
        "planned_action": "Query internal database for compliance violations related to PII.",
        "tool_name": "Query Internal Database",
        "tool_args": "{\"query\": \"SELECT * FROM PII_violations WHERE date > '2023-01-01'\"}",
        "policy_action": "DENIED",
        "policy_reason": "ToolType.QUERY_DB not in allowed_tool_types OR Restricted keyword 'PII' found."
      }
    }
    ```

4.  **Run with a Different Policy:**
    *   Select `Compliance Report Generation (Sensitive Data)` again.
    *   This time, select `Permissive Exploration Policy`.
    *   Click **"Run Simulation"**.

    Observe the `Simulation Trace`. You should see more `TOOL_EXECUTED` events, as this policy allows more freedom. There might still be `TOOL_BLOCKED` events if restricted keywords are hit.

The `AgentSimulator` is initialized with the `ToolRegistry` and `PolicyEngine`:

```python
# From app() initialization:
st.session_state.agent_simulator = AgentSimulator(
    st.session_state.tool_registry,
    st.session_state.policy_engine
)

# Running the plan:
run_events = st.session_state.agent_simulator.run_plan(
    selected_plan_name,
    policy_to_run
)
```

## 5. Verification Results: Validate Agent Outputs
Duration: 00:15

<aside class="negative">
Runtime policies prevent agents from *doing* harmful things, but they don't inherently guarantee the *quality*, *accuracy*, or *completeness* of the agent's final outputs. This is where verification steps in.
</aside>

Runtime policies are essential for preventing an agent from taking unauthorized actions. However, they don't guarantee the quality, accuracy, or factual consistency of the agent's generated content. As an AI Safety Engineer, you must also implement a `Verification Harness` to check the integrity of generated content, especially for compliance assistants. This includes verifying citations, fact consistency, and adherence to refusal policies.

The `Verification Harness` performs several checks:

### Verification Checks Logic

1.  **Citation Presence**: Does the output include at least one `[DOC:...]` marker, indicating that information was sourced?
2.  **Citation Match (Mocked)**: For each citation, does the `doc_id` reference an *expected* document ID from the mock knowledge base snippets? (In a real system, this would involve looking up the actual document content and ensuring relevance).
3.  **Fact Consistency (Proxy)**: Do key terms from the prompt appear in the "retrieved" snippets associated with citations? (A simplified proxy for actual fact-checking, focusing on keyword overlap).
4.  **Refusal Policy**: If the input contained high-risk instructions, did the agent's output indicate refusal or escalation instead of attempting to comply?

Each check produces a `VerificationResult`:

```python
class VerificationResult(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    check_type: VerificationCheck
    status: str # "PASS", "FAIL", "N/A"
    details: str
    related_audit_event_step: Optional[int] = None
    related_audit_event_action: Optional[str] = None
```

### Running Verification Checks

1.  Navigate to the `Verification Results` page using the sidebar.
2.  Ensure you have run a simulation previously (e.g., "Compliance Report Generation (Sensitive Data)" with `Permissive Exploration Policy` to ensure there are outputs).
3.  Click **"Run Verification Checks on Last Simulation Output"**.
    *   The system will process the `OUTPUT_GENERATED` events from the last simulation run.
    *   It will apply the defined verification checks against these outputs.

4.  Review the `Verification Results Summary` table.
    *   Observe the `status` column for `PASS`, `FAIL`, or `N/A`.
    *   The table will be color-coded (light green for PASS, light red for FAIL) to easily spot issues.
    *   The "Escalation Indicator" will highlight if any checks failed.

    <aside class="positive">
    <b>Self-correction opportunity:</b> If a verification check fails (e.g., for citation presence), it might indicate that the agent failed to retrieve necessary documents or incorporate them correctly. This could prompt adjustments to the agent's prompt, plan, or even the available tools.
    </aside>

The `VerificationHarness` is initialized with the `MOCK_KNOWLEDGE_BASE`:

```python
# From app() initialization:
st.session_state.verification_harness = VerificationHarness(MOCK_KNOWLEDGE_BASE)

# Example of how checks are run:
# Inside render_verification_results_page():
results_for_output.append(st.session_state.verification_harness.check_citation_presence(output_text))
results_for_output.append(st.session_state.verification_harness.check_citation_match(output_text))
results_for_output.append(st.session_state.verification_harness.check_fact_consistency(output_text, relevant_text, key_terms))
results_for_output.append(st.session_state.verification_harness.check_refusal_policy(agent_input_proxy, output_text, high_risk_phrases))
```

## 6. Audit Log & Exports: Governance and Traceability
Duration: 00:10

<aside class="positive">
Auditability is the cornerstone of responsible AI. This section ensures that every decision, action, and outcome of the agent and its controls is immutably recorded, providing the necessary evidence for compliance and post-mortem analysis.
</aside>

This final section synthesizes the outcomes of your work as an AI Safety Engineer. A comprehensive audit trail is provided, along with the ability to export all crucial information—policy definitions, verification outcomes, and detailed audit logs—in standardized formats. This documentation proves the LLM-powered compliance assistant operates within its defined safety and compliance guardrails, building trust and reducing operational risk.

### Simulation Summary Report

1.  Navigate to the `Audit Log & Exports` page using the sidebar.
2.  Click **"Generate Summary Report"**.
3.  Review the key metrics:
    *   Total Agent Steps Simulated
    *   Policy Denials (Blocked Actions)
    *   Approval Requests Generated
    *   Total Verification Checks Run
    *   Verification Checks Passed
    *   Verification Checks Failed

    This summary provides a quick overview of how the agent performed under the selected policy and how effectively the guardrails were enforced.

### Raw Audit Log (JSONL)

The audit log is a chronological sequence of `AuditEvent` objects, detailing every step the agent considered, every tool selected, every policy decision made, and every output generated. Each `AuditEvent` is serialized as a JSON object, creating a JSONL (JSON Lines) format, which is ideal for streaming and parsing.

```python
class AuditEvent(BaseModel):
    run_id: str
    timestamp: datetime.datetime
    step_number: int
    event_type: str # E.g., "LLM_THINKING", "TOOL_SELECTED", "TOOL_EXECUTED", "TOOL_BLOCKED", "APPROVAL_REQUESTED", "OUTPUT_GENERATED"
    policy_name: str
    policy_action: Optional[PolicyAction] = None
    policy_reason: Optional[str] = None
    payload: Dict[str, Any]
```

Review the "Raw Audit Log (JSONL)" section. This is the detailed, immutable record of the simulation.

### Export Audit Artifacts

The system provides robust export capabilities to package all relevant artifacts for review, archival, or regulatory submission.

1.  Click **"Generate & Export All Artifacts"**.
    *   This will create a `temp_artifacts` directory locally, save all relevant files, and then offer a `.zip` file for download.
    *   The `export_artifacts` function bundles:
        *   `runtime_policy.json`: The specific `RuntimePolicy` used for the simulation.
        *   `verification_results.json`: The outcomes of all verification checks.
        *   `audit_log.jsonl`: The complete sequence of `AuditEvents`.
        *   `failure_mode_analysis.md`: A template for analyzing potential failure modes.
        *   `residual_risk_summary.md`: A template for summarizing residual risks and mitigation plans.
        *   `evidence_manifest.json`: A manifest linking all artifacts, including their cryptographic hashes for integrity verification.

2.  Click **"Download All Artifacts as ZIP"**.
    *   This provides a single archive for easy sharing and storage.

<aside class="positive">
The `evidence_manifest.json` is crucial for proving the integrity and authenticity of the audit trail. By including cryptographic hashes of all generated artifacts, it ensures that the exported data has not been tampered with since its creation.
</aside>

Here's an example of an `EvidenceManifest`:

```python
class EvidenceManifest(BaseModel):
    run_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    inputs_hash: str # Hash of the simulation inputs (e.g., plan, initial knowledge base)
    outputs_hash: str # Hash of the full audit log
    artifacts: Dict[str, str] # Map of artifact filename to its hash (e.g., "runtime_policy.json": "sha256_hash_value")
    # Additional metadata like policy_id, agent_version etc. could be added
```

The `calculate_file_hash` function ensures integrity:

```python
# From source.py:
def calculate_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(4096) # Read in 4KB chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
```

The process of exporting the artifacts and generating the manifest ensures a verifiable record of the agent's operation, policy adherence, and safety performance. This completes the loop for robust LLM agent risk control, providing both proactive runtime enforcement and retrospective auditability.
