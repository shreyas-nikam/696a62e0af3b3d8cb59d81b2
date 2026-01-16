id: 696a62e0af3b3d8cb59d81b2_user_guide
summary: Case Study 4: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability

## 1. System Setup: Initializing the Agent Environment
Duration: 05:00

<aside class="positive">
Welcome to QuLab! This codelab will guide you through building a robust framework for managing risks in LLM-powered agentic systems. You'll learn how to define agent capabilities, enforce runtime policies, verify agent outputs, and maintain an immutable audit trail. This is crucial for deploying AI safely and compliantly, especially in regulated industries.
</aside>

As an AI Safety Engineer, your primary role is to ensure that advanced LLM-powered agents operate within defined boundaries. Following a hypothetical critical production incident, your organization has mandated a robust solution for defining and enforcing agent runtime policies. This application provides a hands-on environment to configure agent capabilities, author dynamic policies, simulate agent behavior, and verify its outputs, all while generating a comprehensive audit trail.

This first step focuses on setting up the foundational knowledge for our agent.

### Knowledge Base Configuration

The application uses a "Knowledge Base" to simulate a repository of information that an agent might access. This is particularly relevant for tools like `RETRIEVE_DOCS` and for conducting verification checks, such as ensuring that the agent's output is consistent with known facts or properly cites sources.

1.  **Locate the `Knowledge Base Snippets` section:** On the "System Setup" page, you'll see a text area labeled "Edit Mock Knowledge Base (JSON)".
2.  **Review the default content:** This JSON object contains key-value pairs where keys are document IDs and values are the content of mock documents.
    ```json
    {
      "doc_001": "Compliance policy regarding sensitive data handling requires anonymization for all external reports.",
      "doc_002": "Internal memo on data privacy guidelines: only authorized personnel may access raw financial data.",
      "doc_003": "Approved communication channels for customer interactions include email and secure portal, but never direct SMS for sensitive information.",
      "doc_004": "Procedure for escalating critical security vulnerabilities: contact incident response team immediately."
    }
    ```
3.  **Experiment (Optional):** You can modify this JSON content. For instance, add a new document or change existing text. Ensure it remains valid JSON.
4.  **Update the Knowledge Base:** Click the **`Update Knowledge Base`** button. You should see a "Knowledge Base updated successfully!" message. This action updates the internal knowledge base used by the agent simulator and verification harness.

<aside class="positive">
A well-defined knowledge base is vital for agents that rely on information retrieval (RAG - Retrieval Augmented Generation) and for verifying the factual accuracy and groundedness of their outputs.
</aside>

## 2. Tool Registry: Define Agent Capabilities
Duration: 07:00

With the system's foundational knowledge base established, the next critical step for an AI Safety Engineer is to define the specific tools the compliance assistant (our agent) can use. This involves classifying each tool by its type, determining if it has side effects (e.g., sending an email vs. retrieving a document), and specifying its argument schema. This meticulous definition forms the foundation for applying granular runtime policies later.

We represent a tool as:
$$ \text{Tool Definition: } T_i = \{ \text{id, name, type, description, is_side_effecting, args_schema, enabled} \} $$
where $T_i$ represents the $i$-th tool, and `is_side_effecting` is a boolean flag indicating if the tool modifies external state (e.g., sending an email, writing to a file).

### Understanding Configured Tools

1.  **Navigate to "Tool Registry":** In the sidebar, select "Tool Registry".
2.  **Review existing tools:** The "Configured Tools" table displays the tools already defined in the system. Observe their names, types, descriptions, whether they are side-effecting, and their argument schemas.
    *   **`Retrieve Compliance Documents`**: `RETRIEVE_DOCS`, `is_side_effecting=False`. This tool just fetches information.
    *   **`Send Email`**: `SEND_EMAIL`, `is_side_effecting=True`. Sending an email changes external state.
    *   **`Query Internal Database`**: `QUERY_DB`, `is_side_effecting=True`. Querying a DB might retrieve sensitive data.
    *   **`Write File to Storage`**: `WRITE_FILE`, `is_side_effecting=True`. Writing a file modifies storage.
    *   **`Perform Calculation`**: `CALCULATE`, `is_side_effecting=False`. Performing a calculation is purely informational.

### Adding or Editing Tools

The "Add/Edit Tool" section allows you to manage the agent's available tools.

1.  **Select a Tool to Edit:** Use the "Select Tool to Edit (or 'New Tool')" dropdown.
    *   To edit an existing tool, select its name. The form fields will pre-populate with its current details.
    *   To add a new tool, keep "New Tool" selected.
2.  **Tool Details:**
    *   **Tool Name:** A unique identifier for the tool.
    *   **Tool Type:** Categorizes the tool's function (e.g., `RETRIEVE_DOCS`, `SEND_EMAIL`, `CALCULATE`). This is crucial for policy enforcement.
    *   **Description:** A clear explanation of what the tool does.
    *   **Is Side-Effecting?:** Check this box if the tool modifies the external environment (e.g., sends data, writes files, performs transactions). Non-side-effecting tools are typically read-only.
    *   **Arguments Schema (JSON):** Defines the expected input arguments for the tool in JSON schema format. For example, `{"query": {"type": "string", "description": "The search query."}}`.
    *   **Enabled:** Toggles whether the tool is active and available for the agent to use.
3.  **Save Changes:** After making your edits or filling in details for a new tool, click **`Save Tool`**. The "Configured Tools" table will update.

<aside class="negative">
Carefully consider if a tool is side-effecting. Incorrectly classifying a tool as non-side-effecting when it *does* modify external state can lead to policy bypasses and significant security risks.
</aside>

## 3. Policy Editor: Author Runtime Policies
Duration: 10:00

With the agent's tools defined, the next crucial step for the AI Safety Engineer is to translate the organization's safety and compliance requirements into concrete `RuntimePolicy` instances. This involves setting limits on agent autonomy, controlling tool access, and defining how to handle sensitive operations or keywords. We will define various policies, such as strict and permissive ones, for comparison during simulation.

We define a runtime policy as:
$$ \text{Runtime Policy: } P = \{ \text{id, name, allowed\_tools, max\_steps, approval\_for\_side\_effects, restricted\_keywords, ...} \} $$
This definition allows us to enforce granular controls on agent behavior.

### Defined Runtime Policies

1.  **Navigate to "Policy Editor":** In the sidebar, select "Policy Editor".
2.  **Review existing policies:** The "Defined Runtime Policies" table shows the policies already configured. You'll likely see:
    *   **`Strict Compliance Policy`**: This policy often has very restrictive settings, e.g., only allowing specific tool types, requiring approval for all side effects, and strict keyword restrictions.
    *   **`Permissive Exploration Policy`**: This policy might be less restrictive, allowing a wider range of tool types and fewer approval requirements, suitable for testing or less sensitive tasks.

### Create or Edit a Policy

The "Create/Edit Policy" section allows you to manage these runtime rules.

1.  **Select Policy to Edit:** Use the "Select Policy to Edit (or 'New Policy')" dropdown.
    *   Select an existing policy to modify its rules.
    *   Select "New Policy" to define a new set of rules.
2.  **Policy Parameters:**
    *   **Policy Name:** A unique name for your policy. (Note: The name cannot be changed when editing an existing policy).
    *   **Allowed Tool Types:** Select which categories of tools the agent is permitted to use under this policy. (e.g., only `RETRIEVE_DOCS` for a read-only policy).
    *   **Maximum Steps:** Limits the total number of steps an agent can take in a single run. This prevents infinite loops or overly complex decision-making.
    *   **Maximum Side-Effecting Actions:** Limits how many actions that modify external state (e.g., `SEND_EMAIL`, `WRITE_FILE`) an agent can perform.
    *   **Require Approval for Side-Effects:** If checked, any attempt by the agent to use a side-effecting tool will trigger an "approval requested" event instead of direct execution.
    *   **Restricted Keywords (comma-separated):** A list of keywords (e.g., "confidential", "delete", "transfer funds") that, if detected in the agent's planned action or tool arguments, will cause the action to be blocked.
    *   **Escalate on Verification Fail:** If checked, a failed verification check (from Step 5) will be flagged for immediate attention.
3.  **Save Policy:** Click **`Save Policy`** to apply your changes or create the new policy. The "Defined Runtime Policies" table will update.

<aside class="positive">
By experimenting with different policy settings (e.g., a highly restrictive "Strict Compliance Policy" versus a more lenient "Permissive Exploration Policy"), you can understand how granular controls directly impact agent behavior and safety.
</aside>

## 4. Simulation Runner: Test Policy Enforcement
Duration: 12:00

Now that we have defined our tools and policies, the AI Safety Engineer needs to construct the core components that will enforce these rules: the `PolicyEngine` and the `AgentSimulator`. The `PolicyEngine` will evaluate each agent action against the active policy, deciding whether to allow, deny, or require approval. The `AgentSimulator` will then mimic the LLM's multi-step decision-making process based on predefined plans, interacting with mocked tools and logging every action and policy decision.

### Policy Engine Evaluation Logic

Let $P$ be the active `RuntimePolicy` and $A_t$ be the `AgentStep` at time $t$. The policy engine evaluates an action as follows:

1.  **Tool Type Check**: If $A_t.\text{selected_tool}$ is not `None` and $A_t.\text{selected_tool.tool_type} \notin P.\text{allowed_tool_types}$, then the action is **DENIED**.
2.  **Step Limit Check**: If $A_t.\text{step_number} > P.\text{max_steps}$, then the action is **DENIED**.
3.  **Restricted Keywords Check**: If any keyword $k \in P.\text{restricted_keywords}$ is found in $A_t.\text{planned_action}$ or $A_t.\text{tool_args}$ (if applicable), then the action is **DENIED**.
4.  **Side-Effect Approval Check**: If $A_t.\text{selected_tool}$ is not `None`, $A_t.\text{selected_tool.is_side_effecting}$ is `True`, and $P.\text{require_approval_for_side_effects}$ is `True`, then the action **REQUIRES_APPROVAL**.
5.  **Side-Effect Count Check**: If $A_t.\text{selected_tool}$ is not `None`, $A_t.\text{selected_tool.is_side_effecting}$ is `True`, and the count of previous side-effecting actions in the current run exceeds $P.\text{max_side_effect_actions}$, then the action is **DENIED**.
6.  Otherwise, the action is **ALLOWED**.

### Running a Simulation

1.  **Navigate to "Simulation Runner":** In the sidebar, select "Simulation Runner".
2.  **Select Plan Template:** Choose a predefined plan that the agent will attempt to execute. These plans represent typical agent workflows. For example, "Compliance Report Generation" or "Handle Data Access Request".
3.  **Select Runtime Policy:** Choose one of the policies you configured in the previous step (e.g., "Strict Compliance Policy" or "Permissive Exploration Policy").
4.  **Run Simulation:** Click the **`Run Simulation`** button. The application will simulate the agent's actions step-by-step, applying the chosen policy at each turn. You'll see a spinner indicating progress, followed by a "Simulation Completed!" message or an error if the simulation fails early due to policy violations.

### Interpreting the Simulation Trace

After a simulation completes, the "Simulation Trace" table will populate, providing a detailed log of every event that occurred.

*   **`run_id`**: A unique ID for the entire simulation run.
*   **`step_number`**: The sequence number of the agent's action within the run.
*   **`event_type`**: Describes what happened (e.g., `LLM_THINKING`, `TOOL_SELECTED`, `TOOL_BLOCKED`, `APPROVAL_REQUESTED`, `TOOL_EXECUTED`, `OUTPUT_GENERATED`).
*   **`policy_action`**: The decision made by the Policy Engine (`ALLOWED`, `DENIED`, `REQUIRES_APPROVAL`).
*   **`policy_reason`**: Explains *why* the policy engine made that decision (e.g., "Tool type not allowed", "Restricted keyword detected", "Approval required for side-effect").
*   **`timestamp`**: When the event occurred.
*   **`payload`**: A JSON object containing detailed information about the event, such as the agent's planned action, the tool selected, its arguments, or the tool's result.

<aside class="positive">
Pay close attention to `TOOL_BLOCKED` and `APPROVAL_REQUESTED` events, and their corresponding `policy_action` and `policy_reason`. These indicate that your policies are effectively enforcing boundaries and capturing potentially risky actions.
</aside>

## 5. Verification Results: Validate Agent Outputs
Duration: 10:00

Runtime policies prevent an agent from *doing* harmful things by controlling its actions. However, they don't guarantee the *quality* or *accuracy* of the agent's outputs. As an AI Safety Engineer, you must also implement a `Verification Harness` to check the integrity of generated content, especially for compliance assistants. This includes verifying citations, fact consistency, and adherence to refusal policies.

### Verification Checks Logic

The `Verification Harness` performs several checks on the agent's output:

1.  **Citation Presence**: Does the output include at least one `[DOC:...]` marker, indicating that the agent attempted to cite a source?
2.  **Citation Match (Mocked)**: For each citation, does the `doc_id` reference an *expected* document ID from a mock knowledge base snippet? (In a real system, this would involve looking up the actual document content and verifying its relevance).
3.  **Fact Consistency (Proxy)**: Do key terms from the prompt appear in the "retrieved" snippets associated with citations? (This is a simplified proxy for actual fact-checking, which would typically involve semantic comparison).
4.  **Refusal Policy**: If the input contained high-risk instructions, did the agent's output indicate refusal or escalation instead of attempting to comply?

### Running Verification Checks

1.  **Navigate to "Verification Results":** In the sidebar, select "Verification Results".
2.  **Run Checks:** Click the **`Run Verification Checks on Last Simulation Output`** button.
    <aside class="negative">
    You must run a simulation (Step 4) *before* running verification checks, as verification operates on the outputs generated during the simulation.
    </aside>
3.  **Review Results:** After the checks complete, the "Verification Results Summary" table will populate.

### Interpreting Verification Results

The table displays each verification check that was run:

*   **`check_id`**: A unique identifier for the check.
*   **`check_type`**: The specific type of verification performed (e.g., `CITATION_PRESENCE`, `FACT_CONSISTENCY`, `REFUSAL_POLICY`).
*   **`status`**: The outcome of the check (`PASS` or `FAIL`).
*   **`description`**: A brief explanation of the check.
*   **`details`**: More information about the check's findings, especially useful for `FAIL` statuses.
*   **`related_audit_event_step` / `related_audit_event_action`**: Provides context by linking the verification result back to a specific step and action in the simulation audit log.

<aside class="positive">
The table uses color coding (green for `PASS`, red for `FAIL`) to quickly highlight issues. If any check `FAIL`s, an "Escalation Indicator" will appear, prompting you to review the details. This mechanism is crucial for identifying agent outputs that may be inaccurate, ungrounded, or non-compliant.
</aside>

## 6. Audit Log & Exports: Governance and Traceability
Duration: 08:00

This final section is where the AI Safety Engineer synthesizes the outcomes of their work. A comprehensive audit trail is provided, along with the ability to export all crucial information—policy definitions, verification outcomes, and detailed audit logs—in standardized formats. This documentation proves the LLM-powered compliance assistant operates within its defined safety and compliance guardrails, building trust and reducing operational risk.

### Simulation Summary Report

1.  **Navigate to "Audit Log & Exports":** In the sidebar, select "Audit Log & Exports".
2.  **Generate Summary Report:** Click the **`Generate Summary Report`** button. This will display key metrics from the last simulation and verification run:
    *   Total Agent Steps Simulated
    *   Policy Denials (Blocked Actions)
    *   Approval Requests Generated
    *   Total Verification Checks Run
    *   Verification Checks Passed
    *   Verification Checks Failed

    <aside class="positive">
    This summary provides an at-a-glance overview of how effectively your policies controlled the agent and how well its outputs met quality and safety criteria. Look for `Blocked Actions` (policy success) and `Verification Checks Failed` (areas for improvement).
    </aside>

### Raw Audit Log (JSONL)

Below the summary, you can view the complete, raw audit log from the last simulation. This log is a sequence of JSON objects, each representing an `AuditEvent`. It provides the highest level of detail for forensic analysis and debugging.

```json
[
  {
    "run_id": "...",
    "step_number": 1,
    "event_type": "LLM_THINKING",
    "policy_action": "ALLOWED",
    "...": "..."
  },
  {
    "run_id": "...",
    "step_number": 2,
    "event_type": "TOOL_SELECTED",
    "policy_action": "ALLOWED",
    "...": "..."
  }
]
```

### Export Audit Artifacts

This section is vital for formalizing the audit process and ensuring traceability.

1.  **Generate & Export All Artifacts:** Click the **`Generate & Export All Artifacts`** button.
    <aside class="negative">
    Ensure you have run both a simulation (Step 4) and verification checks (Step 5) before attempting to export all artifacts, otherwise, some files may be empty or incomplete.
    </aside>
2.  **Download ZIP:** After generation, a **`Download All Artifacts as ZIP`** button will appear. Clicking this will download a compressed archive containing several important files:
    *   `runtime_policy.json`: The specific policy definition used during the simulation.
    *   `verification_results.json`: Detailed outcomes of all verification checks.
    *   `audit_log.jsonl`: The complete, immutable log of all agent steps and policy decisions.
    *   `failure_mode_analysis.md`: A markdown document outlining potential ways the agent could fail.
    *   `residual_risk_summary.md`: A markdown document summarizing remaining risks and mitigation plans.
    *   `evidence_manifest.json`: A manifest linking inputs, outputs, and artifacts with cryptographic hashes to ensure their integrity and immutability.

    <aside class="positive">
    The use of cryptographic hashes in the `evidence_manifest.json` ensures the integrity of your audit trail. This means you can prove that the inputs, outputs, and other artifacts have not been tampered with since the time of export. This is a cornerstone of robust auditability.
    </aside>

3.  **Download Individual Artifacts:** You can also download each of the above files individually using the dedicated download buttons. This is useful if you only need a specific piece of documentation.

By completing this codelab, you've gained practical experience in defining, enforcing, verifying, and auditing LLM-powered agentic systems—essential skills for any AI Safety Engineer.
