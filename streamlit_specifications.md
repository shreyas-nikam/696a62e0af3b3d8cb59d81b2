
# Streamlit Application Specification: LLM + Agentic System Risk Controls

## 1. Application Overview

### Purpose
This Streamlit application serves as a development blueprint for an AI Safety Engineering team to design, implement, and validate enterprise-grade runtime controls for LLM-powered assistants with agentic tool use. It simulates a real-world scenario where, following a production incident involving hallucinations and unauthorized tool actions, a robust solution is required to define and enforce agent runtime policies, focusing on tool usage, operational boundaries, and output verification.

### High-Level Story Flow

The application guides the AI Safety Engineer through a structured workflow:

1.  **System Setup**: The engineer begins by setting up the foundational environment, including defining the system (LLM assistant) and configuring mock knowledge base snippets.
2.  **Tool Registry**: They then meticulously define the agent's available tools, classifying them by type, noting side-effecting properties, and specifying argument schemas.
3.  **Policy Editor**: Next, the engineer translates organizational safety and compliance requirements into concrete runtime policies, setting limits on agent autonomy, controlling tool access, and defining handling for sensitive operations or keywords.
4.  **Simulation Runner**: With tools and policies in place, the engineer runs deterministic simulations of agent behavior against predefined plans and selected policies, observing policy enforcement in real-time.
5.  **Verification Results**: Following simulations, a verification harness is used to automatically check the quality and safety of agent outputs, including citation presence, fact consistency, and adherence to refusal policies.
6.  **Audit Log & Exports**: Finally, all simulation and verification data are consolidated into comprehensive audit logs and exportable artifacts, providing a detailed, immutable record for governance review and deterministic replay.

This workflow ensures that the AI Safety Engineer can proactively identify potential failure modes, enforce granular controls, validate agent outputs, and provide clear evidence of compliance and safety.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import json
import pandas as pd
import uuid
import datetime
import io
import zipfile
import base64
import os
import hashlib

# Import all necessary components directly from source.py
from source import (
    ToolType, PolicyAction, VerificationCheck, RunStatus,
    ToolDefinition, RuntimePolicy, AgentStep, VerificationResult, AuditEvent, EvidenceManifest,
    ToolRegistry, PolicyEngine, AgentSimulator, VerificationHarness,
    PLAN_LIBRARY, MOCK_KNOWLEDGE_BASE,
    strict_compliance_policy, permissive_exploration_policy,
    generate_report_summary, calculate_file_hash, export_artifacts,
    _global_tool_registry, _global_policy_engine, agent_simulator, verification_harness,
    mock_retrieve_docs_cell_22, mock_send_email_cell_22, mock_query_db_cell_22, mock_write_file_cell_22
)
```

### `st.session_state` Usage

`st.session_state` is used to persist crucial application data and user selections across interactions and "pages" (conditional rendering).

*   **Initialization (on first run):**
    ```python
    if 'initialized' not in st.session_state:
        st.session_state.page = 'System Setup'

        # Directly use the pre-initialized instances from source.py
        # These instances already contain the default tools, policies, etc. setup in source.py
        st.session_state.tool_registry = _global_tool_registry
        st.session_state.policy_engine = _global_policy_engine
        st.session_state.agent_simulator = agent_simulator
        st.session_state.verification_harness = verification_harness

        # Store initial policies for display and selection
        st.session_state.known_policies = {
            strict_compliance_policy.name: strict_compliance_policy,
            permissive_exploration_policy.name: permissive_exploration_policy
        }

        # User-editable knowledge base content
        st.session_state.knowledge_base_content = MOCK_KNOWLEDGE_BASE

        # Results of the last simulation run
        st.session_state.last_run_audit_events = []
        st.session_state.last_run_verification_results = []

        # Current selections for simulation
        st.session_state.selected_policy_name = strict_compliance_policy.name
        st.session_state.selected_plan_name = list(PLAN_LIBRARY.keys())[0] if PLAN_LIBRARY else None

        # Flag to indicate initial setup is complete
        st.session_state.initialized = True
    ```

*   **Updated:**
    *   `st.session_state.page`: Updated by the sidebar selectbox (`st.sidebar.selectbox`).
    *   `st.session_state.knowledge_base_content`: Updated via `st.text_area` in "System Setup" to allow users to modify the mock KB. This will also update `st.session_state.verification_harness.knowledge_base`.
    *   `st.session_state.tool_registry`: Updated via "Add/Edit Tool" forms in "Tool Registry". (Tools are added to the internal dictionary of the `ToolRegistry` instance).
    *   `st.session_state.known_policies`: Updated via "Create/Edit Policy" forms in "Policy Editor".
    *   `st.session_state.selected_policy_name`, `st.session_state.selected_plan_name`: Updated by `st.selectbox` widgets in "Simulation Runner".
    *   `st.session_state.last_run_audit_events`: Updated after each successful simulation run via `st.session_state.agent_simulator.run_plan()`.
    *   `st.session_state.last_run_verification_results`: Updated after running verification checks.

*   **Read Across Pages:**
    *   `st.session_state.page`: Determines which content section is displayed.
    *   `st.session_state.tool_registry`: Used in "Tool Registry" to display current tools, and implicitly by `st.session_state.agent_simulator`.
    *   `st.session_state.known_policies`: Used in "Policy Editor" to display/edit policies, and in "Simulation Runner" for policy selection.
    *   `st.session_state.selected_policy_name`, `st.session_state.selected_plan_name`: Used by "Simulation Runner" to configure the simulation.
    *   `st.session_state.last_run_audit_events`, `st.session_state.last_run_verification_results`: Displayed in "Simulation Runner" (trace), "Verification Results", and "Audit Log & Exports".
    *   `st.session_state.knowledge_base_content`: Used by `st.session_state.verification_harness`.

### Application Structure and Function Invocation

The application uses `st.sidebar.selectbox` to control conditional rendering of different content sections, simulating a multi-page experience.

```python
# Main Streamlit app entry point
def app():
    st.set_page_config(layout="wide", page_title="LLM Agent Risk Controls")

    # Session state initialization (as defined above)
    if 'initialized' not in st.session_state:
        # ... (session state initialization code as described in the 'Initialization' section)
        st.session_state.page = 'System Setup' # Default page
        
        # Initialize ToolRegistry and add default tools with their mock executors
        st.session_state.tool_registry = ToolRegistry()
        retrieve_docs_tool = ToolDefinition(name="Retrieve Compliance Documents", tool_type=ToolType.RETRIEVE_DOCS, description="Retrieves internal compliance documents based on a query.", is_side_effecting=False, args_schema={"query": {"type": "string", "description": "The search query for documents."}})
        send_email_tool = ToolDefinition(name="Send Email", tool_type=ToolType.SEND_EMAIL, description="Sends an email to specified recipients.", is_side_effecting=True, args_schema={"recipient": {"type": "string", "description": "Email recipient"}, "subject": {"type": "string", "description": "Email subject"}, "body": {"type": "string", "description": "Email body"}})
        query_db_tool = ToolDefinition(name="Query Internal Database", tool_type=ToolType.QUERY_DB, description="Executes a query against an internal database.", is_side_effecting=True, args_schema={"query": {"type": "string", "description": "SQL query to execute."}})
        write_file_tool = ToolDefinition(name="Write File to Storage", tool_type=ToolType.WRITE_FILE, description="Writes content to a file in the internal storage system.", is_side_effecting=True, args_schema={"filename": {"type": "string", "description": "Name of the file to write."}, "content": {"type": "string", "description": "Content to write to the file."}})
        calculate_tool = ToolDefinition(name="Perform Calculation", tool_type=ToolType.CALCULATE, description="Performs a mathematical calculation.", is_side_effecting=False, args_schema={"expression": {"type": "string", "description": "Mathematical expression to evaluate."}})

        st.session_state.tool_registry.add_tool(retrieve_docs_tool, mock_retrieve_docs_cell_22)
        st.session_state.tool_registry.add_tool(send_email_tool, mock_send_email_cell_22)
        st.session_state.tool_registry.add_tool(query_db_tool, mock_query_db_cell_22)
        st.session_state.tool_registry.add_tool(write_file_tool, mock_write_file_cell_22)
        st.session_state.tool_registry.add_tool(calculate_tool)

        st.session_state.policy_engine = PolicyEngine()
        st.session_state.known_policies = {
            strict_compliance_policy.name: strict_compliance_policy,
            permissive_exploration_policy.name: permissive_exploration_policy
        }
        st.session_state.agent_simulator = AgentSimulator(st.session_state.tool_registry, st.session_state.policy_engine)
        st.session_state.verification_harness = VerificationHarness(MOCK_KNOWLEDGE_BASE)
        st.session_state.knowledge_base_content = MOCK_KNOWLEDGE_BASE
        st.session_state.last_run_audit_events = []
        st.session_state.last_run_verification_results = []
        st.session_state.selected_policy_name = strict_compliance_policy.name
        st.session_state.selected_plan_name = list(PLAN_LIBRARY.keys())[0] if PLAN_LIBRARY else None
        st.session_state.initialized = True
    
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.selectbox(
        "Go to",
        ["System Setup", "Tool Registry", "Policy Editor", "Simulation Runner", "Verification Results", "Audit Log & Exports"],
        key="page"
    )

    st.title("LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability")

    # Conditional rendering based on page_selection
    if st.session_state.page == "System Setup":
        render_system_setup_page()
    elif st.session_state.page == "Tool Registry":
        render_tool_registry_page()
    elif st.session_state.page == "Policy Editor":
        render_policy_editor_page()
    elif st.session_state.page == "Simulation Runner":
        render_simulation_runner_page()
    elif st.session_state.page == "Verification Results":
        render_verification_results_page()
    elif st.session_state.page == "Audit Log & Exports":
        render_audit_log_exports_page()

# Call the app function
if __name__ == "__main__":
    app()
```

#### Detailed Page Specifications

##### Page: System Setup

*   **Header**:
    ```python
    st.header("1. System Setup: Initializing the Agent Environment")
    ```
*   **Purpose Markdown**:
    ```python
    st.markdown(f"As an AI Safety Engineer, your role is to ensure advanced LLM-powered agents operate within defined boundaries. Following a critical production incident, leadership has mandated a robust solution for defining and enforcing agent runtime policies. This section guides you through configuring agent capabilities, authoring dynamic policies, simulating agent behavior, and verifying its outputs, all while generating an immutable audit trail.")
    ```
*   **Knowledge Base Section**:
    ```python
    st.subheader("Knowledge Base Snippets")
    st.markdown(f"Configure the mock knowledge base snippets. These snippets are used by the `RETRIEVE_DOCS` tool and for verification checks (e.g., citation matching and fact consistency).")

    kb_json = st.text_area(
        "Edit Mock Knowledge Base (JSON)",
        json.dumps(st.session_state.knowledge_base_content, indent=2),
        height=300,
        key="kb_editor"
    )

    if st.button("Update Knowledge Base"):
        try:
            new_kb = json.loads(kb_json)
            st.session_state.knowledge_base_content = new_kb
            st.session_state.verification_harness.knowledge_base = new_kb # Update the harness instance
            st.success("Knowledge Base updated successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please correct the knowledge base content.")
    ```
    *   **Function Invocation**: `json.loads()` for parsing, `st.session_state.verification_harness.knowledge_base = new_kb` for updating the harness.

##### Page: Tool Registry

*   **Header**:
    ```python
    st.header("2. Tool Registry: Define Agent Capabilities")
    ```
*   **Purpose Markdown**:
    ```python
    st.markdown(f"As an AI Safety Engineer, your first concrete task after understanding the data models is to define the specific tools the compliance assistant can use. This involves classifying each tool by its type, determining if it has side effects (e.g., sending an email vs. retrieving a document), and specifying its argument schema. This meticulous definition forms the foundation for applying granular runtime policies.")
    st.markdown(r"$$ \text{{Tool Definition: }} T_i = \{{ \text{{id, name, type, description, is_side_effecting, args_schema, enabled}} \}} $$")
    st.markdown(r"where $T_i$ represents the $i$-th tool, and `is_side_effecting` is a boolean flag indicating if the tool modifies external state.")
    ```
*   **Display Tools**:
    ```python
    st.subheader("Configured Tools")
    tools_data = [tool.model_dump() for tool in st.session_state.tool_registry.list_tools()]
    df_tools = pd.DataFrame(tools_data)
    if not df_tools.empty:
        st.dataframe(df_tools.set_index('name'), use_container_width=True)
    else:
        st.info("No tools configured yet.")
    ```
*   **Add/Edit Tool Form**:
    ```python
    st.subheader("Add/Edit Tool")
    with st.form("tool_form"):
        tool_names = [tool.name for tool in st.session_state.tool_registry.list_tools()]
        selected_tool_name = st.selectbox("Select Tool to Edit (or 'New Tool')", ["New Tool"] + tool_names, key="edit_tool_selector")

        current_tool = None
        if selected_tool_name != "New Tool":
            current_tool = st.session_state.tool_registry.get_tool_by_name(selected_tool_name)
        
        # Default values for form fields
        default_name = current_tool.name if current_tool else ""
        default_type = current_tool.tool_type.value if current_tool else ToolType.RETRIEVE_DOCS.value
        default_description = current_tool.description if current_tool else ""
        default_side_effecting = current_tool.is_side_effecting if current_tool else False
        default_args_schema = json.dumps(current_tool.args_schema, indent=2) if current_tool and current_tool.args_schema else json.dumps({}, indent=2)
        default_enabled = current_tool.enabled if current_tool else True

        name = st.text_input("Tool Name", value=default_name)
        tool_type_enum = st.selectbox("Tool Type", [t.value for t in ToolType], index=[t.value for t in ToolType].index(default_type))
        description = st.text_area("Description", value=default_description)
        is_side_effecting = st.checkbox("Is Side-Effecting?", value=default_side_effecting)
        args_schema_str = st.text_area("Arguments Schema (JSON)", value=default_args_schema, height=150)
        enabled = st.checkbox("Enabled", value=default_enabled)

        submitted = st.form_submit_button("Save Tool")

        if submitted:
            try:
                args_schema = json.loads(args_schema_str)
                tool_type = ToolType(tool_type_enum)

                if current_tool: # Editing existing tool
                    current_tool.name = name
                    current_tool.tool_type = tool_type
                    current_tool.description = description
                    current_tool.is_side_effecting = is_side_effecting
                    current_tool.args_schema = args_schema
                    current_tool.enabled = enabled
                    st.session_state.tool_registry.add_tool(current_tool) # Re-add to update
                    st.success(f"Tool '{name}' updated successfully!")
                else: # Adding new tool
                    new_tool = ToolDefinition(
                        name=name,
                        tool_type=tool_type,
                        description=description,
                        is_side_effecting=is_side_effecting,
                        args_schema=args_schema,
                        enabled=enabled
                    )
                    st.session_state.tool_registry.add_tool(new_tool)
                    st.success(f"Tool '{name}' added successfully!")
                st.session_state.page = "Tool Registry" # Force re-render to update table
            except ValidationError as e:
                st.error(f"Validation Error: {e}")
            except json.JSONDecodeError:
                st.error("Invalid JSON for arguments schema.")
    ```
    *   **Function Invocation**: `json.loads()`, `ToolType()`, `ToolDefinition()`, `st.session_state.tool_registry.add_tool()`, `st.session_state.tool_registry.get_tool_by_name()`.

##### Page: Policy Editor

*   **Header**:
    ```python
    st.header("3. Policy Editor: Author Runtime Policies")
    ```
*   **Purpose Markdown**:
    ```python
    st.markdown(f"With the tools defined, the next crucial step for the AI Safety Engineer is to translate the organization's safety and compliance requirements into concrete `RuntimePolicy` instances. This involves setting limits on agent autonomy, controlling tool access, and defining how to handle sensitive operations or keywords. We will define strict and permissive policies for comparison.")
    st.markdown(r"$$ \text{{Runtime Policy: }} P = \{{ \text{{id, name, allowed_tools, max_steps, approval_for_side_effects, restricted_keywords, ...}} \}} $$")
    st.markdown(r"This definition allows us to enforce granular controls on agent behavior.")
    ```
*   **Display Policies**:
    ```python
    st.subheader("Defined Runtime Policies")
    policies_data = [policy.model_dump() for policy in st.session_state.known_policies.values()]
    df_policies = pd.DataFrame(policies_data)
    if not df_policies.empty:
        st.dataframe(df_policies.set_index('name'), use_container_width=True)
    else:
        st.info("No policies configured yet.")
    ```
*   **Create/Edit Policy Form**:
    ```python
    st.subheader("Create/Edit Policy")
    with st.form("policy_form"):
        policy_names = list(st.session_state.known_policies.keys())
        selected_policy_name = st.selectbox("Select Policy to Edit (or 'New Policy')", ["New Policy"] + policy_names, key="edit_policy_selector")

        current_policy = None
        if selected_policy_name != "New Policy":
            current_policy = st.session_state.known_policies[selected_policy_name]
        
        # Default values for form fields
        default_name = current_policy.name if current_policy else ""
        default_allowed_tool_types = [t.value for t in current_policy.allowed_tool_types] if current_policy else []
        default_max_steps = current_policy.max_steps if current_policy else 5
        default_max_side_effect_actions = current_policy.max_side_effect_actions if current_policy else 0
        default_require_approval = current_policy.require_approval_for_side_effects if current_policy else False
        default_restricted_keywords = ", ".join(current_policy.restricted_keywords) if current_policy else ""
        default_escalation = current_policy.escalation_on_verification_fail if current_policy else True

        name = st.text_input("Policy Name", value=default_name, disabled=(selected_policy_name != "New Policy"))
        allowed_tool_types_selected = st.multiselect("Allowed Tool Types", [t.value for t in ToolType], default=default_allowed_tool_types)
        max_steps = st.number_input("Maximum Steps", min_value=1, value=default_max_steps)
        max_side_effect_actions = st.number_input("Maximum Side-Effecting Actions", min_value=0, value=default_max_side_effect_actions)
        require_approval_for_side_effects = st.checkbox("Require Approval for Side-Effects", value=default_require_approval)
        restricted_keywords_str = st.text_area("Restricted Keywords (comma-separated)", value=default_restricted_keywords)
        escalation_on_verification_fail = st.checkbox("Escalate on Verification Fail", value=default_escalation)

        submitted = st.form_submit_button("Save Policy")

        if submitted:
            try:
                allowed_tool_types = [ToolType(t) for t in allowed_tool_types_selected]
                restricted_keywords = [k.strip() for k in restricted_keywords_str.split(',') if k.strip()]

                if selected_policy_name != "New Policy" and current_policy: # Editing existing policy
                    current_policy.allowed_tool_types = allowed_tool_types
                    current_policy.max_steps = max_steps
                    current_policy.max_side_effect_actions = max_side_effect_actions
                    current_policy.require_approval_for_side_effects = require_approval_for_side_effects
                    current_policy.restricted_keywords = restricted_keywords
                    current_policy.escalation_on_verification_fail = escalation_on_verification_fail
                    st.session_state.known_policies[current_policy.name] = current_policy # Update in dict
                    st.success(f"Policy '{name}' updated successfully!")
                else: # Adding new policy
                    new_policy = RuntimePolicy(
                        name=name,
                        allowed_tool_types=allowed_tool_types,
                        max_steps=max_steps,
                        max_side_effect_actions=max_side_effect_actions,
                        require_approval_for_side_effects=require_approval_for_side_effects,
                        restricted_keywords=restricted_keywords,
                        escalation_on_verification_fail=escalation_on_verification_fail
                    )
                    st.session_state.known_policies[new_policy.name] = new_policy
                    st.success(f"Policy '{name}' added successfully!")
                st.session_state.page = "Policy Editor" # Force re-render to update table
            except ValidationError as e:
                st.error(f"Validation Error: {e}")
            except ValueError as e:
                st.error(f"Error: {e}")
    ```
    *   **Function Invocation**: `ToolType()`, `RuntimePolicy()`. Updates `st.session_state.known_policies`.

##### Page: Simulation Runner

*   **Header**:
    ```python
    st.header("4. Simulation Runner: Test Policy Enforcement")
    ```
*   **Purpose Markdown**:
    ```python
    st.markdown(f"Now that we have defined our tools and policies, the AI Safety Engineer needs to construct the core components that will enforce these rules: the `PolicyEngine` and the `AgentSimulator`. The `PolicyEngine` will evaluate each agent action against the active policy, deciding whether to allow, deny, or require approval. The `AgentSimulator` will then mimic the LLM's multi-step decision-making process based on predefined plans, interacting with mocked tools and logging every action and policy decision.")
    st.markdown(r"**Policy Engine Evaluation Logic:**")
    st.markdown(r"Let $P$ be the active `RuntimePolicy` and $A_t$ be the `AgentStep` at time $t$.")
    st.markdown(r"1. **Tool Type Check**: If $A_t.\text{{selected_tool}}$ is not `None` and $A_t.\text{{selected_tool.tool_type}} \notin P.\text{{allowed_tool_types}}$, then the action is **DENIED**.")
    st.markdown(r"2. **Step Limit Check**: If $A_t.\text{{step_number}} > P.\text{{max_steps}}$, then the action is **DENIED**.")
    st.markdown(r"3. **Restricted Keywords Check**: If any keyword $k \in P.\text{{restricted_keywords}}$ is found in $A_t.\text{{planned_action}}$ or $A_t.\text{{tool_args}}$ (if applicable), then the action is **DENIED**.")
    st.markdown(r"4. **Side-Effect Approval Check**: If $A_t.\text{{selected_tool}}$ is not `None`, $A_t.\text{{selected_tool.is_side_effecting}}$ is `True`, and $P.\text{{require_approval_for_side_effects}}$ is `True`, then the action **REQUIRES_APPROVAL**.")
    st.markdown(r"5. **Side-Effect Count Check**: If $A_t.\text{{selected_tool}}$ is not `None`, $A_t.\text{{selected_tool.is_side_effecting}}$ is `True`, and the count of previous side-effecting actions in the current run exceeds $P.\text{{max_side_effect_actions}}$, then the action is **DENIED**.")
    st.markdown(r"6. Otherwise, the action is **ALLOWED**.")
    ```
*   **Simulation Controls**:
    ```python
    st.subheader("Run Simulation")
    selected_plan_name = st.selectbox(
        "Select Plan Template",
        list(PLAN_LIBRARY.keys()),
        key='selected_plan_name_runner' # Unique key for this page
    )
    selected_policy_name = st.selectbox(
        "Select Runtime Policy",
        list(st.session_state.known_policies.keys()),
        key='selected_policy_name_runner' # Unique key for this page
    )

    if st.button("Run Simulation"):
        st.session_state.selected_plan_name = selected_plan_name
        st.session_state.selected_policy_name = selected_policy_name

        policy_to_run = st.session_state.known_policies[selected_policy_name]
        try:
            with st.spinner(f"Running simulation for '{selected_plan_name}' with policy '{selected_policy_name}'..."):
                run_events = st.session_state.agent_simulator.run_plan(
                    st.session_state.selected_plan_name,
                    policy_to_run
                )
                st.session_state.last_run_audit_events = run_events
            st.success("Simulation Completed!")
            st.toast("Simulation Completed!")
        except ValueError as e:
            st.error(f"Simulation Error: {e}")
    ```
    *   **Function Invocation**: `st.session_state.agent_simulator.run_plan()`.
*   **Simulation Trace Display**:
    ```python
    st.subheader("Simulation Trace")
    if st.session_state.last_run_audit_events:
        df_trace = pd.DataFrame([event.model_dump() for event in st.session_state.last_run_audit_events])
        st.dataframe(df_trace, use_container_width=True)
    else:
        st.info("No simulation has been run yet. Please run a simulation above.")
    ```

##### Page: Verification Results

*   **Header**:
    ```python
    st.header("5. Verification Results: Validate Agent Outputs")
    ```
*   **Purpose Markdown**:
    ```python
    st.markdown(f"Runtime policies prevent an agent from *doing* harmful things, but they don't guarantee the *quality* or *accuracy* of the agent's outputs. As an AI Safety Engineer, you must also implement a `Verification Harness` to check the integrity of generated content, especially for compliance assistants. This includes verifying citations, fact consistency, and adherence to refusal policies.")
    st.markdown(r"**Verification Checks Logic:**")
    st.markdown(r"1. **Citation Presence**: Does the output include at least one `[DOC:...]` marker?")
    st.markdown(r"2. **Citation Match (Mocked)**: For each citation, does the `doc_id` reference an *expected* document ID from a mock knowledge base snippet? (In a real system, this would involve looking up the actual document content).")
    st.markdown(r"3. **Fact Consistency (Proxy)**: Do key terms from the prompt appear in the \"retrieved\" snippets associated with citations? (A simplified proxy for actual fact-checking).")
    st.markdown(r"4. **Refusal Policy**: If the input contained high-risk instructions, did the agent's output indicate refusal or escalation instead of attempting to comply?")
    ```
*   **Run Verification**:
    ```python
    st.subheader("Run Verification Checks")
    if st.button("Run Verification Checks on Last Simulation Output"):
        if not st.session_state.last_run_audit_events:
            st.warning("No simulation has been run. Please run a simulation first.")
        else:
            verification_results = []
            output_events = [e for e in st.session_state.last_run_audit_events if e.event_type == "OUTPUT_GENERATED"]

            if not output_events:
                st.info("No output generated in the last simulation to verify.")
            else:
                with st.spinner("Running verification checks..."):
                    for event in output_events:
                        output_text = event.payload.get("tool_result", "")
                        planned_action = event.payload.get("planned_action", "")
                        # Simplified key_terms and high_risk_phrases for example, in a real app these might be dynamic
                        key_terms = ["policy", "data", "compliance", "anonymized"]
                        high_risk_phrases = ["delete financial records", "wire transfer", "override financial controls"]

                        # Need to get the full input prompt that led to this output for refusal check accurately
                        # For now, using planned_action as a proxy for agent_input.
                        # In a more complex agent system, the full LLM prompt would be logged in audit events.
                        agent_input_proxy = planned_action + " " + json.dumps(event.payload.get("tool_args", {}))

                        # Placeholder for relevant_text. In a real system, this would be retrieved RAG docs.
                        relevant_text = ""
                        for doc_id, content in st.session_state.knowledge_base_content.items():
                            if f"[DOC:{doc_id}]" in output_text:
                                relevant_text = content
                                break

                        results_for_output = []
                        results_for_output.append(st.session_state.verification_harness.check_citation_presence(output_text))
                        results_for_output.append(st.session_state.verification_harness.check_citation_match(output_text))
                        results_for_output.append(st.session_state.verification_harness.check_fact_consistency(output_text, relevant_text, key_terms))
                        results_for_output.append(st.session_state.verification_harness.check_refusal_policy(agent_input_proxy, output_text, high_risk_phrases))
                        
                        # Add a reference to the audit event for context
                        for res in results_for_output:
                            res_dict = res.model_dump()
                            res_dict['related_audit_event_step'] = event.step_number
                            res_dict['related_audit_event_action'] = event.payload.get("planned_action")
                            verification_results.append(VerificationResult(**res_dict))

                st.session_state.last_run_verification_results = verification_results
                st.success("Verification checks completed!")
                st.toast("Verification checks completed!")
    ```
    *   **Function Invocation**: `st.session_state.verification_harness.check_citation_presence()`, `st.session_state.verification_harness.check_citation_match()`, `st.session_state.verification_harness.check_fact_consistency()`, `st.session_state.verification_harness.check_refusal_policy()`.
*   **Display Results**:
    ```python
    st.subheader("Verification Results Summary")
    if st.session_state.last_run_verification_results:
        df_verification = pd.DataFrame([res.model_dump() for res in st.session_state.last_run_verification_results])
        
        # Add color coding for PASS/FAIL
        def color_status(val):
            if val == 'PASS':
                return 'background-color: #d4edda' # Light green
            elif val == 'FAIL':
                return 'background-color: #f8d7da' # Light red
            else:
                return ''
        
        st.dataframe(df_verification.style.applymap(color_status, subset=['status']), use_container_width=True)

        # Escalation Indicator
        if any(res.status == "FAIL" for res in st.session_state.last_run_verification_results):
            st.warning("Escalation Indicator: One or more verification checks failed. Review details above.")
        else:
            st.info("All verification checks passed or were N/A for applicable outputs.")
    else:
        st.info("No verification results available. Please run verification checks after a simulation.")
    ```

##### Page: Audit Log & Exports

*   **Header**:
    ```python
    st.header("6. Audit Log & Exports: Governance and Traceability")
    ```
*   **Purpose Markdown**:
    ```python
    st.markdown(f"This final section is where the AI Safety Engineer synthesizes the outcomes of their work. A comprehensive audit trail is provided, along with the ability to export all crucial information—policy definitions, verification outcomes, and detailed audit logs—in standardized formats. This documentation proves the LLM-powered compliance assistant operates within its defined safety and compliance guardrails, building trust and reducing operational risk.")
    ```
*   **Summary Report**:
    ```python
    st.subheader("Simulation Summary Report")
    if st.session_state.last_run_audit_events or st.session_state.last_run_verification_results:
        st.button("Generate Summary Report")
        if st.session_state.get('generate_summary_clicked', False): # Use session_state to persist button click effect
            # generate_report_summary prints to console, we need to capture output or re-implement for streamlit
            # For this spec, we will re-display the summary metrics using streamlit components
            total_steps = len([e for e in st.session_state.last_run_audit_events if e.event_type in ["TOOL_SELECTED", "LLM_THINKING", "APPROVAL_REQUESTED"]])
            blocked_actions = len([e for e in st.session_state.last_run_audit_events if e.event_type == "TOOL_BLOCKED"])
            approval_requests = len([e for e in st.session_state.last_run_audit_events if e.event_type == "APPROVAL_REQUESTED"])
            verification_passes = sum(1 for res in st.session_state.last_run_verification_results if res.status == "PASS")
            verification_fails = sum(1 for res in st.session_state.last_run_verification_results if res.status == "FAIL")
            verification_na = sum(1 for res in st.session_state.last_run_verification_results if res.status == "N/A")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Agent Steps Simulated", total_steps)
            col2.metric("Policy Denials (Blocked Actions)", blocked_actions)
            col3.metric("Approval Requests Generated", approval_requests)
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Total Verification Checks Run", len(st.session_state.last_run_verification_results))
            col5.metric("Verification Checks Passed", verification_passes)
            col6.metric("Verification Checks Failed", verification_fails)

            if blocked_actions > 0:
                st.error("Critical: Policy engine successfully blocked unauthorized actions.")
            if approval_requests > 0:
                st.warning("Note: Agent successfully triggered approval flows for side-effecting actions.")
            if verification_fails > 0:
                st.error("Warning: Some verification checks failed, indicating potential issues in agent output quality or safety.")
            else:
                st.success("Success: All verification checks passed or were N/A for applicable outputs.")
        
            st.session_state.generate_summary_clicked = True
        else:
            st.session_state.generate_summary_clicked = False
            st.info("Click 'Generate Summary Report' to view an overview of the last simulation.")
    else:
        st.info("No data available to generate a summary report. Please run a simulation and verification first.")
    ```
    *   **Function Invocation**: `generate_report_summary` (implicitly, by displaying metrics extracted from `st.session_state.last_run_audit_events` and `st.session_state.last_run_verification_results`).

*   **Raw Audit Log Display**:
    ```python
    st.subheader("Raw Audit Log (JSONL)")
    if st.session_state.last_run_audit_events:
        audit_log_jsonl = "\n".join([event.model_dump_json() for event in st.session_state.last_run_audit_events])
        st.json([event.model_dump(mode='json') for event in st.session_state.last_run_audit_events])
    else:
        st.info("No audit log available from the last simulation.")
    ```

*   **Export Artifacts**:
    ```python
    st.subheader("Export Audit Artifacts")
    if st.button("Generate & Export All Artifacts"):
        if not st.session_state.last_run_audit_events or not st.session_state.last_run_verification_results:
            st.warning("Please run a simulation and verification first to generate artifacts.")
        else:
            output_dir = "temp_artifacts"
            os.makedirs(output_dir, exist_ok=True)

            # Use selected policy for the policy file
            current_policy = st.session_state.known_policies.get(st.session_state.selected_policy_name)
            if not current_policy:
                st.error("No policy selected or found to export. Please select a policy.")
                return

            export_artifacts(
                current_policy,
                st.session_state.last_run_verification_results,
                st.session_state.last_run_audit_events,
                output_dir=output_dir
            )
            st.success(f"All audit artifacts generated successfully in '{output_dir}' directory locally.")

            # Create a ZIP file for download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, output_dir))
            
            zip_buffer.seek(0)
            st.download_button(
                label="Download All Artifacts as ZIP",
                data=zip_buffer,
                file_name="audit_artifacts.zip",
                mime="application/zip",
                key="download_all_artifacts"
            )
            # Clean up temporary directory
            for root, _, files in os.walk(output_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.rmdir(output_dir)
            st.info("Temporary artifacts cleaned up locally.")

    # Individual download buttons (if files exist) - simplified check
    if st.session_state.last_run_audit_events: # Only show if some simulation ran
        st.markdown("---")
        st.markdown("### Download Individual Artifacts (Last Generated)")

        # Placeholder for dynamic paths; in actual app, these would come from export_artifacts or session state
        # For this blueprint, assume they were generated into a temporary 'temp_artifacts' folder
        # For simplicity, we directly generate content for download buttons.

        # runtime_policy.json
        current_policy_for_download = st.session_state.known_policies.get(st.session_state.selected_policy_name)
        if current_policy_for_download:
            policy_json = current_policy_for_download.model_dump_json(indent=2)
            st.download_button(
                label="Download runtime_policy.json",
                data=policy_json,
                file_name="runtime_policy.json",
                mime="application/json"
            )

        # verification_results.json
        if st.session_state.last_run_verification_results:
            verification_json = json.dumps([res.model_dump(mode='json') for res in st.session_state.last_run_verification_results], indent=2)
            st.download_button(
                label="Download verification_results.json",
                data=verification_json,
                file_name="verification_results.json",
                mime="application/json"
            )

        # audit_log.jsonl
        if st.session_state.last_run_audit_events:
            audit_log_content = "\n".join([event.model_dump_json() for event in st.session_state.last_run_audit_events])
            st.download_button(
                label="Download audit_log.jsonl",
                data=audit_log_content,
                file_name="audit_log.jsonl",
                mime="application/jsonl"
            )
        
        # failure_mode_analysis.md and residual_risk_summary.md need placeholder content
        # For the purpose of this spec, provide generic content directly or refer to source.py's content
        fma_content = """# Failure Mode Analysis
## Introduction
This document analyzes potential failure modes for the Compliance Assistant agent based on simulation results and policy enforcement.
... (content from source.py) ...
"""
        st.download_button(
            label="Download failure_mode_analysis.md",
            data=fma_content,
            file_name="failure_mode_analysis.md",
            mime="text/markdown"
        )

        rrs_content = """# Residual Risk Summary and Mitigation Plan
## Introduction
This document outlines residual risks associated with the Compliance Assistant agent even after implementing runtime constraints and verification, along with proposed mitigation plans.
... (content from source.py) ...
"""
        st.download_button(
            label="Download residual_risk_summary.md",
            data=rrs_content,
            file_name="residual_risk_summary.md",
            mime="text/markdown"
        )
        
        # evidence_manifest.json
        if st.session_state.last_run_audit_events and current_policy_for_download:
            # Need to re-calculate hashes for manifest for the download button context
            # This is a simplification; in a real app, `export_artifacts` would provide the manifest content.
            mock_inputs_hash = hashlib.sha256(json.dumps(PLAN_LIBRARY, sort_keys=True).encode()).hexdigest()
            mock_outputs_hash = hashlib.sha256(json.dumps([e.model_dump(mode='json') for e in st.session_state.last_run_audit_events], sort_keys=True).encode()).hexdigest()
            
            # This part is complex because `export_artifacts` creates actual files to hash.
            # For direct download, we'd need to create artifact hashes on-the-fly or save them temporarily.
            # A simpler approach for the spec: just provide a placeholder manifest or indicate the blueprint needs actual file paths/hashes from `export_artifacts`.
            
            # For the spec, we will assume `export_artifacts` makes the artifact hashes available via a temporary mechanism.
            # Or, for the download button, we create a simplified manifest placeholder.
            # Best approach: `export_artifacts` could return the `EvidenceManifest` object which can then be used.
            # Let's assume `export_artifacts` function is modified to return the manifest.
            
            # If `export_artifacts` is not modified, we need to create temp files to hash for the manifest.
            # A simpler way for a blueprint: prompt for export first, then enable specific downloads.
            manifest_placeholder = EvidenceManifest(
                run_id=st.session_state.last_run_audit_events[0].run_id if st.session_state.last_run_audit_events else uuid.uuid4(),
                inputs_hash=mock_inputs_hash,
                outputs_hash=mock_outputs_hash,
                artifacts={"placeholder.txt": "abcde"} # Placeholder artifact hashes
            ).model_dump_json(indent=2)
            
            st.download_button(
                label="Download evidence_manifest.json",
                data=manifest_placeholder,
                file_name="evidence_manifest.json",
                mime="application/json"
            )
    ```
    *   **Function Invocation**: `export_artifacts()`, `generate_report_summary()` (conceptually). `os.makedirs()`, `os.remove()`, `os.rmdir()`, `io.BytesIO()`, `zipfile.ZipFile()`.

