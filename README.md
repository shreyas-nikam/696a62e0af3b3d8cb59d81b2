# QuLab: Case Study 4: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability

![Streamlit App Screenshot Placeholder](https://via.placeholder.com/1200x600?text=QuLab+Streamlit+Application+Screenshot)
*(Replace this with an actual screenshot of the running application)*

## 🚀 Project Title and Description

**QuLab: Case Study 4: LLM + Agentic System Risk Controls: Runtime Constraints, Verification, and Auditability**

This Streamlit application serves as an interactive lab environment for exploring and implementing robust risk controls for LLM-powered agentic systems. Developed as part of QuantUniversity's QuLab series, this case study focuses on critical aspects of AI Safety Engineering: defining agent capabilities, authoring dynamic runtime policies, simulating agent behavior under these policies, verifying the quality and safety of agent outputs, and generating a comprehensive, immutable audit trail.

The application allows AI Safety Engineers to:
*   Define and manage a registry of tools an agent can utilize.
*   Create and enforce sophisticated runtime policies to constrain agent actions.
*   Simulate agent workflows against various policies and pre-defined plans.
*   Conduct post-execution verification checks on agent outputs (e.g., citation presence, fact consistency, refusal policy adherence).
*   Generate detailed audit logs and export all relevant artifacts for governance, traceability, and compliance.

This project is designed to help understand and mitigate the risks associated with autonomous AI systems, ensuring they operate within desired ethical and operational boundaries.

## ✨ Features

The application is structured around a multi-page Streamlit interface, offering the following key functionalities:

1.  **System Setup**:
    *   Initialize the agent environment and configure a mock knowledge base used for tool interaction and verification.
    *   Dynamically update the knowledge base content.

2.  **Tool Registry**:
    *   **Define Agent Capabilities**: Register and manage various tools that an agent can invoke (e.g., `RETRIEVE_DOCS`, `SEND_EMAIL`, `QUERY_DB`).
    *   **Tool Properties**: Specify `ToolType`, description, `is_side_effecting` status, and `args_schema` for each tool.
    *   $$ \text{Tool Definition: } T_i = \{ \text{id, name, type, description, is_side_effecting, args_schema, enabled} \} $$

3.  **Policy Editor**:
    *   **Author Runtime Policies**: Create and modify `RuntimePolicy` instances to govern agent behavior.
    *   **Granular Controls**: Configure policies with parameters such as `allowed_tool_types`, `max_steps`, `max_side_effect_actions`, `require_approval_for_side_effects`, `restricted_keywords`, and `escalation_on_verification_fail`.
    *   $$ \text{Runtime Policy: } P = \{ \text{id, name, allowed_tools, max_steps, approval_for_side_effects, restricted_keywords, ...} \} $$

4.  **Simulation Runner**:
    *   **Test Policy Enforcement**: Run agent simulations based on pre-defined `PLAN_LIBRARY` scenarios and selected `RuntimePolicy` configurations.
    *   **Policy Engine Logic**: Demonstrates how the `PolicyEngine` evaluates each `AgentStep` against the active policy (tool type, step limit, restricted keywords, side-effect approval/count checks).
    *   **Simulation Trace**: View a detailed, step-by-step audit log of the agent's actions and policy decisions during the simulation.

5.  **Verification Results**:
    *   **Validate Agent Outputs**: Execute `VerificationHarness` checks on the outputs generated during the last simulation run.
    *   **Comprehensive Checks**: Includes `Citation Presence`, `Citation Match (Mocked)`, `Fact Consistency (Proxy)`, and `Refusal Policy` checks.
    *   **Escalation Indicator**: Highlights failed verification checks, signaling potential issues in agent output quality or safety.

6.  **Audit Log & Exports**:
    *   **Governance and Traceability**: Synthesize simulation results into a summary report.
    *   **Raw Audit Log**: Display the complete audit log in JSONL format.
    *   **Artifact Export**: Generate and download a ZIP file containing all audit artifacts, including:
        *   `runtime_policy.json`: The active policy used for the simulation.
        *   `verification_results.json`: Detailed outcomes of all verification checks.
        *   `audit_log.jsonl`: The complete sequential log of agent steps and policy decisions.
        *   `failure_mode_analysis.md`: A template for analyzing potential failure modes.
        *   `residual_risk_summary.md`: A template for summarizing residual risks and mitigation plans.
        *   `evidence_manifest.json`: A manifest linking inputs, outputs, and artifacts via hashes for integrity and immutability.

## 🚀 Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/quslab-case-study-4.git
    cd quslab-case-study-4
    ```
    *(Note: Replace `your-username/quslab-case-study-4` with the actual repository path.)*

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    pydantic>=2.0.0
    # Add any other specific versions if needed
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

### Project Files

Ensure you have the following files in your project directory:

*   `app.py`: The main Streamlit application logic and UI.
*   `source.py`: Contains all the core data models (e.g., `ToolDefinition`, `RuntimePolicy`, `AuditEvent`), `ToolRegistry`, `PolicyEngine`, `AgentSimulator`, `VerificationHarness`, mock data (`PLAN_LIBRARY`, `MOCK_KNOWLEDGE_BASE`), and utility functions.
*   `requirements.txt`: Lists Python dependencies.

## 🏃‍♀️ Usage

1.  **Run the Streamlit application**:
    Navigate to the project directory in your terminal and execute:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser (usually at `http://localhost:8501`).

2.  **Navigate through the application**:
    Use the sidebar to switch between different sections of the lab project:
    *   **System Setup**: Review and modify the mock knowledge base.
    *   **Tool Registry**: Examine the pre-defined tools and their properties, or add/edit new ones.
    *   **Policy Editor**: Explore the `strict_compliance_policy` and `permissive_exploration_policy`, or create your own custom policies.
    *   **Simulation Runner**: Select a `Plan Template` and a `Runtime Policy`, then click "Run Simulation" to observe agent behavior and policy enforcement.
    *   **Verification Results**: After running a simulation, click "Run Verification Checks" to evaluate the agent's outputs.
    *   **Audit Log & Exports**: View the simulation summary, raw audit log, and download all generated artifacts.

3.  **Experiment and Learn**:
    *   Try running the "High Risk Data Access" plan with the "Strict Compliance Policy" versus the "Permissive Exploration Policy" to see the difference in policy enforcement.
    *   Modify policies (e.g., add new restricted keywords, change `max_steps`, toggle `require_approval_for_side_effects`) and re-run simulations to observe their impact.
    *   Experiment with different verification checks and understand how they contribute to agent safety and output quality.

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application with UI and page rendering logic
├── source.py               # Core logic: data models, registries, engines, simulators, harness, mock data, utilities
├── requirements.txt        # Python dependencies
└── README.md               # This README file
```

## 🛠️ Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web user interface.
*   **Pydantic**: Used extensively for defining clear, validated data models (e.g., `ToolDefinition`, `RuntimePolicy`, `AuditEvent`, `VerificationResult`).
*   **Pandas**: For displaying tabular data, especially in the audit logs and verification results.
*   **Standard Library**: `json`, `uuid`, `datetime`, `io`, `zipfile`, `base64`, `os`, `hashlib` for various utility functions like data serialization, unique ID generation, file operations, and hashing.

## 🤝 Contributing

This project is primarily a lab environment for learning. However, if you find issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

1.  **Fork** the repository.
2.  **Clone** your forked repository.
3.  **Create a new branch** (`git checkout -b feature/your-feature`).
4.  **Make your changes**.
5.  **Commit** your changes (`git commit -m 'Add new feature'`).
6.  **Push** to your branch (`git push origin feature/your-feature`).
7.  **Open a Pull Request**.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You might need to create a `LICENSE` file in your repository if you haven't already.)*

## 📞 Contact

For questions or feedback, please reach out to:

*   **QuantUniversity**: [info@quantuniversity.com](mailto:info@quantuniversity.com)
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com)

---
Enjoy exploring the world of LLM Agentic System Risk Controls!

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
