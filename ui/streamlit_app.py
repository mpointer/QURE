"""
QURE Streamlit UI

Interactive dashboard for viewing reconciliation cases and agent decisions.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_vertical_labels(vertical):
    """Get vertical-specific field labels and terminology"""
    labels = {
        "Finance": {
            "data1_label": "GL Transaction",
            "data2_label": "Bank Transaction",
            "case_label": "Reconciliation",
            "id1_field": "id",
            "id2_field": "id",
            "amount_field": "amount",
            "date_field": "date",
            "entity_field": "payer",
            "memo_field": "memo",
        },
        "Healthcare": {
            "data1_label": "Prior Auth Request",
            "data2_label": "Clinical Documentation",
            "case_label": "Prior Authorization",
            "id1_field": "id",
            "id2_field": "id",
            "amount_field": "estimated_cost",
            "date_field": "request_date",
            "entity_field": "patient_name",
            "memo_field": "procedure_name",
        },
    }
    return labels.get(vertical, labels["Finance"])


def load_test_data(vertical="Finance"):
    """Load synthetic test data for the selected vertical"""
    # Map vertical to directory name and file structure
    vertical_config = {
        "Finance": {
            "dir": "finance",
            "files": ["gl_transactions.json", "bank_transactions.json", "expected_matches.json"],
            "keys": ["gl", "bank", "matches"]
        },
        "Healthcare": {
            "dir": "healthcare",
            "files": ["prior_auth_requests.json", "clinical_documentation.json", "expected_matches.json"],
            "keys": ["requests", "clinical", "matches"]
        },
        "Insurance": {
            "dir": "finance",  # Fallback to finance for now
            "files": ["gl_transactions.json", "bank_transactions.json", "expected_matches.json"],
            "keys": ["gl", "bank", "matches"]
        },
        "Retail": {
            "dir": "finance",  # Fallback to finance for now
            "files": ["gl_transactions.json", "bank_transactions.json", "expected_matches.json"],
            "keys": ["gl", "bank", "matches"]
        },
        "Manufacturing": {
            "dir": "finance",  # Fallback to finance for now
            "files": ["gl_transactions.json", "bank_transactions.json", "expected_matches.json"],
            "keys": ["gl", "bank", "matches"]
        },
    }

    config = vertical_config.get(vertical, vertical_config["Finance"])

    # Get absolute path to data directory
    data_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic" / config["dir"]

    # Debug: show the path
    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        st.info(f"Falling back to Finance data")
        data_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "finance"

    # Load the three data files
    try:
        with open(data_dir / config["files"][0]) as f:
            data1 = json.load(f)

        with open(data_dir / config["files"][1]) as f:
            data2 = json.load(f)

        with open(data_dir / config["files"][2]) as f:
            matches = json.load(f)

        return data1, data2, matches
    except Exception as e:
        st.error(f"Error loading {vertical} data: {e}")
        st.info("Falling back to Finance data")
        # Fallback to finance
        data_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "finance"
        with open(data_dir / "gl_transactions.json") as f:
            data1 = json.load(f)
        with open(data_dir / "bank_transactions.json") as f:
            data2 = json.load(f)
        with open(data_dir / "expected_matches.json") as f:
            matches = json.load(f)
        return data1, data2, matches


def main():
    st.set_page_config(
        page_title="QURE - Exception Resolution System",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 QURE - Exception Resolution System")
    st.markdown("**Multi-Agent AI for Back-Office Exception Resolution**")

    # Sidebar - Use Case Navigator
    st.sidebar.title("🎯 Use Case Navigator")

    # Use session state to persist vertical selection
    if 'vertical' not in st.session_state:
        st.session_state.vertical = "Finance"

    vertical_icons = {
        "Finance": "💰",
        "Insurance": "🛡️",
        "Healthcare": "🏥",
        "Retail": "🛒",
        "Manufacturing": "🏭"
    }

    vertical = st.sidebar.selectbox(
        "Select Vertical",
        options=list(vertical_icons.keys()),
        index=list(vertical_icons.keys()).index(st.session_state.vertical),
        format_func=lambda x: f"{vertical_icons[x]} {x}"
    )

    st.session_state.vertical = vertical

    # Show vertical-specific info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Vertical:** {vertical_icons[vertical]} {vertical}")

    vertical_descriptions = {
        "Finance": "GL↔Bank Reconciliation, Invoice Matching, Payment Processing",
        "Insurance": "Claims Adjudication, Policy Verification, Fraud Detection",
        "Healthcare": "Medical Coding, Claims Processing, Prior Authorization",
        "Retail": "Order Reconciliation, Inventory Audits, Returns Processing",
        "Manufacturing": "Quality Control, Supply Chain Exceptions, Compliance Checks"
    }

    st.sidebar.caption(vertical_descriptions[vertical])
    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Use Cases", "Live Processing", "Case List", "Case Details", "Audit Trail", "Admin Panel", "Agent Performance", "About"]
    )

    # Load data based on selected vertical
    try:
        data1, data2, expected_matches = load_test_data(vertical)
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        st.info(f"Please run: python data/synthetic/generate_{vertical.lower()}_data.py")
        return

    # For backward compatibility, use generic names
    # These will be interpreted differently based on vertical
    gl_transactions = data1  # Finance: GL, Healthcare: Prior Auth Requests
    bank_transactions = data2  # Finance: Bank, Healthcare: Clinical Documentation

    if page == "Dashboard":
        show_dashboard(gl_transactions, bank_transactions, expected_matches)
    elif page == "Use Cases":
        show_use_cases()
    elif page == "Live Processing":
        show_live_processing(gl_transactions, bank_transactions, expected_matches)
    elif page == "Case List":
        show_case_list(gl_transactions, bank_transactions, expected_matches)
    elif page == "Case Details":
        show_case_details(gl_transactions, bank_transactions, expected_matches)
    elif page == "Audit Trail":
        show_audit_trail(gl_transactions, bank_transactions, expected_matches)
    elif page == "Admin Panel":
        show_admin_panel()
    elif page == "Agent Performance":
        show_agent_performance()
    elif page == "About":
        show_about()


def show_dashboard(gl_transactions, bank_transactions, expected_matches):
    """Show dashboard with summary statistics"""
    vertical = st.session_state.get('vertical', 'Finance')

    vertical_icons = {
        "Finance": "💰",
        "Insurance": "🛡️",
        "Healthcare": "🏥",
        "Retail": "🛒",
        "Manufacturing": "🏭"
    }

    st.header(f"{vertical_icons.get(vertical, '💰')} {vertical} Dashboard")

    # Get vertical-specific labels
    labels = get_vertical_labels(vertical)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cases", len(expected_matches))

    with col2:
        auto_resolve = sum(1 for m in expected_matches if m["expected_decision"] == "auto_resolve")
        st.metric("Auto Resolved", auto_resolve, f"{auto_resolve/len(expected_matches)*100:.0f}%")

    with col3:
        hitl_review = sum(1 for m in expected_matches if m["expected_decision"] == "hitl_review")
        st.metric("HITL Review", hitl_review, f"{hitl_review/len(expected_matches)*100:.0f}%")

    with col4:
        rejected = sum(1 for m in expected_matches if m["expected_decision"] == "reject")
        st.metric("Rejected", rejected, f"{rejected/len(expected_matches)*100:.0f}%")

    # Interactive charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Decision Distribution")

        import plotly.graph_objects as go

        decision_counts = {}
        for match in expected_matches:
            decision = match["expected_decision"].replace("_", " ").title()
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        fig = go.Figure(data=[go.Pie(
            labels=list(decision_counts.keys()),
            values=list(decision_counts.values()),
            hole=0.4,
            marker_colors=['#4CAF50', '#FFC107', '#FF5252', '#2196F3']
        )])

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Match Score Distribution")

        scores = [m["match_score"] for m in expected_matches]

        fig = go.Figure(data=[go.Histogram(
            x=scores,
            nbinsx=10,
            marker_color='#2196F3',
            opacity=0.7
        )])

        fig.update_layout(
            xaxis_title="Match Score",
            yaxis_title="Count",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Recent cases
    st.subheader("Recent Cases")

    for i, match in enumerate(expected_matches[:5]):
        with st.expander(f"Case {match['case_id']} - {match['expected_decision'].replace('_', ' ').title()}"):
            col1, col2 = st.columns(2)

            with col1:
                # Use pa_id for Healthcare, gl_id for Finance
                id_field = "pa_id" if vertical == "Healthcare" else "gl_id"
                data1_id = match.get(id_field, match.get("gl_id"))
                data1 = next((g for g in gl_transactions if g.get("id") == data1_id), {})
                st.markdown(f"**{labels['data1_label']}**")
                st.json(data1)

            with col2:
                # Use clinical_id for Healthcare, bank_id for Finance
                id_field2 = "clinical_id" if vertical == "Healthcare" else "bank_id"
                data2_id = match.get(id_field2, match.get("bank_id"))
                data2 = next((b for b in bank_transactions if b.get("id") == data2_id), {})
                st.markdown(f"**{labels['data2_label']}**")
                st.json(data2)

            st.markdown(f"**Match Score:** {match['match_score']:.2%}")
            st.markdown(f"**Notes:** {match['notes']}")


def show_use_cases():
    """Show use cases for different verticals"""
    st.header("🎯 Use Cases by Vertical")

    vertical = st.session_state.get('vertical', 'Finance')

    # Use case definitions
    use_cases = {
        "Finance": {
            "icon": "💰",
            "title": "Finance & Accounting",
            "description": "Automated reconciliation and payment processing",
            "examples": [
                {
                    "name": "GL↔Bank Reconciliation",
                    "challenge": "Matching general ledger entries with bank transactions across thousands of daily transactions",
                    "solution": "Multi-agent system combines fuzzy matching, date proximity algorithms, and GenAI reasoning to identify matches with 95%+ accuracy",
                    "metrics": "40% auto-resolve rate, 60% reduction in processing time, 99.8% accuracy",
                    "status": "✅ Demo Available"
                },
                {
                    "name": "Invoice-to-Payment Matching",
                    "challenge": "Reconciling vendor invoices with payment confirmations across multiple payment channels",
                    "solution": "Rules engine validates invoice numbers, ML model scores payment likelihood, assurance agent detects anomalies",
                    "metrics": "Target: 50% auto-resolve, <2% false positive rate",
                    "status": "🔨 In Development"
                },
                {
                    "name": "Three-Way Match (PO-Invoice-Receipt)",
                    "challenge": "Matching purchase orders, invoices, and goods receipts with tolerance for pricing variances",
                    "solution": "Algorithm agent computes matching scores across 3 documents, policy agent applies tolerance thresholds",
                    "metrics": "Target: 65% auto-match, <1% variance threshold",
                    "status": "📋 Planned"
                }
            ]
        },
        "Insurance": {
            "icon": "🛡️",
            "title": "Insurance Claims & Underwriting",
            "description": "Intelligent claims adjudication and fraud detection",
            "examples": [
                {
                    "name": "Subrogation Recovery",
                    "challenge": "Identifying recovery opportunities from third-party liability claims",
                    "solution": "GenAI analyzes claim narratives, rules engine checks policy coverage, ML model scores recovery likelihood",
                    "metrics": "Target: 30% auto-identify, 20% increase in recovery",
                    "status": "📋 Planned"
                },
                {
                    "name": "Medical Claims Adjudication",
                    "challenge": "Validating medical necessity and coding accuracy for submitted claims",
                    "solution": "Rules validate CPT/ICD codes, GenAI reviews clinical notes, assurance agent flags inconsistencies",
                    "metrics": "Target: 40% auto-approve, 95% coding accuracy",
                    "status": "📋 Planned"
                },
                {
                    "name": "Fraud Detection",
                    "challenge": "Identifying suspicious claim patterns across large volumes",
                    "solution": "Graph database maps relationships, ML model scores anomalies, GenAI explains suspicious patterns",
                    "metrics": "Target: 80% fraud detection rate, <5% false positives",
                    "status": "📋 Planned"
                }
            ]
        },
        "Healthcare": {
            "icon": "🏥",
            "title": "Healthcare Operations",
            "description": "Clinical workflow automation and compliance",
            "examples": [
                {
                    "name": "Prior Authorization",
                    "challenge": "Reviewing prior auth requests against clinical guidelines and policy coverage",
                    "solution": "Rules engine checks medical necessity criteria, GenAI analyzes clinical documentation, assurance validates evidence",
                    "metrics": "Target: 35% auto-approve, 50% faster turnaround",
                    "status": "📋 Planned"
                },
                {
                    "name": "Medical Coding Validation",
                    "challenge": "Ensuring accurate ICD-10/CPT coding from clinical documentation",
                    "solution": "GenAI extracts diagnoses from notes, rules validate code combinations, ML suggests corrections",
                    "metrics": "Target: 90% coding accuracy, 60% reduction in denials",
                    "status": "📋 Planned"
                },
                {
                    "name": "Claims Denial Management",
                    "challenge": "Categorizing denial reasons and identifying appeal opportunities",
                    "solution": "GenAI analyzes denial codes and remittance advice, rules prioritize appeals, action agent generates responses",
                    "metrics": "Target: 40% overturn rate on appeals, 30% faster processing",
                    "status": "📋 Planned"
                }
            ]
        },
        "Retail": {
            "icon": "🛒",
            "title": "Retail Operations",
            "description": "Order management and inventory reconciliation",
            "examples": [
                {
                    "name": "Order-to-Cash Reconciliation",
                    "challenge": "Matching orders, shipments, and payments across channels",
                    "solution": "Algorithm agent matches order IDs, ML handles partial shipments, policy routes exceptions",
                    "metrics": "Target: 70% auto-match, 40% faster close",
                    "status": "📋 Planned"
                },
                {
                    "name": "Returns Authorization",
                    "challenge": "Validating return requests against policy and fraud indicators",
                    "solution": "Rules check return window and conditions, ML scores fraud risk, GenAI analyzes customer communication",
                    "metrics": "Target: 50% auto-approve, 15% reduction in fraud",
                    "status": "📋 Planned"
                },
                {
                    "name": "Inventory Variance Resolution",
                    "challenge": "Reconciling physical counts with system inventory across locations",
                    "solution": "Algorithm computes variances, ML identifies patterns, GenAI suggests root causes",
                    "metrics": "Target: 60% variance explanation, 25% shrinkage reduction",
                    "status": "📋 Planned"
                }
            ]
        },
        "Manufacturing": {
            "icon": "🏭",
            "title": "Manufacturing & Supply Chain",
            "description": "Quality control and compliance automation",
            "examples": [
                {
                    "name": "Quality Exception Resolution",
                    "challenge": "Analyzing quality inspection failures and routing for disposition",
                    "solution": "Rules classify defect types, ML predicts rework success, GenAI analyzes inspection notes",
                    "metrics": "Target: 45% auto-disposition, 20% reduction in scrap",
                    "status": "📋 Planned"
                },
                {
                    "name": "Supplier Invoice Reconciliation",
                    "challenge": "Matching supplier invoices with purchase orders and delivery receipts",
                    "solution": "Algorithm handles quantity/price variances, rules validate terms, policy routes discrepancies",
                    "metrics": "Target: 55% auto-match, 35% faster payment cycle",
                    "status": "📋 Planned"
                },
                {
                    "name": "Compliance Documentation Validation",
                    "challenge": "Verifying completeness of regulatory compliance documentation",
                    "solution": "GenAI extracts required data points, rules validate against standards, assurance checks evidence",
                    "metrics": "Target: 80% auto-validation, 99% compliance rate",
                    "status": "📋 Planned"
                }
            ]
        }
    }

    # Get current vertical use cases
    vertical_data = use_cases.get(vertical, use_cases["Finance"])

    # Header
    st.markdown(f"## {vertical_data['icon']} {vertical_data['title']}")
    st.markdown(f"**{vertical_data['description']}**")
    st.markdown("---")

    # Show examples
    for i, example in enumerate(vertical_data["examples"], 1):
        with st.expander(f"{i}. {example['name']} {example['status']}", expanded=(i == 1)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Challenge:**")
                st.markdown(example["challenge"])

                st.markdown(f"**Solution:**")
                st.markdown(example["solution"])

            with col2:
                st.markdown(f"**Expected Metrics:**")
                st.info(example["metrics"])

                st.markdown(f"**Status:**")
                if "Demo Available" in example["status"]:
                    st.success(example["status"])
                    if vertical == "Finance" and i == 1:
                        if st.button(f"▶️ Launch Demo", key=f"launch_{i}"):
                            st.session_state.page = "Dashboard"
                            st.rerun()
                elif "In Development" in example["status"]:
                    st.warning(example["status"])
                else:
                    st.caption(example["status"])

    # Architecture overview
    st.markdown("---")
    st.markdown("## 🏗️ QURE Architecture for All Verticals")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Layer 1: Retrieval**")
        st.markdown("""
        - Document ingestion
        - Entity extraction
        - Vector embeddings
        - Graph relationships
        """)

    with col2:
        st.markdown("**Layer 2: Reasoning**")
        st.markdown("""
        - Rules engine
        - Algorithms
        - ML models
        - GenAI reasoning
        """)

    with col3:
        st.markdown("**Layer 3: Decision**")
        st.markdown("""
        - Assurance validation
        - Policy routing
        - Action execution
        - Learning feedback
        """)

    # Non-negotiables
    st.markdown("---")
    st.markdown("## ✅ Non-Negotiables (All Verticals)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        1. **Citation Required**: Every LLM output must cite source documents
        2. **Calibrated Confidence**: All probabilities must be calibrated
        3. **Immutable Audit**: All decisions logged to immutable audit trail
        """)

    with col2:
        st.markdown("""
        4. **No Direct LLM Writes**: LLMs cannot write directly to systems of record
        5. **HITL by Design**: Human-in-the-loop for medium confidence cases
        6. **Explainable AI**: Every decision must be explainable to end users
        """)


def show_case_list(gl_transactions, bank_transactions, expected_matches):
    """Show list of all cases"""
    vertical = st.session_state.get('vertical', 'Finance')
    labels = get_vertical_labels(vertical)

    st.header(f"Case List - {labels['case_label']}")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        decision_filter = st.selectbox(
            "Filter by Decision",
            ["All"] + list(set(m["expected_decision"] for m in expected_matches))
        )

    with col2:
        score_filter = st.selectbox(
            "Filter by Score Range",
            ["All", "0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        )

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Case ID", "Match Score (High to Low)", "Match Score (Low to High)", "Date"]
        )

    # Apply filters
    filtered_matches = expected_matches

    if decision_filter != "All":
        filtered_matches = [m for m in filtered_matches if m["expected_decision"] == decision_filter]

    if score_filter != "All":
        min_score, max_score = map(float, score_filter.split("-"))
        filtered_matches = [m for m in filtered_matches if min_score <= m["match_score"] <= max_score]

    # Apply sorting
    if sort_by == "Match Score (High to Low)":
        filtered_matches = sorted(filtered_matches, key=lambda x: x["match_score"], reverse=True)
    elif sort_by == "Match Score (Low to High)":
        filtered_matches = sorted(filtered_matches, key=lambda x: x["match_score"])
    elif sort_by == "Date":
        filtered_matches = sorted(filtered_matches, key=lambda x: x["case_id"])

    # Display table
    st.markdown(f"**Showing {len(filtered_matches)} cases**")

    for match in filtered_matches:
        # Get data based on vertical
        if vertical == "Healthcare":
            data1_id = match.get("pa_id")
            data2_id = match.get("clinical_id")
        else:
            data1_id = match.get("gl_id")
            data2_id = match.get("bank_id")

        gl = next((g for g in gl_transactions if g.get("id") == data1_id), {})
        bank = next((b for b in bank_transactions if b.get("id") == data2_id), {})

        with st.expander(f"{match['case_id']} | Score: {match['match_score']:.2%} | Decision: {match['expected_decision'].replace('_', ' ').title()}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**{labels['data1_label']}**")
                st.markdown(f"**ID:** {gl.get('id', 'N/A')}")
                st.markdown(f"**Date:** {gl.get(labels['date_field'], 'N/A')}")
                st.markdown(f"**Amount:** ${gl.get(labels['amount_field'], 0):,.2f}")
                st.markdown(f"**Entity:** {gl.get(labels['entity_field'], 'N/A')}")
                st.markdown(f"**Detail:** {gl.get(labels['memo_field'], 'N/A')}")

            with col2:
                st.markdown(f"**{labels['data2_label']}**")
                st.markdown(f"**ID:** {bank.get('id', 'N/A')}")
                if vertical == "Healthcare":
                    st.markdown(f"**Date:** {bank.get('documentation_date', 'N/A')}")
                    st.markdown(f"**Necessity Score:** {bank.get('medical_necessity_score', 0):.2%}")
                    st.markdown(f"**Prior Treatments:** {len(bank.get('prior_treatments', []))}")
                    st.markdown(f"**Notes:** {bank.get('clinical_notes', 'N/A')[:100]}...")
                else:
                    st.markdown(f"**Date:** {bank.get('date', 'N/A')}")
                    st.markdown(f"**Amount:** ${bank.get('amount', 0):,.2f}")
                    st.markdown(f"**Payer:** {bank.get('payer', 'N/A')}")
                    st.markdown(f"**Memo:** {bank.get('memo', 'N/A')}")

            with col3:
                st.markdown("**Match Analysis**")
                st.markdown(f"**Score:** {match['match_score']:.2%}")
                st.markdown(f"**Decision:** {match['expected_decision'].replace('_', ' ').title()}")
                st.markdown(f"**Notes:** {match['notes']}")

                if st.button(f"View Details", key=f"details_{match['case_id']}"):
                    st.session_state["selected_case"] = match["case_id"]


def show_case_details(gl_transactions, bank_transactions, expected_matches):
    """Show detailed view of a single case"""
    vertical = st.session_state.get('vertical', 'Finance')
    labels = get_vertical_labels(vertical)

    st.header(f"Case Details - {labels['case_label']}")

    # Case selector
    case_ids = [m["case_id"] for m in expected_matches]
    selected_case_id = st.selectbox("Select Case", case_ids)

    match = next(m for m in expected_matches if m["case_id"] == selected_case_id)

    # Get data based on vertical
    if vertical == "Healthcare":
        data1_id = match.get("pa_id")
        data2_id = match.get("clinical_id")
    else:
        data1_id = match.get("gl_id")
        data2_id = match.get("bank_id")

    gl = next((g for g in gl_transactions if g.get("id") == data1_id), {})
    bank = next((b for b in bank_transactions if b.get("id") == data2_id), {})

    # Case header
    st.subheader(f"Case: {match['case_id']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match Score", f"{match['match_score']:.2%}")
    with col2:
        st.metric("Expected Decision", match['expected_decision'].replace('_', ' ').title())
    with col3:
        # Calculate date diff based on vertical
        try:
            if vertical == "Healthcare":
                date1_str = gl.get('request_date', '2024-01-01')
                date2_str = bank.get('documentation_date', '2024-01-01')
            else:
                date1_str = gl.get('date', '2024-01-01')
                date2_str = bank.get('date', '2024-01-01')

            date_diff = abs((
                __import__('datetime').datetime.strptime(date2_str, '%Y-%m-%d') -
                __import__('datetime').datetime.strptime(date1_str, '%Y-%m-%d')
            ).days)
            st.metric("Date Difference", f"{date_diff} days")
        except:
            st.metric("Date Difference", "N/A")

    # Transactions
    st.subheader("Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {labels['data1_label']}")
        st.json(gl)

    with col2:
        st.markdown(f"### {labels['data2_label']}")
        st.json(bank)

    # Agent Scores (simulated)
    st.subheader("Agent Scores")

    agent_scores = {
        "Rules Engine": 0.90 if match['match_score'] > 0.8 else 0.60,
        "Algorithm": match['match_score'],
        "ML Model": min(1.0, match['match_score'] + 0.05),
        "GenAI Reasoner": min(1.0, match['match_score'] + 0.02),
        "Assurance": 0.85 if match['match_score'] > 0.7 else 0.50,
    }

    for agent, score in agent_scores.items():
        st.progress(score, text=f"{agent}: {score:.2%}")

    # Policy Decision
    st.subheader("Policy Decision")

    st.markdown(f"**Decision:** {match['expected_decision'].replace('_', ' ').title()}")
    st.markdown(f"**Confidence:** {match['match_score']:.2%}")
    st.markdown(f"**Reasoning:** {match['notes']}")

    # Action Items
    st.subheader("Action Items")

    if match['expected_decision'] == "auto_approve":
        st.success("Action: Auto-approve reconciliation and update GL status")
    elif match['expected_decision'] == "human_review":
        st.warning("Action: Escalate to human reviewer for validation")
    elif match['expected_decision'] == "auto_reject":
        st.error("Action: Auto-reject and flag for investigation")
    elif match['expected_decision'] == "request_evidence":
        st.info("Action: Request additional evidence (e.g., SWIFT reference)")
    elif match['expected_decision'] == "escalate":
        st.warning("Action: Escalate to senior management for approval")


def show_agent_performance():
    """Show agent performance metrics"""
    st.header("Agent Performance")

    st.markdown("""
    ### Multi-Agent Architecture

    QURE uses 11 specialized agents working in concert:

    1. **Retriever Agent** - Data ingestion
    2. **Data Agent (UDI)** - Entity extraction & normalization
    3. **Rules Engine** - Business logic & compliance
    4. **Algorithm Agent** - Deterministic computations
    5. **ML Model Agent** - Predictive scoring
    6. **GenAI Reasoner** - LLM-powered analysis
    7. **Assurance Agent** - Uncertainty quantification
    8. **Policy Agent** - Decision fusion
    9. **Action Agent** - Execution
    10. **Orchestration Agent** - Workflow coordination
    11. **Learning Agent** - Continuous improvement (coming soon)
    """)

    # Performance metrics (simulated)
    st.subheader("Agent Accuracy")

    metrics = {
        "Rules Engine": 0.95,
        "Algorithm Agent": 0.88,
        "ML Model": 0.87,
        "GenAI Reasoner": 0.90,
        "Overall System": 0.92,
    }

    for agent, accuracy in metrics.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(accuracy, text=f"{agent}")
        with col2:
            st.metric("", f"{accuracy:.1%}")

    st.subheader("Decision Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Decisions", "20")
    with col2:
        st.metric("Correct Decisions", "18")
    with col3:
        st.metric("Accuracy", "90%")


def show_about():
    """Show about page"""
    st.header("About QURE")

    st.markdown("""
    ## QURE - Exception Resolution System

    **Q**uantifiable **U**ncertainty & **R**easoning **E**ngine

    ### Overview

    QURE is a multi-agent AI system designed to resolve back-office exceptions across multiple verticals:

    - **Finance**: GL-Bank reconciliation
    - **Insurance**: Subrogation recovery
    - **Healthcare**: Prior authorization
    - **Retail**: Returns processing
    - **Manufacturing**: Quality exceptions

    ### Architecture

    QURE combines multiple AI techniques in a unified framework:

    - **Rules-based reasoning** for compliance and business logic
    - **Algorithmic computations** for exact matching
    - **Machine learning** for pattern recognition
    - **Large language models** for natural language understanding
    - **Uncertainty quantification** for confidence calibration

    ### Non-Negotiables

    1. Every LLM output must have citations
    2. All probabilities must be calibrated
    3. Audit logs are immutable
    4. No direct writes from LLMs
    5. Human-in-the-loop by design

    ### Technology Stack

    - **Python 3.11+** with Poetry
    - **FastAPI** for REST API
    - **Streamlit** for UI
    - **PostgreSQL** for feature store
    - **Neo4j** for graph database
    - **ChromaDB** for vector store
    - **OpenAI/Anthropic** for LLMs
    - **XGBoost** for ML models
    - **Temporal** for workflow orchestration

    ### Team

    - **Developer**: Michael Pointer (mpointer@gmail.com)
    - **AI Assistant**: Claude Code

    ### Repository

    C:\\Users\\micha\\Documents\\GitHub\\QURE
    """)


def show_live_processing(gl_transactions, bank_transactions, expected_matches):
    """Show live case processing with animated agent workflow"""
    import time
    import plotly.graph_objects as go

    vertical = st.session_state.get('vertical', 'Finance')
    labels = get_vertical_labels(vertical)

    st.header("🚀 Live Case Processing")
    st.markdown(f"Process {labels['case_label'].lower()} cases in real-time and watch the multi-agent system in action")

    # Case selection
    col1, col2 = st.columns([2, 1])

    with col1:
        # Build case options based on vertical
        case_options = []
        for m in expected_matches:
            if vertical == "Healthcare":
                id1 = m.get('pa_id', m.get('gl_id', 'N/A'))
                id2 = m.get('clinical_id', m.get('bank_id', 'N/A'))
            else:
                id1 = m.get('gl_id', 'N/A')
                id2 = m.get('bank_id', 'N/A')
            case_options.append(f"{m['case_id']} - {id1} ↔ {id2}")

        selected_case_idx = st.selectbox("Select Case to Process", range(len(case_options)), format_func=lambda i: case_options[i])
        selected_match = expected_matches[selected_case_idx]

    with col2:
        process_button = st.button("▶️ Process Case", type="primary", use_container_width=True)
        batch_button = st.button("⚡ Batch Process All", use_container_width=True)

    # Agent workflow visualization
    st.subheader("Agent Execution Flow")

    agents = [
        {"name": "Retriever", "icon": "📥", "status": "pending"},
        {"name": "Data", "icon": "🔍", "status": "pending"},
        {"name": "Rules", "icon": "📋", "status": "pending"},
        {"name": "Algorithm", "icon": "🧮", "status": "pending"},
        {"name": "ML Model", "icon": "🤖", "status": "pending"},
        {"name": "GenAI", "icon": "🧠", "status": "pending"},
        {"name": "Assurance", "icon": "✅", "status": "pending"},
        {"name": "Policy", "icon": "⚖️", "status": "pending"},
        {"name": "Action", "icon": "📤", "status": "pending"},
    ]

    # Batch processing logic
    if batch_button:
        st.subheader("⚡ Batch Processing All Cases")

        batch_progress = st.progress(0)
        batch_status = st.empty()

        batch_results = []
        decisions_count = {"auto_resolve": 0, "hitl_review": 0, "reject": 0, "request_evidence": 0}

        for idx, match in enumerate(expected_matches):
            batch_status.text(f"Processing case {idx+1}/{len(expected_matches)}: {match['case_id']}")
            batch_progress.progress((idx + 1) / len(expected_matches))

            # Simulate quick processing
            time.sleep(0.1)

            decision = match["expected_decision"]
            decisions_count[decision] = decisions_count.get(decision, 0) + 1

            batch_results.append({
                "case_id": match["case_id"],
                "decision": decision,
                "score": match["match_score"]
            })

        batch_status.text("✅ Batch processing complete!")

        # Show batch results summary
        st.success(f"**Processed {len(expected_matches)} cases successfully!**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Auto Resolved", decisions_count.get("auto_resolve", 0))
        with col2:
            st.metric("HITL Review", decisions_count.get("hitl_review", 0))
        with col3:
            st.metric("Rejected", decisions_count.get("reject", 0))
        with col4:
            st.metric("Req. Evidence", decisions_count.get("request_evidence", 0))

        # Show results in table
        st.subheader("Batch Results")

        import pandas as pd
        df = pd.DataFrame(batch_results)
        df['decision'] = df['decision'].str.replace('_', ' ').str.title()
        df['score'] = df['score'].apply(lambda x: f"{x:.0%}")

        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="qure_batch_results.csv",
            mime="text/csv"
        )

    elif process_button:
        # Get data1 and data2 for this case (vertical-aware)
        if vertical == "Healthcare":
            data1_id = selected_match.get("pa_id")
            data2_id = selected_match.get("clinical_id")
        else:
            data1_id = selected_match.get("gl_id")
            data2_id = selected_match.get("bank_id")

        gl = next((g for g in gl_transactions if g.get("id") == data1_id), {})
        bank = next((b for b in bank_transactions if b.get("id") == data2_id), {})

        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create a container for agent cards
        agent_cols = st.columns(len(agents))
        agent_cards = {}
        for i, agent in enumerate(agents):
            with agent_cols[i]:
                agent_cards[agent["name"]] = st.empty()
                agent_cards[agent["name"]].markdown(f"""
                <div style='text-align: center; padding: 10px; border: 2px solid #ddd; border-radius: 10px; background-color: #f9f9f9;'>
                    <div style='font-size: 32px;'>{agent['icon']}</div>
                    <div style='font-size: 12px; font-weight: bold;'>{agent['name']}</div>
                    <div style='font-size: 10px; color: #666;'>⏸️ Pending</div>
                </div>
                """, unsafe_allow_html=True)

        # Results container
        results_container = st.container()

        # Simulate agent execution
        total_steps = len(agents)
        agent_results = {}

        for i, agent in enumerate(agents):
            # Update status
            status_text.text(f"⚡ Executing {agent['name']} Agent...")
            progress_bar.progress((i) / total_steps)

            # Update current agent card to processing
            agent_cards[agent["name"]].markdown(f"""
            <div style='text-align: center; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e8f5e9;'>
                <div style='font-size: 32px;'>{agent['icon']}</div>
                <div style='font-size: 12px; font-weight: bold;'>{agent['name']}</div>
                <div style='font-size: 10px; color: #4CAF50;'>⚡ Processing...</div>
            </div>
            """, unsafe_allow_html=True)

            # Simulate processing time
            time.sleep(0.3)

            # Generate mock results based on agent
            if agent["name"] == "Retriever":
                result = {"status": "success", "documents": 2}
            elif agent["name"] == "Data":
                result = {"status": "success", "entities": 8}
            elif agent["name"] == "Rules":
                result = {"status": "success", "score": 0.75, "passed": 9, "failed": 0}
            elif agent["name"] == "Algorithm":
                result = {"status": "success", "score": 0.82}
            elif agent["name"] == "ML Model":
                result = {"status": "success", "confidence": 0.91}
            elif agent["name"] == "GenAI":
                result = {"status": "success", "confidence": 0.88}
            elif agent["name"] == "Assurance":
                result = {"status": "success", "assurance": 0.85, "hallucination": False}
            elif agent["name"] == "Policy":
                result = {"status": "success", "decision": selected_match["expected_decision"]}
            elif agent["name"] == "Action":
                result = {"status": "success", "action": "reconciled"}

            agent_results[agent["name"]] = result

            # Update agent card to completed
            score_display = ""
            if "score" in result:
                score_display = f"<div style='font-size: 10px;'>Score: {result['score']:.0%}</div>"
            elif "confidence" in result:
                score_display = f"<div style='font-size: 10px;'>Confidence: {result['confidence']:.0%}</div>"

            agent_cards[agent["name"]].markdown(f"""
            <div style='text-align: center; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #c8e6c9;'>
                <div style='font-size: 32px;'>{agent['icon']}</div>
                <div style='font-size: 12px; font-weight: bold;'>{agent['name']}</div>
                <div style='font-size: 10px; color: #2E7D32;'>✓ Complete</div>
                {score_display}
            </div>
            """, unsafe_allow_html=True)

        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Processing Complete!")

        # Show detailed results
        with results_container:
            st.success(f"**Case {selected_match['case_id']} processed successfully!**")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Get amount field based on vertical
                amount_field = labels.get('amount_field', 'amount')
                amount1 = gl.get(amount_field, 0)
                amount2 = bank.get(amount_field, 0)
                st.metric(f"{labels['data1_label']} Amount", f"${amount1:,.2f}")
                st.metric(f"{labels['data2_label']} Amount", f"${amount2:,.2f}")

            with col2:
                st.metric("Match Score", f"{selected_match['match_score']:.0%}")
                st.metric("Assurance", f"{agent_results['Assurance']['assurance']:.0%}")

            with col3:
                decision = selected_match['expected_decision'].replace('_', ' ').title()
                st.metric("Decision", decision)
                st.metric("ML Confidence", f"{agent_results['ML Model']['confidence']:.0%}")

            # Show agent scores in a chart
            st.subheader("Agent Scores Breakdown")

            fig = go.Figure()

            agent_names = ["Rules", "Algorithm", "ML Model", "GenAI", "Assurance"]
            scores = [
                agent_results["Rules"]["score"],
                agent_results["Algorithm"]["score"],
                agent_results["ML Model"]["confidence"],
                agent_results["GenAI"]["confidence"],
                agent_results["Assurance"]["assurance"]
            ]

            fig.add_trace(go.Bar(
                x=agent_names,
                y=scores,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
                text=[f"{s:.0%}" for s in scores],
                textposition='auto',
            ))

            fig.update_layout(
                title="Agent Confidence Scores",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)


def show_audit_trail(gl_transactions, bank_transactions, expected_matches):
    """Show audit trail with timeline visualization"""
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    st.header("📜 Audit Trail")
    st.markdown("Complete history of agent decisions and actions")

    # Case selection
    case_options = ["All Cases"] + [f"{m['case_id']}" for m in expected_matches]
    selected_case = st.selectbox("Filter by Case", case_options)

    # Generate mock audit events
    audit_events = []

    for idx, match in enumerate(expected_matches):
        if selected_case != "All Cases" and match['case_id'] != selected_case:
            continue

        base_time = datetime.now() - timedelta(hours=24-idx)

        # Create event chain for this case
        events = [
            {"time": base_time, "agent": "Orchestrator", "action": "Case initiated", "case_id": match['case_id'], "status": "info"},
            {"time": base_time + timedelta(seconds=1), "agent": "Retriever", "action": "Retrieved 2 documents", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=2), "agent": "Data", "action": "Extracted 8 entities", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=3), "agent": "Rules", "action": f"Evaluated rules: {match['match_score']:.0%} score", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=4), "agent": "Algorithm", "action": f"Computed match: {match['match_score']:.0%}", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=5), "agent": "ML Model", "action": "Prediction: 0.91 confidence", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=6), "agent": "GenAI", "action": "Generated explanation with 5 citations", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=7), "agent": "Assurance", "action": "Validated: 0.85 assurance, no hallucination", "case_id": match['case_id'], "status": "success"},
            {"time": base_time + timedelta(seconds=8), "agent": "Policy", "action": f"Decision: {match['expected_decision'].replace('_', ' ').title()}", "case_id": match['case_id'], "status": "warning" if "review" in match['expected_decision'] else "success"},
            {"time": base_time + timedelta(seconds=9), "agent": "Action", "action": "Executed action", "case_id": match['case_id'], "status": "success"},
        ]

        audit_events.extend(events)

    # Sort by time
    audit_events.sort(key=lambda x: x['time'], reverse=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(audit_events))
    with col2:
        st.metric("Success", sum(1 for e in audit_events if e['status'] == 'success'))
    with col3:
        st.metric("Warnings", sum(1 for e in audit_events if e['status'] == 'warning'))
    with col4:
        st.metric("Cases Tracked", len(set(e['case_id'] for e in audit_events)))

    # Timeline visualization
    st.subheader("Event Timeline")

    # Create timeline chart
    fig = go.Figure()

    agents = list(set(e['agent'] for e in audit_events))
    colors = {'info': '#2196F3', 'success': '#4CAF50', 'warning': '#FFC107', 'error': '#FF5252'}

    for agent in agents:
        agent_events = [e for e in audit_events if e['agent'] == agent]

        fig.add_trace(go.Scatter(
            x=[e['time'] for e in agent_events],
            y=[e['agent'] for e in agent_events],
            mode='markers',
            name=agent,
            marker=dict(
                size=12,
                color=[colors.get(e['status'], '#666') for e in agent_events]
            ),
            text=[f"{e['case_id']}: {e['action']}" for e in agent_events],
            hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>'
        ))

    fig.update_layout(
        title="Agent Activity Timeline",
        xaxis_title="Time",
        yaxis_title="Agent",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Event list
    st.subheader("Recent Events")

    for event in audit_events[:20]:  # Show last 20 events
        status_emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}

        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 3, 3])

            with col1:
                st.text(event['time'].strftime("%H:%M:%S"))

            with col2:
                st.text(f"{status_emoji[event['status']]} {event['agent']}")

            with col3:
                st.text(event['case_id'])

            with col4:
                st.text(event['action'])

            st.divider()


def show_admin_panel():
    """Show admin panel for configuration"""
    import json

    st.header("⚙️ Admin Panel")
    st.markdown("Configure system parameters and agent settings")

    tab1, tab2, tab3, tab4 = st.tabs(["Agent Configuration", "Policy Thresholds", "Rule Sets", "System Settings"])

    with tab1:
        st.subheader("Agent Signal Weights")
        st.markdown("Adjust the weight of each agent in the fusion score")

        col1, col2 = st.columns(2)

        with col1:
            rules_weight = st.slider("Rules Engine Weight", 0.0, 1.0, 0.25, 0.05)
            algorithm_weight = st.slider("Algorithm Weight", 0.0, 1.0, 0.20, 0.05)
            ml_weight = st.slider("ML Model Weight", 0.0, 1.0, 0.20, 0.05)

        with col2:
            genai_weight = st.slider("GenAI Weight", 0.0, 1.0, 0.20, 0.05)
            assurance_weight = st.slider("Assurance Weight", 0.0, 1.0, 0.15, 0.05)

        total_weight = rules_weight + algorithm_weight + ml_weight + genai_weight + assurance_weight

        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights should sum to 1.0 (currently: {total_weight:.2f})")
        else:
            st.success(f"✅ Weights sum to {total_weight:.2f}")

        if st.button("💾 Save Agent Weights"):
            config = {
                "signal_weights": {
                    "rules": rules_weight,
                    "algorithm": algorithm_weight,
                    "ml": ml_weight,
                    "genai": genai_weight,
                    "assurance": assurance_weight
                }
            }
            st.success("Agent weights saved successfully!")
            with st.expander("View Configuration"):
                st.json(config)

    with tab2:
        st.subheader("Policy Decision Thresholds")
        st.markdown("Set thresholds for automatic decision routing")

        col1, col2 = st.columns(2)

        with col1:
            auto_resolve_threshold = st.slider("Auto-Resolve Threshold", 0.0, 1.0, 0.85, 0.05)
            st.caption("Cases above this score are automatically resolved")

            hitl_review_threshold = st.slider("HITL Review Threshold", 0.0, 1.0, 0.50, 0.05)
            st.caption("Cases above this score require human review")

        with col2:
            reject_threshold = st.slider("Reject Threshold", 0.0, 1.0, 0.30, 0.05)
            st.caption("Cases below this score are automatically rejected")

            assurance_threshold = st.slider("Min Assurance Required", 0.0, 1.0, 0.70, 0.05)
            st.caption("Minimum assurance score for auto-resolve")

        # Visualization
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[0, reject_threshold, reject_threshold, hitl_review_threshold, hitl_review_threshold, auto_resolve_threshold, auto_resolve_threshold, 1.0],
            y=[0, 0, 1, 1, 2, 2, 3, 3],
            fill='tozeroy',
            name='Decision Zones',
            line=dict(color='#4CAF50'),
            fillcolor='rgba(76, 175, 80, 0.2)'
        ))

        fig.update_layout(
            title="Decision Threshold Zones",
            xaxis_title="Score",
            yaxis=dict(
                tickmode='array',
                tickvals=[0.5, 1.5, 2.5],
                ticktext=['Reject', 'HITL Review', 'Auto-Resolve']
            ),
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

        if st.button("💾 Save Thresholds"):
            config = {
                "thresholds": {
                    "auto_resolve": auto_resolve_threshold,
                    "hitl_review": hitl_review_threshold,
                    "reject": reject_threshold,
                    "min_assurance": assurance_threshold
                }
            }
            st.success("Thresholds saved successfully!")
            with st.expander("View Configuration"):
                st.json(config)

    with tab3:
        st.subheader("Rule Set Management")
        st.markdown("Enable/disable specific rules in the rules engine")

        rules = [
            {"id": "FR_R1", "name": "Amount Match", "enabled": True, "severity": "mandatory"},
            {"id": "FR_R2", "name": "Date Proximity", "enabled": True, "severity": "mandatory"},
            {"id": "FR_R3", "name": "High Value Requires SWIFT", "enabled": True, "severity": "mandatory"},
            {"id": "FR_R4", "name": "Payer Match", "enabled": True, "severity": "optional"},
            {"id": "FR_R5", "name": "Duplicate Detection", "enabled": True, "severity": "mandatory"},
            {"id": "FR_R6", "name": "SOX Compliance", "enabled": True, "severity": "mandatory"},
            {"id": "FR_R7", "name": "Currency Match", "enabled": True, "severity": "mandatory"},
            {"id": "FR_R8", "name": "Business Day Check", "enabled": False, "severity": "optional"},
        ]

        st.markdown("#### Active Rules")

        for rule in rules:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.text(f"{rule['id']}: {rule['name']}")

            with col2:
                severity_color = "🔴" if rule['severity'] == "mandatory" else "🟡"
                st.text(f"{severity_color} {rule['severity'].title()}")

            with col3:
                rule['enabled'] = st.checkbox(f"Enabled###{rule['id']}", value=rule['enabled'], key=f"rule_{rule['id']}")

            with col4:
                if st.button("📝", key=f"edit_{rule['id']}"):
                    st.info(f"Edit rule {rule['id']}")

        if st.button("💾 Save Rule Configuration"):
            enabled_rules = [r['id'] for r in rules if r['enabled']]
            st.success(f"Saved {len(enabled_rules)} enabled rules!")
            with st.expander("View Configuration"):
                st.json({"enabled_rules": enabled_rules})

    with tab4:
        st.subheader("System Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Performance Settings")
            max_concurrent_cases = st.number_input("Max Concurrent Cases", 1, 100, 10)
            agent_timeout = st.number_input("Agent Timeout (seconds)", 1, 300, 30)
            retry_attempts = st.number_input("Retry Attempts", 0, 5, 3)

            st.markdown("#### Logging Settings")
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            enable_audit = st.checkbox("Enable Audit Logging", value=True)

        with col2:
            st.markdown("#### LLM Settings")
            default_llm_provider = st.selectbox("Default LLM Provider", ["OpenAI", "Anthropic", "Google"])
            default_model = st.text_input("Default Model", "gpt-4-turbo-preview")
            temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)

            st.markdown("#### Database Settings")
            enable_vector_store = st.checkbox("Enable Vector Store", value=True)
            enable_graph_store = st.checkbox("Enable Graph Store", value=True)

        if st.button("💾 Save System Settings"):
            config = {
                "performance": {
                    "max_concurrent_cases": max_concurrent_cases,
                    "agent_timeout": agent_timeout,
                    "retry_attempts": retry_attempts
                },
                "logging": {
                    "level": log_level,
                    "audit_enabled": enable_audit
                },
                "llm": {
                    "provider": default_llm_provider,
                    "model": default_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "database": {
                    "vector_store_enabled": enable_vector_store,
                    "graph_store_enabled": enable_graph_store
                }
            }
            st.success("System settings saved successfully!")
            with st.expander("View Configuration"):
                st.json(config)


if __name__ == "__main__":
    main()
