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


def load_test_data():
    """Load synthetic test data"""
    # Get absolute path to data directory
    data_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "finance"

    # Debug: show the path
    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        st.info(f"__file__ = {__file__}")
        st.info(f"parent = {Path(__file__).parent}")
        st.info(f"parent.parent = {Path(__file__).parent.parent}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    with open(data_dir / "gl_transactions.json") as f:
        gl_transactions = json.load(f)

    with open(data_dir / "bank_transactions.json") as f:
        bank_transactions = json.load(f)

    with open(data_dir / "expected_matches.json") as f:
        expected_matches = json.load(f)

    return gl_transactions, bank_transactions, expected_matches


def main():
    st.set_page_config(
        page_title="QURE - Exception Resolution System",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 QURE - Exception Resolution System")
    st.markdown("**Multi-Agent AI for Back-Office Exception Resolution**")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Live Processing", "Case List", "Case Details", "Agent Performance", "About"]
    )

    # Load data
    try:
        gl_transactions, bank_transactions, expected_matches = load_test_data()
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        st.info("Please run: python data/synthetic/generate_finance_data.py")
        return

    if page == "Dashboard":
        show_dashboard(gl_transactions, bank_transactions, expected_matches)
    elif page == "Live Processing":
        show_live_processing(gl_transactions, bank_transactions, expected_matches)
    elif page == "Case List":
        show_case_list(gl_transactions, bank_transactions, expected_matches)
    elif page == "Case Details":
        show_case_details(gl_transactions, bank_transactions, expected_matches)
    elif page == "Agent Performance":
        show_agent_performance()
    elif page == "About":
        show_about()


def show_dashboard(gl_transactions, bank_transactions, expected_matches):
    """Show dashboard with summary statistics"""
    st.header("Dashboard")

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
                gl = next(g for g in gl_transactions if g["id"] == match["gl_id"])
                st.markdown("**GL Transaction**")
                st.json(gl)

            with col2:
                bank = next(b for b in bank_transactions if b["id"] == match["bank_id"])
                st.markdown("**Bank Transaction**")
                st.json(bank)

            st.markdown(f"**Match Score:** {match['match_score']:.2%}")
            st.markdown(f"**Notes:** {match['notes']}")


def show_case_list(gl_transactions, bank_transactions, expected_matches):
    """Show list of all cases"""
    st.header("Case List")

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
        gl = next(g for g in gl_transactions if g["id"] == match["gl_id"])
        bank = next(b for b in bank_transactions if b["id"] == match["bank_id"])

        with st.expander(f"{match['case_id']} | Score: {match['match_score']:.2%} | Decision: {match['expected_decision'].replace('_', ' ').title()}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**GL Transaction**")
                st.markdown(f"**ID:** {gl['id']}")
                st.markdown(f"**Date:** {gl['date']}")
                st.markdown(f"**Amount:** ${gl['amount']:,.2f}")
                st.markdown(f"**Payer:** {gl['payer']}")
                st.markdown(f"**Memo:** {gl['memo']}")

            with col2:
                st.markdown("**Bank Transaction**")
                st.markdown(f"**ID:** {bank['id']}")
                st.markdown(f"**Date:** {bank['date']}")
                st.markdown(f"**Amount:** ${bank['amount']:,.2f}")
                st.markdown(f"**Payer:** {bank['payer']}")
                st.markdown(f"**Memo:** {bank['memo']}")

            with col3:
                st.markdown("**Match Analysis**")
                st.markdown(f"**Score:** {match['match_score']:.2%}")
                st.markdown(f"**Decision:** {match['expected_decision'].replace('_', ' ').title()}")
                st.markdown(f"**Notes:** {match['notes']}")

                if st.button(f"View Details", key=f"details_{match['case_id']}"):
                    st.session_state["selected_case"] = match["case_id"]


def show_case_details(gl_transactions, bank_transactions, expected_matches):
    """Show detailed view of a single case"""
    st.header("Case Details")

    # Case selector
    case_ids = [m["case_id"] for m in expected_matches]
    selected_case_id = st.selectbox("Select Case", case_ids)

    match = next(m for m in expected_matches if m["case_id"] == selected_case_id)
    gl = next(g for g in gl_transactions if g["id"] == match["gl_id"])
    bank = next(b for b in bank_transactions if b["id"] == match["bank_id"])

    # Case header
    st.subheader(f"Case: {match['case_id']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match Score", f"{match['match_score']:.2%}")
    with col2:
        st.metric("Expected Decision", match['expected_decision'].replace('_', ' ').title())
    with col3:
        date_diff = abs((
            __import__('datetime').datetime.strptime(bank['date'], '%Y-%m-%d') -
            __import__('datetime').datetime.strptime(gl['date'], '%Y-%m-%d')
        ).days)
        st.metric("Date Difference", f"{date_diff} days")

    # Transactions
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### GL Transaction")
        st.json(gl)

    with col2:
        st.markdown("### Bank Transaction")
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

    st.header("🚀 Live Case Processing")
    st.markdown("Process reconciliation cases in real-time and watch the multi-agent system in action")

    # Case selection
    col1, col2 = st.columns([2, 1])

    with col1:
        case_options = [f"{m['case_id']} - {m['gl_id']} ↔ {m['bank_id']}" for m in expected_matches]
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
        # Get GL and Bank transactions for this case
        gl = next(g for g in gl_transactions if g["id"] == selected_match["gl_id"])
        bank = next(b for b in bank_transactions if b["id"] == selected_match["bank_id"])

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
                st.metric("GL Amount", f"${gl['amount']:,.2f}")
                st.metric("Bank Amount", f"${bank['amount']:,.2f}")

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


if __name__ == "__main__":
    main()
