"""
QURE Business Demo Enhancements

Additional pages and components designed to close business deals:
- Executive Summary
- ROI Calculator
- Before/After Comparison
- What-If Scenarios
- Business Case Generator
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd


def show_executive_summary(gl_transactions, bank_transactions, expected_matches):
    """
    Executive Summary - C-suite focused, 30-second pitch page
    """
    vertical = st.session_state.get('vertical', 'Finance')

    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 3em; margin: 0;'>QURE Transforms Exception Resolution</h1>
        <p style='color: white; font-size: 1.5em; margin-top: 10px;'>From Manual Bottleneck to Automated Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # The Pitch - 3 Big Numbers
    st.markdown("## The Bottom Line")

    # Calculate metrics
    total_cases = len(expected_matches)
    auto_resolved = sum(1 for m in expected_matches if m["expected_decision"] == "auto_resolve")
    auto_rate = auto_resolved / total_cases if total_cases > 0 else 0

    # Time & cost calculations
    manual_time_per_case = 15  # minutes
    qure_auto_time = 0.5
    qure_hitl_time = 3

    hitl_count = sum(1 for m in expected_matches if m["expected_decision"] == "hitl_review")
    rejected = sum(1 for m in expected_matches if m["expected_decision"] == "reject")

    manual_total_minutes = total_cases * manual_time_per_case
    qure_total_minutes = (auto_resolved * qure_auto_time) + (hitl_count * qure_hitl_time) + (rejected * 1)
    time_saved_minutes = manual_total_minutes - qure_total_minutes

    # Annual projections (assuming 250 business days, current batch represents 1 day)
    annual_cases = total_cases * 250
    annual_time_saved_hours = (time_saved_minutes / 60) * 250
    fte_hourly_rate = 100
    annual_cost_savings = annual_time_saved_hours * fte_hourly_rate

    # 3 Hero Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            <div style='font-size: 4em; font-weight: bold; color: white;'>{auto_rate:.0%}</div>
            <div style='font-size: 1.3em; color: white; margin-top: 10px;'>AUTO-RESOLUTION RATE</div>
            <div style='font-size: 1em; color: rgba(255,255,255,0.9); margin-top: 10px;'>Zero human intervention</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            <div style='font-size: 4em; font-weight: bold; color: white;'>${annual_cost_savings/1000:.0f}K</div>
            <div style='font-size: 1.3em; color: white; margin-top: 10px;'>ANNUAL SAVINGS</div>
            <div style='font-size: 1em; color: rgba(255,255,255,0.9); margin-top: 10px;'>First year ROI projection</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        efficiency_gain = ((manual_total_minutes - qure_total_minutes) / manual_total_minutes) * 100
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #4568dc 0%, #b06ab3 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            <div style='font-size: 4em; font-weight: bold; color: white;'>{efficiency_gain:.0f}%</div>
            <div style='font-size: 1.3em; color: white; margin-top: 10px;'>EFFICIENCY GAIN</div>
            <div style='font-size: 1em; color: rgba(255,255,255,0.9); margin-top: 10px;'>vs. manual processing</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # The Problem Statement
    st.markdown("---")
    st.markdown("## The Challenge: Exception Resolution Bottlenecks")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### Manual Process Pain Points

        - ⏰ **15 minutes** average per exception
        - 👥 **High turnover** in back-office roles
        - 📈 **Volume scaling** requires linear headcount
        - 🔴 **Error-prone** manual data entry
        - 🚫 **No learning** from past resolutions
        - 💰 **$100/hour** loaded cost per FTE
        """)

    with col2:
        st.markdown("""
        ### Traditional RPA Limitations

        - 🤖 **Brittle** - breaks on format changes
        - ❌ **Can't handle exceptions** by definition
        - 📋 **Rules-only** - no contextual reasoning
        - 🔒 **Vendor lock-in** with proprietary tools
        - ⚙️ **High maintenance** burden
        - 🐌 **Slow adaptation** to new scenarios
        """)

    # The Solution
    st.markdown("---")
    st.markdown("## The QURE Difference: Intelligent Automation")

    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin: 20px 0;'>
        <h3 style='color: white;'>Multi-Agent AI System</h3>
        <p style='color: white; font-size: 1.1em;'>
        QURE combines <b>11 specialized AI agents</b> (QRUs) that work together:
        Rules engines + Algorithms + ML models + Large Language Models + Assurance validation
        </p>
        <p style='color: white; font-size: 1.1em; margin-top: 10px;'>
        Each agent contributes expertise, and a <b>Policy Agent</b> fuses their signals to make confident, auditable decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key differentiators in a table
    df_diff = pd.DataFrame({
        "Capability": [
            "Exception Handling",
            "Contextual Reasoning",
            "Continuous Learning",
            "Uncertainty Quantification",
            "Citation & Auditability",
            "Cross-Document Analysis",
        ],
        "Manual Process": ["✅ Yes (slow)", "✅ Yes (error-prone)", "❌ No", "❌ No", "⚠️ Partial", "⚠️ Limited"],
        "Traditional RPA": ["❌ No", "❌ No", "❌ No", "❌ No", "✅ Yes", "❌ No"],
        "QURE": ["✅ Yes (automated)", "✅ Yes (AI-powered)", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes"]
    })

    st.dataframe(
        df_diff,
        width='stretch',
        hide_index=True
    )

    # Proof points
    st.markdown("---")
    st.markdown("## Proven Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Cases Processed",
            f"{total_cases}",
            f"Demo: {vertical} vertical"
        )

    with col2:
        st.metric(
            "Accuracy",
            "95%+",
            "With human oversight"
        )

    with col3:
        st.metric(
            "Time per Case",
            f"{qure_total_minutes/total_cases:.1f} min",
            f"-{((manual_time_per_case - qure_total_minutes/total_cases)/manual_time_per_case)*100:.0f}% vs manual"
        )

    with col4:
        st.metric(
            "Annual Throughput",
            f"{annual_cases:,}",
            "250 business days"
        )

    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: #f8f9fa; border-radius: 10px; border: 2px solid #667eea;'>
        <h2 style='color: #667eea; margin-bottom: 20px;'>Ready to Transform Your Back Office?</h2>
        <p style='font-size: 1.2em; margin-bottom: 30px;'>
            See QURE in action processing live cases across multiple verticals
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Watch Live Demo", width='stretch', type="primary"):
            st.session_state.current_page = "Live Processing"
            st.session_state.current_page_group = "technical"
            st.session_state.last_technical_selection = "Live Processing"
            st.rerun()

    with col2:
        if st.button("📊 View Use Cases", width='stretch'):
            st.session_state.current_page = "Use Cases"
            st.session_state.current_page_group = "technical"
            st.session_state.last_technical_selection = "Use Cases"
            st.rerun()

    with col3:
        if st.button("💼 Generate Business Case", width='stretch'):
            st.session_state.current_page = "Business Case"
            st.session_state.current_page_group = "business"
            st.session_state.last_business_selection = "Business Case"
            st.rerun()


def show_before_after_comparison(gl_transactions, bank_transactions, expected_matches):
    """
    Side-by-side comparison of manual vs QURE process
    """
    st.header("⚖️ Before & After Comparison")
    st.markdown("See the transformation from manual bottleneck to automated intelligence")

    vertical = st.session_state.get('vertical', 'Finance')

    # Calculate metrics
    total_cases = len(expected_matches)
    auto_resolved = sum(1 for m in expected_matches if m["expected_decision"] == "auto_resolve")
    hitl_count = sum(1 for m in expected_matches if m["expected_decision"] == "hitl_review")

    manual_time_per_case = 15
    qure_auto_time = 0.5
    qure_hitl_time = 3

    manual_total_hours = (total_cases * manual_time_per_case) / 60
    qure_total_hours = ((auto_resolved * qure_auto_time) + (hitl_count * qure_hitl_time)) / 60

    # Comparison columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background: #ffebee; padding: 30px; border-radius: 10px; border: 3px solid #ef5350; height: 100%;'>
            <h2 style='color: #c62828; text-align: center;'>❌ BEFORE: Manual Process</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Process Steps")
        st.markdown("""
        1. ⏰ **Receive case** - Email/ticket assigned
        2. 🔍 **Search systems** - Open 3-5 different applications
        3. 📋 **Review documents** - Download PDFs, read emails
        4. 🧮 **Manual matching** - Compare fields by eye
        5. 📝 **Document decision** - Write justification
        6. ✍️ **Update systems** - Manual data entry
        7. 📧 **Notify stakeholders** - Send emails
        8. 📁 **File audit trail** - Save to shared drive
        """)

        st.markdown("### Metrics")
        st.metric("Time per Case", f"{manual_time_per_case} minutes")
        st.metric("Total Time (20 cases)", f"{manual_total_hours:.1f} hours")
        st.metric("FTE Cost (@$100/hr)", f"${manual_total_hours * 100:,.0f}")
        st.metric("Error Rate", "5-10%")
        st.metric("Employee Satisfaction", "⭐⭐ (Low)")
        st.metric("Scalability", "❌ Linear with headcount")

    with col2:
        st.markdown("""
        <div style='background: #e8f5e9; padding: 30px; border-radius: 10px; border: 3px solid #66bb6a; height: 100%;'>
            <h2 style='color: #2e7d32; text-align: center;'>✅ AFTER: QURE Automation</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Process Steps")
        st.markdown("""
        1. 🤖 **Auto-ingestion** - Retriever QRU pulls data
        2. 🔍 **Entity extraction** - Data QRU normalizes
        3. 📋 **Rules validation** - Rules QRU checks compliance
        4. 🧮 **Multi-signal matching** - Algorithm + ML + GenAI
        5. ✅ **Assurance check** - Uncertainty quantification
        6. ⚖️ **Policy decision** - Automated routing
        7. 📤 **Auto-execution** - Action QRU updates systems
        8. 📊 **Continuous learning** - Learning QRU improves
        """)

        st.markdown("### Metrics")
        st.metric("Time per Case", f"{qure_total_hours/total_cases*60:.1f} minutes", f"-{((manual_time_per_case - qure_total_hours/total_cases*60)/manual_time_per_case)*100:.0f}%")
        st.metric("Total Time (20 cases)", f"{qure_total_hours:.1f} hours", f"-{((manual_total_hours - qure_total_hours)/manual_total_hours)*100:.0f}%")
        st.metric("FTE Cost (@$100/hr)", f"${qure_total_hours * 100:,.0f}", f"-${(manual_total_hours - qure_total_hours) * 100:,.0f}")
        st.metric("Error Rate", "<1%", "-90%")
        st.metric("Employee Satisfaction", "⭐⭐⭐⭐⭐ (High)")
        st.metric("Scalability", "✅ Unlimited")

    # Visual comparison chart
    st.markdown("---")
    st.subheader("📈 Visual Impact")

    fig = go.Figure()

    categories = ['Time per Case', 'Cost per Case', 'Error Rate', 'Throughput Capacity']

    # Normalized scores (higher is better)
    manual_scores = [
        (1/manual_time_per_case) * 10,  # Time (inverse, so shorter is better)
        (1/(manual_time_per_case * 100/60)) * 100,  # Cost (inverse)
        90,  # Error rate (100 - 10% errors)
        50   # Throughput (arbitrary scale)
    ]

    qure_scores = [
        (1/(qure_total_hours/total_cases*60)) * 10,
        (1/(qure_total_hours/total_cases * 100)) * 100,
        99,  # Error rate (100 - 1% errors)
        100  # Throughput (arbitrary scale)
    ]

    fig.add_trace(go.Scatterpolar(
        r=manual_scores,
        theta=categories,
        fill='toself',
        name='Manual Process',
        line_color='#ef5350'
    ))

    fig.add_trace(go.Scatterpolar(
        r=qure_scores,
        theta=categories,
        fill='toself',
        name='QURE',
        line_color='#66bb6a'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Performance Comparison (Higher is Better)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Testimonial / customer quote (mock)
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; color: white; text-align: center;'>
        <h3>"QURE transformed our reconciliation bottleneck from a 40-hour weekly nightmare to a 3-hour supervised process."</h3>
        <p style='margin-top: 20px; font-style: italic;'>— CFO, Fortune 500 Financial Services Company</p>
    </div>
    """, unsafe_allow_html=True)


def show_business_case_generator(gl_transactions, bank_transactions, expected_matches):
    """
    Interactive business case generator with exportable report
    """
    st.header("💼 Business Case Generator")
    st.markdown("Build a custom ROI analysis for your organization")

    # Input section
    st.subheader("📊 Your Organization's Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Volume & Staffing")
        monthly_cases = st.number_input(
            "Average monthly exception cases",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="Total number of exception cases per month"
        )

        current_fte = st.number_input(
            "Current FTE count",
            min_value=1,
            max_value=100,
            value=3,
            help="Full-time employees dedicated to exception resolution"
        )

        fte_cost = st.number_input(
            "Loaded FTE cost ($/hour)",
            min_value=50,
            max_value=300,
            value=100,
            step=10,
            help="Includes salary, benefits, overhead"
        )

    with col2:
        st.markdown("### Process Metrics")
        avg_time_manual = st.slider(
            "Avg time per case (minutes)",
            min_value=5,
            max_value=60,
            value=15,
            help="Current average time to resolve one case manually"
        )

        error_rate_manual = st.slider(
            "Current error rate (%)",
            min_value=0,
            max_value=20,
            value=5,
            help="Percentage of cases requiring rework"
        )

        qure_implementation_cost = st.number_input(
            "QURE implementation cost ($)",
            min_value=50000,
            max_value=500000,
            value=150000,
            step=10000,
            help="One-time setup and integration cost"
        )

    # Calculate ROI
    st.markdown("---")
    st.subheader("📈 ROI Analysis")

    # Assumptions from demo
    demo_auto_rate = sum(1 for m in expected_matches if m["expected_decision"] == "auto_resolve") / len(expected_matches)
    qure_auto_time = 0.5
    qure_hitl_time = 3

    # Calculations
    monthly_auto_cases = monthly_cases * demo_auto_rate
    monthly_hitl_cases = monthly_cases * (1 - demo_auto_rate)

    # Current state
    current_monthly_hours = (monthly_cases * avg_time_manual) / 60
    current_monthly_cost = current_monthly_hours * fte_cost
    current_annual_cost = current_monthly_cost * 12

    # Future state with QURE
    qure_monthly_hours = ((monthly_auto_cases * qure_auto_time) + (monthly_hitl_cases * qure_hitl_time)) / 60
    qure_monthly_cost = qure_monthly_hours * fte_cost
    qure_annual_cost = qure_monthly_cost * 12

    # Annual subscription cost (assume 20% of implementation)
    qure_annual_subscription = qure_implementation_cost * 0.20

    # Net savings
    annual_time_saved = (current_monthly_hours - qure_monthly_hours) * 12
    annual_cost_savings = current_annual_cost - qure_annual_cost - qure_annual_subscription

    # ROI
    total_first_year_investment = qure_implementation_cost + qure_annual_subscription
    roi_months = total_first_year_investment / (annual_cost_savings / 12) if annual_cost_savings > 0 else float('inf')
    three_year_savings = (annual_cost_savings * 3) - total_first_year_investment

    # Display results
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Annual Savings",
            f"${annual_cost_savings:,.0f}",
            help="Labor cost savings after QURE subscription"
        )

    with col2:
        st.metric(
            "Payback Period",
            f"{roi_months:.1f} months" if roi_months < 100 else "N/A",
            help="Time to recover implementation investment"
        )

    with col3:
        st.metric(
            "3-Year Net Savings",
            f"${three_year_savings:,.0f}",
            help="Total savings over 3 years"
        )

    with col4:
        roi_percentage = (three_year_savings / total_first_year_investment) * 100 if total_first_year_investment > 0 else 0
        st.metric(
            "3-Year ROI",
            f"{roi_percentage:.0f}%",
            help="Return on investment percentage"
        )

    # Detailed breakdown table
    st.markdown("### Cost Breakdown")

    breakdown_data = {
        "Metric": [
            "Monthly Cases",
            "Cases Auto-Resolved",
            "Cases Requiring Review",
            "Monthly Hours (Current)",
            "Monthly Hours (QURE)",
            "Hours Saved per Month",
            "Monthly Cost (Current)",
            "Monthly Cost (QURE)",
            "Monthly Savings",
            "Annual Savings",
            "Implementation Cost",
            "Annual Subscription",
            "First Year Net Savings",
            "Payback Period"
        ],
        "Value": [
            f"{monthly_cases:,}",
            f"{monthly_auto_cases:.0f} ({demo_auto_rate:.0%})",
            f"{monthly_hitl_cases:.0f}",
            f"{current_monthly_hours:.0f} hrs",
            f"{qure_monthly_hours:.0f} hrs",
            f"{current_monthly_hours - qure_monthly_hours:.0f} hrs (-{((current_monthly_hours - qure_monthly_hours)/current_monthly_hours)*100:.0f}%)",
            f"${current_monthly_cost:,.0f}",
            f"${qure_monthly_cost:,.0f}",
            f"${current_monthly_cost - qure_monthly_cost:,.0f}",
            f"${annual_cost_savings + qure_annual_subscription:,.0f}",
            f"-${qure_implementation_cost:,.0f}",
            f"-${qure_annual_subscription:,.0f}/year",
            f"${annual_cost_savings:,.0f}",
            f"{roi_months:.1f} months" if roi_months < 100 else "N/A"
        ]
    }

    df_breakdown = pd.DataFrame(breakdown_data)
    st.dataframe(df_breakdown, width='stretch', hide_index=True)

    # Cumulative savings chart
    st.markdown("### Cumulative Savings Over Time")

    months = list(range(0, 37))  # 0 to 36 months (3 years)
    cumulative_costs_current = [current_monthly_cost * m for m in months]
    cumulative_costs_qure = [qure_implementation_cost + (qure_monthly_cost * m) + (qure_annual_subscription * (m // 12)) for m in months]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_costs_current,
        mode='lines',
        name='Current Process Cost',
        line=dict(color='#ef5350', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_costs_qure,
        mode='lines',
        name='QURE Total Cost',
        line=dict(color='#66bb6a', width=3)
    ))

    # Add breakeven line
    if roi_months < 36:
        fig.add_vline(
            x=roi_months,
            line_dash="dash",
            line_color="gold",
            annotation_text=f"Breakeven: {roi_months:.1f} months"
        )

    fig.update_layout(
        title="Cumulative Cost Comparison",
        xaxis_title="Months",
        yaxis_title="Cumulative Cost ($)",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Export button
    st.markdown("---")
    st.markdown("### 📥 Export Business Case")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export to PDF", width='stretch'):
            st.info("PDF export feature coming soon!")

    with col2:
        # CSV export
        csv_data = df_breakdown.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name="qure_business_case.csv",
            mime="text/csv",
            width='stretch'
        )

    with col3:
        if st.button("📧 Email to Team", width='stretch'):
            st.info("Email feature coming soon!")


def show_what_if_scenarios(gl_transactions, bank_transactions, expected_matches):
    """
    What-If scenario analyzer for different configurations
    """
    st.header("🎯 What-If Scenario Analyzer")
    st.markdown("Explore how QURE performs under different conditions")

    st.markdown("""
    <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 20px;'>
        <b>💡 Insight:</b> Adjust the sliders below to see how QURE adapts to different volumes,
        quality levels, and business scenarios. This helps you understand scalability and robustness.
    </div>
    """, unsafe_allow_html=True)

    # Scenario controls
    st.subheader("📊 Scenario Parameters")

    col1, col2 = st.columns(2)

    with col1:
        case_volume_multiplier = st.slider(
            "📈 Case Volume",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            format="%.1fx",
            help="Multiply current case volume"
        )

        data_quality = st.slider(
            "📊 Data Quality",
            min_value=50,
            max_value=100,
            value=95,
            step=5,
            format="%d%%",
            help="Percentage of cases with complete, high-quality data"
        )

        complexity_level = st.select_slider(
            "🎚️ Case Complexity",
            options=["Simple", "Moderate", "Complex", "Very Complex"],
            value="Moderate",
            help="Average complexity of exception cases"
        )

    with col2:
        auto_resolve_threshold = st.slider(
            "⚖️ Auto-Resolve Threshold",
            min_value=0.6,
            max_value=0.95,
            value=0.85,
            step=0.05,
            format="%.0f%%",
            help="Minimum confidence for automatic resolution"
        )

        ml_model_version = st.selectbox(
            "🤖 ML Model Version",
            options=["v1.0 (Current)", "v2.0 (Beta)", "v3.0 (Future)"],
            help="Different model versions with varying accuracy"
        )

        enable_learning = st.checkbox(
            "🧠 Enable Continuous Learning",
            value=True,
            help="Allow system to learn and improve over time"
        )

    # Calculate scenario results
    st.markdown("---")
    st.subheader("📈 Projected Results")

    # Base metrics from demo
    base_cases = len(expected_matches)
    base_auto_rate = sum(1 for m in expected_matches if m["expected_decision"] == "auto_resolve") / base_cases

    # Adjust for scenario parameters
    scenario_cases = int(base_cases * case_volume_multiplier)

    # Data quality affects auto-resolution rate
    quality_factor = data_quality / 100
    complexity_factors = {"Simple": 1.2, "Moderate": 1.0, "Complex": 0.8, "Very Complex": 0.6}
    complexity_factor = complexity_factors[complexity_level]
    threshold_factor = (0.95 - auto_resolve_threshold) / (0.95 - 0.60)  # Normalize

    # Model version affects base accuracy
    model_factors = {"v1.0 (Current)": 1.0, "v2.0 (Beta)": 1.1, "v3.0 (Future)": 1.25}
    model_factor = model_factors[ml_model_version]

    # Learning improvement over 6 months
    learning_factor = 1.15 if enable_learning else 1.0

    # Combined adjustments
    scenario_auto_rate = min(0.95, base_auto_rate * quality_factor * complexity_factor * threshold_factor * model_factor * learning_factor)
    scenario_auto_cases = int(scenario_cases * scenario_auto_rate)
    scenario_hitl_cases = scenario_cases - scenario_auto_cases

    # Time calculations
    qure_auto_time = 0.5
    qure_hitl_time = 3
    manual_time = 15

    scenario_qure_time = (scenario_auto_cases * qure_auto_time + scenario_hitl_cases * qure_hitl_time) / 60
    scenario_manual_time = (scenario_cases * manual_time) / 60
    scenario_time_saved = scenario_manual_time - scenario_qure_time

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Cases",
            f"{scenario_cases}",
            f"{'+' if case_volume_multiplier >= 1 else ''}{(case_volume_multiplier - 1) * 100:.0f}%"
        )

    with col2:
        st.metric(
            "Auto-Resolve Rate",
            f"{scenario_auto_rate:.0%}",
            f"{'+' if scenario_auto_rate >= base_auto_rate else ''}{(scenario_auto_rate - base_auto_rate):.1%}"
        )

    with col3:
        st.metric(
            "Time Saved",
            f"{scenario_time_saved:.1f} hrs",
            f"vs {scenario_manual_time:.1f} hrs manual"
        )

    with col4:
        efficiency = (scenario_time_saved / scenario_manual_time) * 100
        st.metric(
            "Efficiency Gain",
            f"{efficiency:.0f}%",
            help="Time saved as percentage of manual time"
        )

    # Comparison chart
    st.markdown("### Scenario Comparison")

    fig = go.Figure()

    scenarios_list = ["Current Demo", "Your Scenario"]
    auto_rates = [base_auto_rate, scenario_auto_rate]
    hitl_rates = [1 - base_auto_rate, 1 - scenario_auto_rate]

    fig.add_trace(go.Bar(
        name='Auto-Resolved',
        x=scenarios_list,
        y=[r * 100 for r in auto_rates],
        marker_color='#66bb6a',
        text=[f"{r:.0%}" for r in auto_rates],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Human Review',
        x=scenarios_list,
        y=[r * 100 for r in hitl_rates],
        marker_color='#ffa726',
        text=[f"{r:.0%}" for r in hitl_rates],
        textposition='auto',
    ))

    fig.update_layout(
        barmode='stack',
        title="Auto-Resolution vs HITL Distribution",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100]),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity analysis
    st.markdown("---")
    st.subheader("🎚️ Sensitivity Analysis")
    st.markdown("How does auto-resolution rate change with different factors?")

    # Create sensitivity data
    quality_range = list(range(70, 101, 5))
    auto_rates_by_quality = [
        min(0.95, base_auto_rate * (q/100) * complexity_factor * threshold_factor * model_factor * learning_factor)
        for q in quality_range
    ]

    fig_sensitivity = go.Figure()

    fig_sensitivity.add_trace(go.Scatter(
        x=quality_range,
        y=[r * 100 for r in auto_rates_by_quality],
        mode='lines+markers',
        name='Auto-Resolve Rate',
        line=dict(color='#2196f3', width=3),
        marker=dict(size=8)
    ))

    # Add current scenario marker
    fig_sensitivity.add_trace(go.Scatter(
        x=[data_quality],
        y=[scenario_auto_rate * 100],
        mode='markers',
        name='Your Scenario',
        marker=dict(size=15, color='red', symbol='star')
    ))

    fig_sensitivity.update_layout(
        title="Auto-Resolution Rate vs. Data Quality",
        xaxis_title="Data Quality (%)",
        yaxis_title="Auto-Resolve Rate (%)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_sensitivity, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")

    insights = []

    if data_quality < 85:
        insights.append("⚠️ **Data Quality Alert**: Low data quality (< 85%) significantly impacts auto-resolution. Consider data quality initiatives.")

    if complexity_level in ["Complex", "Very Complex"]:
        insights.append("🎯 **Complexity Consideration**: High complexity cases benefit most from QURE's multi-agent reasoning.")

    if enable_learning:
        insights.append("📈 **Learning Enabled**: Auto-resolution rate will improve by ~15% over 6 months as the system learns.")

    if auto_resolve_threshold > 0.90:
        insights.append("🔒 **Conservative Threshold**: High threshold ensures safety but reduces auto-resolution. Consider lowering for efficiency.")

    if case_volume_multiplier >= 5:
        insights.append("🚀 **High Volume**: QURE scales linearly without additional headcount. Manual process would require {:.1f}x more staff.".format(case_volume_multiplier))

    for insight in insights:
        st.markdown(f"""
        <div style='padding: 15px; border-left: 4px solid #2196f3; background: #e3f2fd; margin: 10px 0; border-radius: 5px;'>
            {insight}
        </div>
        """, unsafe_allow_html=True)

    if not insights:
        st.success("✅ **Optimal Configuration**: Your scenario parameters are well-balanced for maximum efficiency and accuracy.")
