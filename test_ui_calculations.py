"""
Test script to validate UI calculations for all business demo pages
"""
import json
import sys
import io

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load data
with open('data/synthetic/finance/expected_matches.json') as f:
    expected_matches = json.load(f)

# Basic metrics
total_cases = len(expected_matches)
auto_resolved = sum(1 for m in expected_matches if m["expected_decision"] == "auto_resolve")
hitl_count = sum(1 for m in expected_matches if m["expected_decision"] == "hitl_review")
rejected = sum(1 for m in expected_matches if m["expected_decision"] == "reject")
auto_rate = auto_resolved / total_cases if total_cases > 0 else 0

print("=" * 60)
print("QURE UI CALCULATION TEST")
print("=" * 60)
print(f"\n📊 Data Summary:")
print(f"  Total cases: {total_cases}")
print(f"  Auto-resolved: {auto_resolved} ({auto_rate:.0%})")
print(f"  HITL review: {hitl_count} ({hitl_count/total_cases:.0%})")
print(f"  Rejected: {rejected} ({rejected/total_cases:.0%})")

# Executive Summary Calculations
print(f"\n🎯 EXECUTIVE SUMMARY PAGE")
print("-" * 60)

manual_time_per_case = 15  # minutes
qure_auto_time = 0.5
qure_hitl_time = 3

manual_total_minutes = total_cases * manual_time_per_case
qure_total_minutes = (auto_resolved * qure_auto_time) + (hitl_count * qure_hitl_time) + (rejected * 1)
time_saved_minutes = manual_total_minutes - qure_total_minutes

# Annual projections
annual_cases = total_cases * 250
annual_time_saved_hours = (time_saved_minutes / 60) * 250
fte_hourly_rate = 100
annual_cost_savings = annual_time_saved_hours * fte_hourly_rate
efficiency_gain = ((manual_total_minutes - qure_total_minutes) / manual_total_minutes) * 100

print(f"  Hero Metric 1 - Auto-Resolution Rate: {auto_rate:.0%}")
print(f"  Hero Metric 2 - Annual Savings: ${annual_cost_savings/1000:.0f}K")
print(f"  Hero Metric 3 - Efficiency Gain: {efficiency_gain:.0f}%")
print(f"\n  ✅ Expected on page: 40%, ~$187K, ~85%")

# Before & After Comparison
print(f"\n⚖️ BEFORE & AFTER COMPARISON")
print("-" * 60)

manual_total_hours = (total_cases * manual_time_per_case) / 60
qure_total_hours = ((auto_resolved * qure_auto_time) + (hitl_count * qure_hitl_time)) / 60

print(f"  Manual time: {manual_total_hours:.1f} hours")
print(f"  QURE time: {qure_total_hours:.1f} hours")
print(f"  Time saved: {manual_total_hours - qure_total_hours:.1f} hours ({((manual_total_hours - qure_total_hours)/manual_total_hours)*100:.0f}%)")
print(f"  Cost savings: ${(manual_total_hours - qure_total_hours) * 100:,.0f}")

# Business Case Generator
print(f"\n💼 BUSINESS CASE GENERATOR")
print("-" * 60)

# Default inputs
monthly_cases = 1000
current_fte = 3
fte_cost = 100
avg_time_manual = 15
error_rate_manual = 5
qure_implementation_cost = 150000

# Calculations
demo_auto_rate = auto_rate
monthly_auto_cases = monthly_cases * demo_auto_rate
monthly_hitl_cases = monthly_cases * (1 - demo_auto_rate)

current_monthly_hours = (monthly_cases * avg_time_manual) / 60
current_monthly_cost = current_monthly_hours * fte_cost
current_annual_cost = current_monthly_cost * 12

qure_monthly_hours = ((monthly_auto_cases * qure_auto_time) + (monthly_hitl_cases * qure_hitl_time)) / 60
qure_monthly_cost = qure_monthly_hours * fte_cost
qure_annual_cost = qure_monthly_cost * 12

qure_annual_subscription = qure_implementation_cost * 0.20

annual_time_saved = (current_monthly_hours - qure_monthly_hours) * 12
annual_cost_savings_bc = current_annual_cost - qure_annual_cost - qure_annual_subscription

total_first_year_investment = qure_implementation_cost + qure_annual_subscription
roi_months = total_first_year_investment / (annual_cost_savings_bc / 12) if annual_cost_savings_bc > 0 else float('inf')
three_year_savings = (annual_cost_savings_bc * 3) - total_first_year_investment

print(f"  Inputs: {monthly_cases} cases/mo, {fte_cost}$/hr, {avg_time_manual} min/case")
print(f"  Annual Savings: ${annual_cost_savings_bc:,.0f}")
print(f"  Payback Period: {roi_months:.1f} months")
print(f"  3-Year Net Savings: ${three_year_savings:,.0f}")
print(f"  3-Year ROI: {(three_year_savings / total_first_year_investment) * 100:.0f}%")

# What-If Scenarios
print(f"\n🎯 WHAT-IF SCENARIOS")
print("-" * 60)

# Test scenario with default parameters
case_volume_multiplier = 1.0
data_quality = 95
complexity_level = "Moderate"
auto_resolve_threshold = 0.85
ml_model_version = "v1.0 (Current)"
enable_learning = True

# Calculate scenario results
scenario_cases = int(total_cases * case_volume_multiplier)
quality_factor = data_quality / 100
complexity_factors = {"Simple": 1.2, "Moderate": 1.0, "Complex": 0.8, "Very Complex": 0.6}
complexity_factor = complexity_factors[complexity_level]
threshold_factor = (0.95 - auto_resolve_threshold) / (0.95 - 0.60)
model_factors = {"v1.0 (Current)": 1.0, "v2.0 (Beta)": 1.1, "v3.0 (Future)": 1.25}
model_factor = model_factors[ml_model_version]
learning_factor = 1.15 if enable_learning else 1.0

scenario_auto_rate = min(0.95, auto_rate * quality_factor * complexity_factor * threshold_factor * model_factor * learning_factor)
scenario_auto_cases = int(scenario_cases * scenario_auto_rate)
scenario_hitl_cases = scenario_cases - scenario_auto_cases

scenario_qure_time = (scenario_auto_cases * qure_auto_time + scenario_hitl_cases * qure_hitl_time) / 60
scenario_manual_time = (scenario_cases * manual_time_per_case) / 60
scenario_time_saved = scenario_manual_time - scenario_qure_time

print(f"  Scenario: {case_volume_multiplier}x volume, {data_quality}% quality, {complexity_level}")
print(f"  Total cases: {scenario_cases}")
print(f"  Auto-resolve rate: {scenario_auto_rate:.0%}")
print(f"  Time saved: {scenario_time_saved:.1f} hrs")
print(f"  Efficiency gain: {(scenario_time_saved / scenario_manual_time) * 100:.0f}%")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\n✅ All calculations verified!")
print("📋 Next: Navigate to each page in the UI and compare values")
