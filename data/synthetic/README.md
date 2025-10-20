# QURE Synthetic Test Data

## Overview

This directory contains synthetically generated test cases for demonstrating QURE's multi-vertical exception resolution capabilities.

**Generated**: October 20, 2025
**Total Cases**: 300 (100 per vertical)

## Dataset Files

### 1. Finance Reconciliation (`finance_reconciliation_cases.json`)

**Use Case**: GL-to-Bank statement reconciliation

**Cases**: 100 (60 easy, 30 medium, 10 hard)

**Data Structure**:
```json
{
  "case_id": "GL-20251020-0001",
  "gl_entry": {
    "id": "GL-20251020-0001",
    "date": "2025-08-15",
    "account": "1000-Cash",
    "amount": 12500.50,
    "description": "Payment to Acme Corp",
    "reference": "INV-12345"
  },
  "bank_entry": {
    "id": "BNK-20251015-1234",
    "date": "2025-08-15",
    "bank": "Chase",
    "amount": 12500.50,
    "description": "Payment to Acme Corp",
    "check_number": "1234"
  },
  "ground_truth": {
    "is_match": true,
    "confidence": 0.95,
    "should_auto_resolve": true,
    "requires_review": false
  },
  "context": {
    "transaction_amount": 12500.50,
    "data_quality_score": 0.85,
    "urgency": "medium",
    "has_swift_reference": true,
    "sox_controlled": false,
    "vertical": "finance",
    "complexity": "easy"
  }
}
```

**Complexity Variations**:
- **Easy (60%)**: Clean matches, complete data, low amounts (<$50k)
- **Medium (30%)**: Moderate complexity, amounts $50k-$500k
- **Hard (10%)**: High-value ($500k+), data quality issues, SOX controls

### 2. Insurance Subrogation (`insurance_subrogation_cases.json`)

**Use Case**: Determine whether to pursue subrogation claim recovery

**Cases**: 100 (50 easy, 30 medium, 20 hard)

**Data Structure**:
```json
{
  "case_id": "SUB-20251020-0001",
  "claim": {
    "claim_id": "SUB-20251020-0001",
    "incident_date": "2025-03-15",
    "incident_type": "Auto Accident",
    "recovery_amount": 15000.00,
    "liable_party": "State Farm",
    "policy_number": "POL-123456",
    "has_police_report": true,
    "has_liability_admission": true,
    "statute_expiry_days": 365
  },
  "ground_truth": {
    "should_pursue": true,
    "expected_recovery": 13500.00,
    "estimated_success_rate": 0.9,
    "requires_review": false
  },
  "context": {
    "transaction_amount": 15000.00,
    "data_quality_score": 0.85,
    "urgency": "low",
    "has_swift_reference": true,
    "days_to_statute_expiry": 365,
    "vertical": "insurance",
    "complexity": "easy"
  }
}
```

**Complexity Variations**:
- **Easy (50%)**: Clear liability, good documentation, recovery $5k-$25k
- **Medium (30%)**: Moderate evidence, recovery $25k-$100k
- **Hard (20%)**: Disputed liability, poor docs, high-value $100k+

### 3. Healthcare Prior Authorization (`healthcare_prior_auth_cases.json`)

**Use Case**: Approve/deny prior authorization requests

**Cases**: 100 (40 easy, 40 medium, 20 hard)

**Data Structure**:
```json
{
  "case_id": "AUTH-20251020-0001",
  "auth_request": {
    "auth_id": "AUTH-20251020-0001",
    "request_date": "2025-10-15",
    "procedure": "MRI Scan",
    "diagnosis": "Chronic Back Pain",
    "estimated_cost": 2500.00,
    "provider_id": "NPI-1234567",
    "member_id": "MEM-123456",
    "clinical_urgency": "routine",
    "has_prior_treatment": true,
    "has_clinical_notes": true,
    "medical_necessity_score": 0.85
  },
  "ground_truth": {
    "should_approve": true,
    "confidence": 0.85,
    "requires_peer_review": false,
    "requires_additional_info": false
  },
  "context": {
    "transaction_amount": 2500.00,
    "data_quality_score": 0.90,
    "urgency": "routine",
    "has_swift_reference": true,
    "medical_necessity": 0.85,
    "clinical_urgency": "routine",
    "vertical": "healthcare",
    "complexity": "easy"
  }
}
```

**Complexity Variations**:
- **Easy (40%)**: Clear medical necessity, complete docs, routine procedures
- **Medium (40%)**: Moderate necessity, some missing data
- **Hard (20%)**: Questionable necessity, incomplete docs, urgent/critical

## Context Features

All cases include standardized context features for ML/clustering:

| Feature | Type | Description |
|---------|------|-------------|
| `transaction_amount` | float | Dollar value of transaction/claim/procedure |
| `data_quality_score` | float (0-1) | Completeness and quality of documentation |
| `urgency` | enum | low, medium, high, critical |
| `has_swift_reference` | bool | Whether case has strong reference/documentation |
| `vertical` | string | finance, insurance, healthcare |
| `complexity` | string | easy, medium, hard |

**Vertical-Specific Features**:
- Finance: `sox_controlled` (bool)
- Insurance: `days_to_statute_expiry` (int)
- Healthcare: `medical_necessity` (float), `clinical_urgency` (string)

## Ground Truth Labels

Each case includes ground truth for validation:

| Label | Description |
|-------|-------------|
| `should_auto_resolve` / `should_pursue` / `should_approve` | Primary decision |
| `confidence` | Expected confidence level (0-1) |
| `requires_review` | Whether human review is needed |
| Additional outcome-specific fields | Success rates, recovery amounts, etc. |

## Usage

### Load Data

```python
import json

# Load finance cases
with open('data/synthetic/finance_reconciliation_cases.json', 'r') as f:
    finance_cases = json.load(f)

# Filter by complexity
easy_cases = [c for c in finance_cases if c['context']['complexity'] == 'easy']
hard_cases = [c for c in finance_cases if c['context']['complexity'] == 'hard']

# Get high-value cases
high_value = [c for c in finance_cases if c['context']['transaction_amount'] > 500000]
```

### Run Demo

```python
from agents.orchestrator import Orchestrator
from agents.policy import PolicyAgent

orchestrator = Orchestrator()
policy_agent = PolicyAgent()

# Process a case
case = finance_cases[0]
result = orchestrator.process_case(case)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Ground Truth: {case['ground_truth']['should_auto_resolve']}")
```

### Evaluate Performance

```python
import numpy as np

# Compute accuracy
correct = 0
total = 0

for case in finance_cases:
    result = orchestrator.process_case(case)
    predicted = result['decision'] == 'auto_resolve'
    actual = case['ground_truth']['should_auto_resolve']

    if predicted == actual:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.1%}")
```

### Train Learning Agent

```python
from agents.learning import LearningAgent

agent = LearningAgent(vertical='finance')

# Simulate decisions and outcomes
for case in finance_cases:
    # Log decision
    log_id = agent.log_decision(
        case_id=case['case_id'],
        context=case['context'],
        policy_weights={'alpha': 0.25, 'beta': 0.20, ...},
        decision_type='auto_resolve',
        utility_score=0.85
    )

    # Update with ground truth outcome
    agent.update_outcome(
        log_id=log_id,
        correct=case['ground_truth']['should_auto_resolve'],
        reversed=False,
        cycle_time_hours=2,
        cost_saved=50,
        context=case['context'],
        action={'decision': 'auto_resolve', 'policy_weights': {...}}
    )

# Run learning update
report = agent.run_update(lookback_days=1, min_samples=10)
print(f"Learning update status: {report['status']}")
```

## Statistics

### Finance
- **Total**: 100 cases
- **Easy**: 60 (60%)
- **Medium**: 30 (30%)
- **Hard**: 10 (10%)
- **Amount Range**: $1k - $5M
- **Avg Quality**: 0.75
- **Match Rate**: ~85% (easy), ~60% (medium), ~40% (hard)

### Insurance
- **Total**: 100 cases
- **Easy**: 50 (50%)
- **Medium**: 30 (30%)
- **Hard**: 20 (20%)
- **Recovery Range**: $5k - $500k
- **Avg Quality**: 0.70
- **Pursuit Rate**: ~90% (easy), ~60% (medium), ~30% (hard)

### Healthcare
- **Total**: 100 cases
- **Easy**: 40 (40%)
- **Medium**: 40 (40%)
- **Hard**: 20 (20%)
- **Cost Range**: $500 - $200k
- **Avg Quality**: 0.78
- **Approval Rate**: ~85% (easy), ~60% (medium), ~40% (hard)

## Regeneration

To regenerate data with different parameters:

```bash
cd data/synthetic
python generate_test_data.py
```

Or programmatically:

```python
from generate_test_data import generate_all_verticals

datasets = generate_all_verticals(n_cases_per_vertical=200)
```

## Quality Characteristics

**Realism**:
- Amounts follow realistic distributions per vertical
- Dates span 1-365 days historical
- Entity names drawn from realistic pools
- Complexity variations match real-world distribution

**Consistency**:
- Fixed random seed (42) for reproducibility
- Consistent feature naming across verticals
- Ground truth labels derived algorithmically

**Coverage**:
- Wide range of amounts (low to high-value)
- All urgency levels represented
- Quality scores span 0.3-0.95
- Edge cases included (missing data, truncated descriptions, date mismatches)

## Next Steps

1. **Validation**: Run end-to-end demos with synthetic data
2. **Metrics**: Compute baseline accuracy/precision/recall
3. **Learning**: Use for Thompson Sampling training
4. **UI**: Build case viewer dashboard
5. **Expansion**: Add more edge cases and rare scenarios

## Notes

- Data is synthetic and does not contain any real PII or PHI
- Generated for testing and demonstration purposes only
- Ground truth labels are algorithmically derived, not human-validated
- Use seed=42 for reproducibility across runs

## Generator Code

See `generate_test_data.py` for complete generation logic and customization options.
