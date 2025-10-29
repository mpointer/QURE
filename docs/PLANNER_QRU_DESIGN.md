# Planner QRU Design Document

## Overview

The **Planner QRU** (Planning Agent) is a meta-orchestrator that sits at the beginning of the QURE pipeline. It analyzes each incoming case and dynamically determines which specialized QRUs should be invoked to solve that specific problem.

## Problem Statement

Currently, QURE invokes all 11 QRUs for every case, which is:
- **Inefficient**: Not all QRUs are needed for every case
- **Expensive**: Running LLM-based QRUs costs money and time
- **Sub-optimal**: Some cases only need rules-based or algorithmic QRUs

## Architecture Position

```
Case Intake
    ↓
[Planner QRU] ← Analyzes case and creates execution plan
    ↓
Dynamic QRU Selection
    ↓
[Retriever QRU] → [Selected QRUs based on plan] → [Assurance QRU]
    ↓
[Policy QRU] → Decision
```

## Core Responsibilities

### 1. Case Classification
Analyze the incoming case and classify it by:
- **Complexity**: Simple, Moderate, Complex, Highly Complex
- **Data Quality**: Complete, Incomplete, Ambiguous, Conflicting
- **Required Reasoning**: Rules-only, Algorithmic, ML, Contextual (LLM)
- **Domain Specificity**: Finance, Healthcare, Insurance, Retail, Manufacturing

### 2. QRU Selection Strategy

The Planner QRU uses a decision tree to determine which QRUs to invoke:

```
IF data_quality == "Complete" AND complexity == "Simple":
    INVOKE: [Retriever, Data, Rules, Algorithm, Policy, Action]
    SKIP: [ML Model, GenAI, Assurance, Orchestration, Learning]
    REASONING: "Perfect match scenario - rules and algorithms sufficient"

ELIF complexity == "Moderate" AND matching_pattern == "fuzzy":
    INVOKE: [Retriever, Data, Rules, Algorithm, ML Model, Assurance, Policy, Action]
    SKIP: [GenAI]
    REASONING: "Fuzzy matching needs ML but not LLM context"

ELIF data_quality == "Ambiguous" OR complexity == "Complex":
    INVOKE: [Retriever, Data, Rules, Algorithm, ML Model, GenAI, Assurance, Policy, Action]
    SKIP: [None] (Full pipeline)
    REASONING: "Complex case requires full multi-agent reasoning"

ELIF first_time_scenario == True:
    INVOKE: [All QRUs including Learning]
    REASONING: "New scenario - need full pipeline and learning"
```

### 3. Execution Plan Generation

The Planner QRU outputs an **Execution Plan**:

```json
{
  "case_id": "RECON_2024_001",
  "plan_id": "PLAN_20241028_001",
  "classification": {
    "complexity": "Simple",
    "data_quality": "Complete",
    "reasoning_type": "Rules + Algorithm",
    "confidence": 0.95
  },
  "selected_qrus": [
    {
      "qru_name": "Retriever",
      "priority": 1,
      "required": true,
      "reason": "Fetch source data"
    },
    {
      "qru_name": "Data",
      "priority": 2,
      "required": true,
      "reason": "Entity extraction and normalization"
    },
    {
      "qru_name": "Rules",
      "priority": 3,
      "required": true,
      "reason": "Business rule validation"
    },
    {
      "qru_name": "Algorithm",
      "priority": 4,
      "required": true,
      "reason": "Exact matching algorithm"
    },
    {
      "qru_name": "Policy",
      "priority": 5,
      "required": true,
      "reason": "Final decision fusion"
    },
    {
      "qru_name": "Action",
      "priority": 6,
      "required": true,
      "reason": "Execute resolution action"
    }
  ],
  "skipped_qrus": [
    "ML Model",
    "GenAI",
    "Assurance",
    "Orchestration",
    "Learning"
  ],
  "estimated_cost": 0.02,
  "estimated_time_seconds": 3.5,
  "reasoning": "Perfect match case with complete data - rules and algorithms sufficient"
}
```

## Decision Logic

### Case Complexity Assessment

```python
def assess_complexity(case_data):
    """
    Determine case complexity based on multiple factors
    """
    complexity_score = 0

    # Factor 1: Data completeness (0-3 points)
    missing_fields = count_missing_fields(case_data)
    if missing_fields == 0:
        complexity_score += 0
    elif missing_fields <= 2:
        complexity_score += 1
    elif missing_fields <= 5:
        complexity_score += 2
    else:
        complexity_score += 3

    # Factor 2: Data conflicts (0-3 points)
    conflicts = detect_data_conflicts(case_data)
    complexity_score += min(len(conflicts), 3)

    # Factor 3: Historical precedent (0-2 points)
    similar_cases = find_similar_cases(case_data)
    if len(similar_cases) == 0:
        complexity_score += 2  # Never seen before
    elif len(similar_cases) < 5:
        complexity_score += 1  # Few precedents

    # Factor 4: Amount variance (0-2 points)
    if has_large_amount_variance(case_data):
        complexity_score += 2

    # Classification
    if complexity_score <= 2:
        return "Simple"
    elif complexity_score <= 5:
        return "Moderate"
    elif complexity_score <= 8:
        return "Complex"
    else:
        return "Highly Complex"
```

### QRU Selection Matrix

| Complexity | Data Quality | Selected QRUs | Skip QRUs | Reasoning |
|------------|--------------|---------------|-----------|-----------|
| Simple | Complete | Retriever, Data, Rules, Algorithm, Policy, Action | ML, GenAI, Assurance, Learning | Rules and algorithms sufficient |
| Simple | Incomplete | +ML Model, +Assurance | GenAI, Learning | Need ML for missing data inference |
| Moderate | Complete | +ML Model | GenAI, Learning | Fuzzy matching needs ML |
| Moderate | Incomplete | +ML Model, +GenAI, +Assurance | Learning | Need context from LLM |
| Complex | Any | All except Learning | None | Full pipeline required |
| Highly Complex | Any | All QRUs | None | Full pipeline + learning required |

## Implementation Details

### Planner QRU Class Structure

```python
class PlannerQRU:
    """
    Meta-orchestrator that determines which QRUs to invoke for each case
    """

    def __init__(self, knowledge_substrate):
        self.knowledge = knowledge_substrate
        self.decision_tree = self._load_decision_tree()
        self.cost_model = self._load_cost_model()

    def analyze_case(self, case_data: dict) -> ExecutionPlan:
        """
        Main entry point: Analyze case and generate execution plan
        """
        # Step 1: Classify the case
        classification = self._classify_case(case_data)

        # Step 2: Select QRUs based on classification
        selected_qrus = self._select_qrus(classification, case_data)

        # Step 3: Estimate cost and time
        cost_estimate = self._estimate_cost(selected_qrus)
        time_estimate = self._estimate_time(selected_qrus)

        # Step 4: Generate execution plan
        plan = ExecutionPlan(
            case_id=case_data["case_id"],
            classification=classification,
            selected_qrus=selected_qrus,
            estimated_cost=cost_estimate,
            estimated_time=time_estimate
        )

        return plan

    def _classify_case(self, case_data: dict) -> Classification:
        """
        Classify case by complexity, data quality, etc.
        """
        complexity = self._assess_complexity(case_data)
        data_quality = self._assess_data_quality(case_data)
        reasoning_type = self._determine_reasoning_type(complexity, data_quality)

        return Classification(
            complexity=complexity,
            data_quality=data_quality,
            reasoning_type=reasoning_type
        )

    def _select_qrus(self, classification: Classification, case_data: dict) -> List[QRUSelection]:
        """
        Select which QRUs to invoke based on classification
        """
        # Always required
        selected = [
            QRUSelection("Retriever", priority=1, required=True),
            QRUSelection("Data", priority=2, required=True),
        ]

        # Conditional QRUs
        if classification.complexity in ["Simple", "Moderate"]:
            selected.append(QRUSelection("Rules", priority=3, required=True))
            selected.append(QRUSelection("Algorithm", priority=4, required=True))

        if classification.complexity in ["Moderate", "Complex", "Highly Complex"]:
            selected.append(QRUSelection("ML Model", priority=5, required=True))

        if classification.data_quality in ["Ambiguous", "Conflicting"] or \
           classification.complexity in ["Complex", "Highly Complex"]:
            selected.append(QRUSelection("GenAI", priority=6, required=True))
            selected.append(QRUSelection("Assurance", priority=7, required=True))

        # Always required at the end
        selected.append(QRUSelection("Policy", priority=98, required=True))
        selected.append(QRUSelection("Action", priority=99, required=True))

        # Learning only for new scenarios or highly complex
        if classification.complexity == "Highly Complex" or self._is_new_scenario(case_data):
            selected.append(QRUSelection("Learning", priority=100, required=False))

        return sorted(selected, key=lambda x: x.priority)
```

## Cost Optimization

### QRU Cost Model

| QRU | Cost/Invocation | Time (seconds) | Notes |
|-----|----------------|----------------|-------|
| Retriever | $0.001 | 0.5 | API calls |
| Data | $0.002 | 1.0 | NER + normalization |
| Rules | $0.0005 | 0.2 | Deterministic |
| Algorithm | $0.001 | 0.5 | Computation |
| ML Model | $0.01 | 2.0 | Model inference |
| GenAI | $0.05 | 5.0 | LLM call (GPT-4) |
| Assurance | $0.01 | 1.5 | Uncertainty quantification |
| Orchestration | $0.002 | 0.5 | Coordination |
| Policy | $0.005 | 1.0 | Thompson sampling |
| Action | $0.001 | 0.5 | API calls |
| Learning | $0.003 | 1.0 | Model update |

### Cost Savings Example

**Before Planner QRU (All QRUs invoked):**
- Total Cost: $0.001 + $0.002 + $0.0005 + $0.001 + $0.01 + $0.05 + $0.01 + $0.002 + $0.005 + $0.001 + $0.003 = **$0.0855 per case**
- Total Time: 14.2 seconds

**After Planner QRU (Simple case - 6 QRUs):**
- Total Cost: $0.001 + $0.002 + $0.0005 + $0.001 + $0.005 + $0.001 = **$0.0105 per case**
- Total Time: 3.7 seconds
- **Savings: 87.7% cost reduction, 74% time reduction**

**After Planner QRU (Complex case - All QRUs):**
- Same as before, but only when necessary

## Integration with Existing QURE

### Updated Pipeline Flow

```
1. Case Intake
   ↓
2. [Planner QRU] ← NEW
   - Analyzes case characteristics
   - Generates execution plan
   - Selects QRUs to invoke
   ↓
3. Execute Selected QRUs
   - Only invoke QRUs in the plan
   - Skip unnecessary QRUs
   ↓
4. [Policy QRU]
   - Receives signals from selected QRUs
   - Makes final decision
   ↓
5. [Action QRU]
   - Executes resolution action
```

### Modified Orchestration Logic

```python
class QUREOrchestrator:
    """
    Modified orchestrator that uses Planner QRU
    """

    def __init__(self):
        self.planner = PlannerQRU()
        self.qrus = {
            "Retriever": RetrieverQRU(),
            "Data": DataQRU(),
            "Rules": RulesQRU(),
            "Algorithm": AlgorithmQRU(),
            "ML Model": MLModelQRU(),
            "GenAI": GenAIQRU(),
            "Assurance": AssuranceQRU(),
            "Policy": PolicyQRU(),
            "Action": ActionQRU(),
            "Learning": LearningQRU()
        }

    def process_case(self, case_data: dict) -> Resolution:
        """
        Process case using Planner QRU for dynamic QRU selection
        """
        # Step 1: Generate execution plan
        plan = self.planner.analyze_case(case_data)

        # Step 2: Execute selected QRUs in order
        signals = {}
        for qru_selection in plan.selected_qrus:
            qru = self.qrus[qru_selection.qru_name]
            signal = qru.execute(case_data, signals)
            signals[qru_selection.qru_name] = signal

        # Step 3: Policy fusion (always runs)
        decision = self.qrus["Policy"].fuse_signals(signals)

        # Step 4: Execute action (always runs)
        resolution = self.qrus["Action"].execute(decision, case_data)

        # Store execution plan for audit trail
        resolution.execution_plan = plan

        return resolution
```

## Learning and Adaptation

### Planner QRU Learning Loop

The Planner QRU should learn from outcomes:

```python
def update_planner_model(self, case_id: str, plan: ExecutionPlan, outcome: Resolution):
    """
    Update Planner QRU based on actual outcomes
    """
    # Did we skip too many QRUs?
    if outcome.confidence < 0.85 and len(plan.skipped_qrus) > 3:
        self._record_underestimation(case_id, plan, outcome)
        # Next time: Invoke more QRUs for similar cases

    # Did we invoke too many QRUs?
    if outcome.confidence > 0.95 and len(plan.selected_qrus) > 6:
        self._record_overestimation(case_id, plan, outcome)
        # Next time: Skip more QRUs for similar cases

    # Update decision tree weights
    self._update_classification_model(case_id, plan, outcome)
```

## Monitoring and Metrics

### Key Metrics to Track

1. **QRU Utilization Rate**: % of cases where each QRU is invoked
2. **Average QRUs per Case**: Trend over time (should decrease with learning)
3. **Cost per Case**: Average cost with Planner vs. without
4. **Accuracy by Complexity**: Does skipping QRUs hurt accuracy?
5. **Planner Confidence**: How often is the Planner's classification correct?

### Dashboard Additions

Add to QURE dashboard:
- Planner QRU efficiency chart (cases by number of QRUs invoked)
- Cost savings chart (actual vs. baseline)
- QRU selection heatmap (which QRUs are selected together)

## Phase 5 Implementation Plan

### Phase 5a: Planner QRU Core (2 weeks)
- [ ] Implement PlannerQRU class
- [ ] Build case classification logic
- [ ] Create QRU selection decision tree
- [ ] Add cost estimation model

### Phase 5b: Integration (1 week)
- [ ] Modify QUREOrchestrator to use Planner
- [ ] Update ExecutionPlan data model
- [ ] Add Planner signals to audit trail

### Phase 5c: Learning Loop (1 week)
- [ ] Implement Planner learning from outcomes
- [ ] Add feedback mechanism for under/over-estimation
- [ ] Build classification model updater

### Phase 5d: UI & Monitoring (1 week)
- [ ] Add Planner visualization to UI
- [ ] Create QRU utilization dashboard
- [ ] Add cost savings metrics

## Success Criteria

- [ ] 70%+ reduction in average QRUs invoked per case
- [ ] 80%+ cost savings on simple/moderate cases
- [ ] No degradation in accuracy (<1% change)
- [ ] Planner classification accuracy >90%
- [ ] Execution time reduced by 60%+

## Future Enhancements

### Phase 6: Advanced Planning
- **Multi-stage planning**: Plan → Execute → Re-plan based on intermediate results
- **Parallel QRU execution**: Identify QRUs that can run in parallel
- **Adaptive thresholds**: Learn optimal confidence thresholds per case type
- **Cost-aware optimization**: Balance accuracy vs. cost based on case value

---

**Document Version**: 1.0
**Last Updated**: October 28, 2025
**Status**: Design Phase
