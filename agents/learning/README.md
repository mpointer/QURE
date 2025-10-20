# Learning Agent - Thompson Sampling Policy Optimization

## Overview

The Learning Agent implements a **contextual multi-armed bandit** using **Thompson Sampling** to continuously optimize policy weights for the QURE decision-making system. It learns from real-world outcomes to improve decision quality over time.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Learning Loop                           │
└─────────────────────────────────────────────────────────────┘

Production System → Decision Logger (JSONL)
                           ↓
                    Context Clusterer (K-means)
                           ↓
                    Reward Shaper (Multi-objective)
                           ↓
              Thompson Sampling Bandit (Bayesian)
                           ↓
                    Policy Weights → Policy Agent

Monitoring: Drift Detector (Evidently AI) → Alerts
Evaluation: Counterfactual Evaluator (IPS) → A/B Testing
```

## Components

### 1. Thompson Sampling Bandit (`thompson_sampling.py`)

**Purpose**: Select optimal policy weights using Bayesian inference

**Algorithm**:
- Maintains posterior distributions (mean, variance) for each (context, action) pair
- Samples from posteriors and selects action with highest sampled value
- Updates posteriors with observed rewards using Bayesian update

**Key Features**:
- 5 pre-defined weight configurations (balanced, rules-heavy, ML-heavy, GenAI-heavy, assurance-heavy)
- Context-specific learning (different policies for different case types)
- Automatic exploration-exploitation balance

**Usage**:
```python
from agents.learning import ThompsonSamplingBandit

bandit = ThompsonSamplingBandit(n_contexts=10)

# Select action (explore)
action_id, weights = bandit.select_action(context_id=3, explore=True)

# Observe reward and update
bandit.update(context_id=3, action_id=action_id, reward=25.0)
```

### 2. Decision Logger (`logging_pipeline.py`)

**Purpose**: Immutable append-only logging of all decisions

**Key Features**:
- JSONL format for efficient streaming
- Hash-chaining for tamper detection (SHA256)
- Propensity score logging for counterfactual evaluation
- Outcome updates (append-only, never modify original logs)

**Log Format**:
```json
{
  "timestamp": "2025-10-20T10:30:00Z",
  "case_id": "GL-20251020-0042",
  "vertical": "finance",
  "context": {
    "transaction_amount": 125000,
    "data_quality_score": 0.85,
    "urgency": "high",
    ...
  },
  "action": {
    "policy_weights": {"alpha": 0.25, "beta": 0.20, ...},
    "decision": "auto_resolve",
    "utility_score": 0.86
  },
  "propensity": 0.9,
  "outcome": {
    "correct": true,
    "reversed": false,
    "cycle_time_hours": 2,
    "cost_saved": 50
  },
  "reward": 39.6,
  "log_hash": "abc123...",
  "prev_hash": "def456..."
}
```

**Usage**:
```python
from agents.learning import DecisionLogger

logger = DecisionLogger(log_dir="data/logs/decisions")

# Log decision
log_id = logger.log_decision(
    case_id="GL-20251020-0042",
    vertical="finance",
    context={...},
    action={...},
    propensity=0.9
)

# Later: update with outcome
logger.update_outcome(
    log_id=log_id,
    outcome={"correct": True, "reversed": False, ...},
    reward=39.6
)

# Load logs for training
logs = logger.load_logs(vertical="finance", with_outcomes_only=True)
```

### 3. Reward Shaper (`reward_shaper.py`)

**Purpose**: Convert outcomes to scalar rewards for learning

**Reward Function**:
```
Reward = Accuracy + Time Savings + Cost Savings - Reversal Penalty - Risk Penalty

Where:
- Accuracy: +10 to +18 if correct (vertical-specific)
- Reversal: -20 to -40 if decision reversed
- Time Savings: +0.1 to +0.3 per hour saved vs. baseline
- Cost Savings: +0.5 to +1.0 per dollar saved
- Risk Penalty: -5 to -15 scaled by transaction amount and error severity
```

**Vertical-Specific Shapers**:
- **Finance**: High accuracy weight, strong reversal penalty (SOX compliance)
- **Insurance**: Time-critical, recovery amount bonus
- **Healthcare**: Highest accuracy weight, patient urgency bonus, strong denial penalty

**Usage**:
```python
from agents.learning import get_reward_shaper

shaper = get_reward_shaper('finance')

reward = shaper.compute_reward(
    outcome={
        'correct': True,
        'reversed': False,
        'cycle_time_hours': 2,
        'cost_saved': 50
    },
    context={'transaction_amount': 125000, 'sox_controlled': True},
    action={'decision': 'auto_resolve'}
)
# reward = 39.6 (high reward for fast, correct, compliant decision)
```

### 4. Context Clusterer (`context_clusterer.py`)

**Purpose**: Group similar contexts into clusters for contextual learning

**Algorithm**: K-means clustering on standardized context features

**Default Features**:
- `transaction_amount`: Dollar value
- `data_quality_score`: Data completeness (0-1)
- `urgency_encoded`: low=0, medium=1, high=2, critical=3
- `has_swift_reference`: Boolean
- `rule_pass_count`: Number of rules passed
- `rule_fail_count`: Number of rules failed
- `ml_confidence`: ML prediction confidence
- `genai_confidence`: GenAI prediction confidence
- `assurance_score`: Quality assurance score

**Cluster Labels** (auto-generated):
- `high_value_X`: Large transaction amounts
- `low_quality_X`: Poor data quality
- `urgent_X`: High urgency cases
- `rule_failures_X`: Multiple rule failures
- `routine_X`: Standard cases

**Usage**:
```python
from agents.learning import ContextClusterer

clusterer = ContextClusterer(n_clusters=10)

# Fit on historical contexts
clusterer.fit(contexts)

# Predict cluster for new context
cluster_id = clusterer.predict(context)
# cluster_id = 3 ("high_value_0")
```

### 5. Drift Detector (`drift_detector.py`)

**Purpose**: Monitor for data drift and performance degradation

**Uses**: Evidently AI for statistical drift detection

**Monitors**:
- **Data Drift**: Changes in context feature distributions
- **Target Drift**: Changes in reward distributions
- **Prediction Drift**: Changes in action selection patterns

**Triggers Retraining When**:
- >50% of features show significant drift
- Performance trending downward (negative slope)
- Accuracy drops below 70%
- Reversal rate exceeds 10%

**Usage**:
```python
from agents.learning import DriftDetector, run_drift_check

detector = DriftDetector(
    reference_window_days=7,
    current_window_days=1,
    drift_threshold=0.5
)

result = run_drift_check(decision_logs, detector)

if result['should_retrain']:
    print(f"⚠️ Retraining recommended: {result['retrain_reason']}")
    # Trigger learning update
```

### 6. Counterfactual Evaluator (`counterfactual_evaluator.py`)

**Purpose**: Offline policy evaluation without deployment

**Methods**:

1. **Inverse Propensity Scoring (IPS)**:
   - Reweights logged data by action probability ratio
   - Unbiased but high variance

2. **Direct Method (DM)**:
   - Uses learned value function Q(context, action)
   - Low variance but biased if model wrong

3. **Doubly Robust (DR)**:
   - Combines IPS + DM
   - Unbiased if either method correct
   - Recommended default

**Answers**:
- "What if we used ML-heavy policy instead of balanced?"
- "Would assurance-heavy policy improve rewards?"
- "Is Policy A significantly better than Policy B?"

**Usage**:
```python
from agents.learning import CounterfactualEvaluator

evaluator = CounterfactualEvaluator()

# Define policies to compare
policies = {
    'balanced': balanced_policy_fn,
    'ml_heavy': ml_heavy_policy_fn,
    'rules_heavy': rules_heavy_policy_fn
}

# Evaluate using IPS
results = evaluator.compare_policies(
    decision_logs,
    policies,
    method='ips'
)

# Statistical comparison
test = evaluator.statistical_test(
    results['ml_heavy'],
    results['balanced']
)
# → "ml_heavy wins with p=0.032"
```

### 7. Learning Agent (`learning_agent.py`)

**Purpose**: Master orchestrator for the complete learning loop

**Workflow** (nightly job):
1. Load decision logs with outcomes
2. Fit/update context clusterer
3. Compute rewards for all outcomes
4. Update Thompson Sampling bandit
5. Check for drift
6. Evaluate counterfactual policies
7. Save updated state
8. Generate report

**Integration Points**:
- **Policy Agent**: Provides policy weights via `get_policy_for_context()`
- **Monitoring**: Exposes metrics via `get_dashboard_metrics()`
- **Alerts**: Triggers on drift or performance degradation

**Usage**:
```python
from agents.learning import LearningAgent

# Initialize
agent = LearningAgent(vertical='finance')

# Log decisions in production
log_id = agent.log_decision(
    case_id="GL-20251020-0042",
    context={...},
    policy_weights={...},
    decision_type="auto_resolve",
    utility_score=0.86
)

# Later: update with outcome
agent.update_outcome(
    log_id=log_id,
    correct=True,
    reversed=False,
    cycle_time_hours=2,
    cost_saved=50
)

# Nightly: run learning update
report = agent.run_update(lookback_days=7)

# In Policy Agent: get optimal weights
weights, propensity = agent.get_policy_for_context(context)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Evidently AI (optional, for drift monitoring)
pip install evidently

# Verify installation
python -c "from agents.learning import LearningAgent; print('✓ Learning Agent installed')"
```

## Quick Start

### 1. Run Test Suite

Each component has built-in tests in its `__main__` block:

```bash
# Test Thompson Sampling
python -m agents.learning.thompson_sampling

# Test Decision Logging
python -m agents.learning.logging_pipeline

# Test Reward Computation
python -m agents.learning.reward_shaper

# Test Context Clustering
python -m agents.learning.context_clusterer

# Test Drift Detection
python -m agents.learning.drift_detector

# Test Counterfactual Evaluation
python -m agents.learning.counterfactual_evaluator
```

### 2. Run Learning Update

```bash
# Run nightly learning update
python -m agents.learning.learning_agent --vertical finance --lookback-days 7

# View dashboard metrics
python -m agents.learning.learning_agent --vertical finance --dashboard
```

### 3. Integration Example

```python
from agents.learning import LearningAgent

# Initialize agent
agent = LearningAgent(vertical='finance')

# In Policy Agent decision loop
context = extract_context(case)
weights, propensity = agent.get_policy_for_context(context, explore=True)

# Make decision using weights
decision = policy_agent.decide(case, weights)

# Log decision
log_id = agent.log_decision(
    case_id=case.id,
    context=context,
    policy_weights=weights,
    decision_type=decision.type,
    utility_score=decision.utility
)

# When outcome known (hours/days later)
agent.update_outcome(
    log_id=log_id,
    correct=outcome.correct,
    reversed=outcome.reversed,
    cycle_time_hours=outcome.cycle_time,
    cost_saved=outcome.cost_saved
)
```

## Monitoring & Dashboards

### Key Metrics

**Decision Quality**:
- Accuracy rate (% correct decisions)
- Reversal rate (% decisions reversed)
- Average reward
- Reward trend (improving/declining)

**Learning Performance**:
- Total decisions logged
- Bandit update count
- Drift detection status
- Best policy per context

**System Health**:
- Last update timestamp
- Logs processed
- Drift alerts
- Retraining triggers

### Dashboard Access

```python
from agents.learning import LearningAgent

agent = LearningAgent(vertical='finance')
metrics = agent.get_dashboard_metrics()

print(f"Last 24h: {metrics['last_24h']['total_decisions']} decisions")
print(f"Accuracy: {metrics['last_24h']['accuracy']:.1%}")
print(f"Avg Reward: {metrics['last_24h']['avg_reward']:.2f}")
```

## Configuration

### Weight Configurations

Edit default actions in `thompson_sampling.py`:

```python
def _get_default_actions(self):
    return [
        # Balanced
        {'alpha': 0.25, 'beta': 0.20, 'gamma': 0.20, 'delta': 0.20, 'lambda': 0.15},

        # Rules-heavy
        {'alpha': 0.40, 'beta': 0.15, 'gamma': 0.15, 'delta': 0.15, 'lambda': 0.15},

        # ML-heavy
        {'alpha': 0.20, 'beta': 0.20, 'gamma': 0.35, 'delta': 0.15, 'lambda': 0.10},

        # GenAI-heavy
        {'alpha': 0.20, 'beta': 0.15, 'gamma': 0.15, 'delta': 0.35, 'lambda': 0.15},

        # Assurance-heavy
        {'alpha': 0.25, 'beta': 0.15, 'gamma': 0.15, 'delta': 0.15, 'lambda': 0.30}
    ]
```

### Reward Weights

Customize reward function in `reward_shaper.py`:

```python
class FinanceRewardShaper(RewardShaper):
    def __init__(self):
        weights = RewardWeights(
            w_accuracy=15.0,       # Higher for compliance
            w_reversal=-30.0,      # Strong SOX penalty
            w_cycle_time=0.15,     # Time value
            w_cost_saved=0.8,      # Cost importance
            w_risk=-10.0,          # Risk aversion
            baseline_cycle_time=48.0
        )
        super().__init__(weights)
```

### Context Features

Customize features in `context_clusterer.py`:

```python
def _get_default_features(self):
    return [
        'transaction_amount',
        'data_quality_score',
        'urgency_encoded',
        # Add your features here
        'custom_feature_1',
        'custom_feature_2'
    ]
```

## Deployment

### Scheduled Job (Recommended)

Use cron or scheduler to run nightly updates:

```bash
# crontab entry: Run at 2 AM daily
0 2 * * * cd /path/to/qure && python -m agents.learning.learning_agent --vertical finance >> logs/learning.log 2>&1
```

### Manual Trigger

```python
from agents.learning import LearningAgent

agent = LearningAgent(vertical='finance')
report = agent.run_update(lookback_days=7, min_samples=50)

if report['should_retrain']:
    print(f"⚠️ Retraining recommended: {report['retrain_reason']}")
```

### Production Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure log directory with sufficient storage
- [ ] Set up nightly scheduled job
- [ ] Configure alerting for drift detection
- [ ] Set up monitoring dashboard
- [ ] Test full learning loop with historical data
- [ ] Verify hash chain integrity
- [ ] Document custom reward weights for your vertical

## Troubleshooting

### "Insufficient samples" Error

**Cause**: Not enough decisions with outcomes to update bandit

**Solution**:
- Lower `min_samples` threshold
- Wait for more outcomes to be logged
- Check that outcomes are being updated via `update_outcome()`

### Drift Always Detected

**Cause**: Too sensitive drift threshold

**Solution**: Increase `drift_threshold` in `DriftDetector`:
```python
detector = DriftDetector(drift_threshold=0.7)  # Default 0.5
```

### Poor Exploration

**Cause**: Bandit converging too quickly, not exploring enough

**Solution**: Increase exploration temperature or prior uncertainty:
```python
# Higher temperature = more exploration
weights, _ = agent.get_policy_for_context(context, temperature=2.0)

# Or increase prior std (before training)
bandit = ThompsonSamplingBandit(prior_std=20.0)  # Default 10.0
```

### Hash Chain Broken

**Cause**: Log file corrupted or tampered

**Solution**:
```python
from agents.learning import DecisionLogger

logger = DecisionLogger()
is_valid = logger.verify_chain()

if not is_valid:
    print("⚠️ Hash chain broken - investigate for tampering")
    # Review logs manually, restore from backup if needed
```

## Performance

**Scalability**:
- Handles 100K+ decisions efficiently
- Bandit update: O(1) per sample
- Context clustering: O(n × k × f) where n=samples, k=clusters, f=features
- Drift detection: O(n) with report generation

**Storage**:
- JSONL logs: ~1KB per decision
- 1M decisions ≈ 1GB storage
- Compress old logs with gzip for archival

**Update Frequency**:
- Recommended: Nightly (2 AM)
- Minimum: Weekly (for low-volume systems)
- Real-time learning: Not recommended (adds latency)

## References

### Thompson Sampling
- Chapelle & Li (2011). "An Empirical Evaluation of Thompson Sampling"
- Russo et al. (2018). "A Tutorial on Thompson Sampling"

### Contextual Bandits
- Li et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- Agarwal et al. (2014). "Taming the Monster: A Fast and Simple Algorithm for Contextual Bandits"

### Counterfactual Evaluation
- Dudík et al. (2014). "Doubly Robust Policy Evaluation and Optimization"
- Swaminathan & Joachims (2015). "The Self-Normalized Estimator for Counterfactual Learning"

### Drift Detection
- Evidently AI: https://evidentlyai.com/

## License

QURE Learning Agent © 2025. All rights reserved.

## Support

For issues or questions:
1. Check `PROJECT_STATUS.md` for known issues
2. Review test outputs in each module
3. Examine recent update reports in `data/learning/`
4. Contact: QURE development team
