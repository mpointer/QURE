"""
Reward Shaper

Computes rewards from decision outcomes for reinforcement learning.
Reward = Benefit - Cost - Risk Penalty
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    """Weights for reward computation components"""
    w_accuracy: float = 10.0      # Reward for correct decisions
    w_reversal: float = -20.0     # Penalty for reversals
    w_cycle_time: float = 0.1     # Reward per hour saved
    w_cost_saved: float = 0.5     # Reward per dollar saved
    w_risk: float = -5.0          # Risk penalty multiplier
    baseline_cycle_time: float = 48.0  # Baseline manual process time (hours)


class RewardShaper:
    """
    Computes rewards from decision outcomes.

    Reward function balances multiple objectives:
    1. Accuracy: Did we make the right decision?
    2. Efficiency: How fast did we resolve it?
    3. Cost: How much did we save vs. manual process?
    4. Risk: Penalty for errors on high-value transactions

    The reward should reflect business value:
    - Correct auto-resolution: HIGH reward (saved time + cost)
    - Incorrect auto-resolution: NEGATIVE reward (reversal cost + risk)
    - Correct escalation: SMALL reward (avoided error, but no time savings)
    - Incorrect escalation: SMALL penalty (wasted manual effort)
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        """
        Initialize reward shaper.

        Args:
            weights: Custom reward weights (default: balanced weights)
        """
        self.weights = weights or RewardWeights()

    def compute_reward(
        self,
        outcome: Dict,
        context: Dict,
        action: Dict
    ) -> float:
        """
        Compute reward from outcome.

        Args:
            outcome: Observed outcome dict
                {
                    'correct': bool,
                    'reversed': bool,
                    'cycle_time_hours': float,
                    'cost_saved': float,
                    'manual_review_required': bool (optional),
                    'error_severity': str (optional: low/medium/high/critical)
                }
            context: Context at decision time
                {
                    'transaction_amount': float,
                    'vertical': str,
                    'urgency': str,
                    'data_quality_score': float,
                    ...
                }
            action: Action taken
                {
                    'decision': str (auto_resolve/hitl_review/request_info/...),
                    'utility_score': float,
                    ...
                }

        Returns:
            reward: Scalar reward value
        """
        w = self.weights

        # Base accuracy component
        accuracy_reward = w.w_accuracy if outcome.get('correct', False) else 0.0

        # Reversal penalty (strong negative signal)
        reversal_penalty = w.w_reversal if outcome.get('reversed', False) else 0.0

        # Cycle time savings (vs. baseline manual process)
        cycle_time_hours = outcome.get('cycle_time_hours', w.baseline_cycle_time)
        time_saved = max(0, w.baseline_cycle_time - cycle_time_hours)
        time_reward = w.w_cycle_time * time_saved

        # Cost savings
        cost_saved = outcome.get('cost_saved', 0)
        cost_reward = w.w_cost_saved * cost_saved

        # Risk-adjusted penalty for high-value errors
        risk_penalty = 0.0
        if not outcome.get('correct', True):
            # Scale penalty by transaction amount
            transaction_amount = context.get('transaction_amount', 0)
            risk_factor = transaction_amount / 100000  # Normalize by $100k

            # Scale by error severity
            severity = outcome.get('error_severity', 'medium')
            severity_multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 2.0,
                'critical': 5.0
            }
            severity_mult = severity_multipliers.get(severity, 1.0)

            risk_penalty = w.w_risk * risk_factor * severity_mult

        # Total reward
        reward = (
            accuracy_reward +
            reversal_penalty +
            time_reward +
            cost_reward +
            risk_penalty
        )

        # Log reward breakdown
        logger.debug(
            f"Reward breakdown: "
            f"accuracy={accuracy_reward:.2f}, "
            f"reversal={reversal_penalty:.2f}, "
            f"time={time_reward:.2f}, "
            f"cost={cost_reward:.2f}, "
            f"risk={risk_penalty:.2f}, "
            f"total={reward:.2f}"
        )

        return reward

    def compute_batch_rewards(
        self,
        outcomes: list,
        contexts: list,
        actions: list
    ) -> list:
        """
        Compute rewards for a batch of outcomes.

        Args:
            outcomes: List of outcome dicts
            contexts: List of context dicts
            actions: List of action dicts

        Returns:
            rewards: List of scalar rewards
        """
        rewards = []
        for outcome, context, action in zip(outcomes, contexts, actions):
            reward = self.compute_reward(outcome, context, action)
            rewards.append(reward)

        return rewards

    def normalize_rewards(
        self,
        rewards: list,
        method: str = 'zscore'
    ) -> list:
        """
        Normalize rewards for stable learning.

        Args:
            rewards: List of rewards
            method: Normalization method ('zscore', 'minmax', or 'none')

        Returns:
            normalized_rewards: List of normalized rewards
        """
        if method == 'none':
            return rewards

        import numpy as np

        rewards_array = np.array(rewards)

        if method == 'zscore':
            # Z-score normalization
            mean = rewards_array.mean()
            std = rewards_array.std()
            if std > 0:
                normalized = (rewards_array - mean) / std
            else:
                normalized = rewards_array - mean

        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = rewards_array.min()
            max_val = rewards_array.max()
            if max_val > min_val:
                normalized = (rewards_array - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(rewards_array)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.tolist()


# Specialized reward functions for different verticals

class FinanceRewardShaper(RewardShaper):
    """Reward shaper optimized for finance reconciliation"""

    def __init__(self):
        weights = RewardWeights(
            w_accuracy=15.0,       # Higher accuracy weight for compliance
            w_reversal=-30.0,      # Strong penalty for SOX violations
            w_cycle_time=0.15,     # Time is valuable in month-end close
            w_cost_saved=0.8,      # Cost savings important
            w_risk=-10.0,          # High risk penalty for financial errors
            baseline_cycle_time=48.0
        )
        super().__init__(weights)

    def compute_reward(self, outcome: Dict, context: Dict, action: Dict) -> float:
        # Base reward
        reward = super().compute_reward(outcome, context, action)

        # Additional SOX compliance bonus
        if outcome.get('correct', False) and context.get('sox_controlled', False):
            reward += 5.0  # Bonus for correct SOX-controlled transaction

        # Additional penalty for high-value errors
        if not outcome.get('correct', True):
            amount = context.get('transaction_amount', 0)
            if amount > 1_000_000:
                reward -= 20.0  # Extra penalty for $1M+ errors

        return reward


class InsuranceRewardShaper(RewardShaper):
    """Reward shaper optimized for insurance subrogation"""

    def __init__(self):
        weights = RewardWeights(
            w_accuracy=12.0,
            w_reversal=-25.0,
            w_cycle_time=0.2,      # Time critical for statute of limitations
            w_cost_saved=1.0,      # Recovery amount is key metric
            w_risk=-8.0,
            baseline_cycle_time=120.0  # Subro takes longer baseline
        )
        super().__init__(weights)

    def compute_reward(self, outcome: Dict, context: Dict, action: Dict) -> float:
        # Base reward
        reward = super().compute_reward(outcome, context, action)

        # Recovery success bonus
        recovery_amount = outcome.get('recovery_amount', 0)
        if recovery_amount > 0:
            reward += 0.1 * recovery_amount  # 10% of recovery as bonus

        # Statute of limitations urgency
        days_to_statute = context.get('days_to_statute_expiry', 999)
        if days_to_statute < 30 and outcome.get('correct', False):
            reward += 10.0  # Bonus for urgent cases handled correctly

        return reward


class HealthcareRewardShaper(RewardShaper):
    """Reward shaper optimized for healthcare prior authorization"""

    def __init__(self):
        weights = RewardWeights(
            w_accuracy=18.0,       # Patient care accuracy is paramount
            w_reversal=-40.0,      # Strong penalty for denying needed care
            w_cycle_time=0.3,      # Time critical for patient care
            w_cost_saved=0.3,      # Cost less important than accuracy
            w_risk=-15.0,          # High risk for patient safety
            baseline_cycle_time=72.0
        )
        super().__init__(weights)

    def compute_reward(self, outcome: Dict, context: Dict, action: Dict) -> float:
        # Base reward
        reward = super().compute_reward(outcome, context, action)

        # Patient urgency bonus
        urgency = context.get('clinical_urgency', 'routine')
        if urgency == 'urgent' and outcome.get('correct', False):
            reward += 15.0  # Bonus for urgent cases handled correctly

        # False denial penalty (denied but should have approved)
        if outcome.get('appeal_overturned', False):
            reward -= 30.0  # Strong penalty for incorrect denials

        return reward


def get_reward_shaper(vertical: str) -> RewardShaper:
    """
    Get appropriate reward shaper for vertical.

    Args:
        vertical: Business vertical (finance, insurance, healthcare, etc.)

    Returns:
        RewardShaper instance
    """
    shapers = {
        'finance': FinanceRewardShaper,
        'insurance': InsuranceRewardShaper,
        'healthcare': HealthcareRewardShaper
    }

    shaper_class = shapers.get(vertical, RewardShaper)
    return shaper_class()


if __name__ == "__main__":
    # Test reward computation
    logging.basicConfig(level=logging.DEBUG)

    print("=== Reward Shaper Test ===\n")

    # Test case 1: Correct auto-resolution
    print("Test 1: Correct auto-resolution (high-value transaction)")
    shaper = FinanceRewardShaper()

    outcome = {
        'correct': True,
        'reversed': False,
        'cycle_time_hours': 2,
        'cost_saved': 50
    }

    context = {
        'transaction_amount': 125_000,
        'vertical': 'finance',
        'sox_controlled': True
    }

    action = {
        'decision': 'auto_resolve',
        'utility_score': 0.86
    }

    reward = shaper.compute_reward(outcome, context, action)
    print(f"Reward: {reward:.2f}\n")

    # Test case 2: Incorrect auto-resolution (should have escalated)
    print("Test 2: Incorrect auto-resolution (error on $1M+ transaction)")
    outcome = {
        'correct': False,
        'reversed': True,
        'cycle_time_hours': 2,
        'cost_saved': 0,
        'error_severity': 'high'
    }

    context = {
        'transaction_amount': 1_500_000,
        'vertical': 'finance',
        'sox_controlled': True
    }

    reward = shaper.compute_reward(outcome, context, action)
    print(f"Reward: {reward:.2f}\n")

    # Test case 3: Correct escalation (avoided error)
    print("Test 3: Correct escalation (low confidence, escalated to human)")
    outcome = {
        'correct': True,
        'reversed': False,
        'cycle_time_hours': 24,  # Manual review took time
        'cost_saved': 0  # No cost savings vs. baseline
    }

    context = {
        'transaction_amount': 500_000,
        'vertical': 'finance',
        'sox_controlled': False,
        'data_quality_score': 0.4  # Low quality -> correct to escalate
    }

    action = {
        'decision': 'hitl_review',
        'utility_score': 0.55
    }

    reward = shaper.compute_reward(outcome, context, action)
    print(f"Reward: {reward:.2f}\n")

    # Test batch computation
    print("Test 4: Batch reward computation")
    outcomes = [outcome] * 3
    contexts = [context] * 3
    actions = [action] * 3

    rewards = shaper.compute_batch_rewards(outcomes, contexts, actions)
    print(f"Batch rewards: {rewards}\n")

    # Test normalization
    print("Test 5: Reward normalization")
    raw_rewards = [50.0, 25.0, -10.0, 5.0, -30.0]
    normalized = shaper.normalize_rewards(raw_rewards, method='zscore')
    print(f"Raw: {raw_rewards}")
    print(f"Normalized: {[f'{r:.2f}' for r in normalized]}\n")
