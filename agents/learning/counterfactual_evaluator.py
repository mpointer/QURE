"""
Counterfactual Policy Evaluator

Evaluates alternative policies using logged data WITHOUT deploying them.

Uses techniques:
1. Inverse Propensity Scoring (IPS): Reweight logged data by action probability
2. Doubly Robust (DR): Combines IPS with value function for variance reduction
3. Direct Method (DM): Model-based estimation

Enables answering:
- "What if we used different policy weights?"
- "Would ML-heavy policy perform better than rules-heavy?"
- "Is the new policy actually better, or just lucky?"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PolicyEvaluation:
    """Results from evaluating a policy"""
    policy_name: str
    value_estimate: float  # Estimated average reward
    std_error: float       # Standard error of estimate
    confidence_interval: Tuple[float, float]  # 95% CI
    n_samples: int
    method: str  # 'ips', 'dr', 'dm'


class CounterfactualEvaluator:
    """
    Offline policy evaluation using logged bandit data.

    Allows evaluating new policies without deploying them,
    using historical logs from the deployed policy.
    """

    def __init__(
        self,
        clip_propensity: float = 0.01,
        confidence_level: float = 0.95
    ):
        """
        Initialize counterfactual evaluator.

        Args:
            clip_propensity: Minimum propensity to prevent extreme weights
            confidence_level: Confidence level for intervals (0-1)
        """
        self.clip_propensity = clip_propensity
        self.confidence_level = confidence_level

    def inverse_propensity_score(
        self,
        decision_logs: List[Dict],
        target_policy: callable,
        context_clusterer: Optional[callable] = None
    ) -> PolicyEvaluation:
        """
        Evaluate target policy using Inverse Propensity Scoring.

        IPS reweights logged samples by the ratio:
            π_target(a|x) / π_logged(a|x)

        Args:
            decision_logs: Logged decisions with propensities
            target_policy: Function (context) -> (action, propensity)
            context_clusterer: Optional clusterer for context features

        Returns:
            PolicyEvaluation with IPS estimate
        """
        importance_weights = []
        weighted_rewards = []

        for log in decision_logs:
            # Skip logs without outcomes
            if log.get('reward') is None:
                continue

            context = log['context']
            logged_action = log['action']
            logged_propensity = log['propensity']
            reward = log['reward']

            # Get target policy's action and propensity
            target_action, target_propensity = target_policy(context)

            # Check if target policy would have taken same action
            if self._actions_match(logged_action, target_action):
                # Importance weight
                propensity_ratio = target_propensity / max(logged_propensity, self.clip_propensity)
                importance_weights.append(propensity_ratio)
                weighted_rewards.append(propensity_ratio * reward)
            else:
                # Target policy would have taken different action
                # This sample doesn't contribute to IPS estimate
                pass

        if not weighted_rewards:
            logger.warning("No matching actions for IPS evaluation")
            return PolicyEvaluation(
                policy_name="target",
                value_estimate=0.0,
                std_error=float('inf'),
                confidence_interval=(-float('inf'), float('inf')),
                n_samples=0,
                method='ips'
            )

        # Compute value estimate
        value_estimate = np.mean(weighted_rewards)

        # Compute standard error
        # SE = sqrt(Var(importance_weight * reward) / n)
        variance = np.var(weighted_rewards, ddof=1)
        std_error = np.sqrt(variance / len(weighted_rewards))

        # Confidence interval
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        ci_lower = value_estimate - z_score * std_error
        ci_upper = value_estimate + z_score * std_error

        logger.info(
            f"IPS Evaluation: value={value_estimate:.2f} ± {std_error:.2f}, "
            f"n={len(weighted_rewards)}"
        )

        return PolicyEvaluation(
            policy_name="target",
            value_estimate=value_estimate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_samples=len(weighted_rewards),
            method='ips'
        )

    def direct_method(
        self,
        decision_logs: List[Dict],
        target_policy: callable,
        value_function: callable
    ) -> PolicyEvaluation:
        """
        Evaluate policy using Direct Method (model-based).

        DM uses a learned value function Q(context, action) to estimate
        expected reward under target policy.

        Args:
            decision_logs: Logged decisions
            target_policy: Function (context) -> (action, propensity)
            value_function: Function (context, action) -> expected_reward

        Returns:
            PolicyEvaluation with DM estimate
        """
        predicted_values = []

        for log in decision_logs:
            context = log['context']

            # Get target policy's action
            target_action, _ = target_policy(context)

            # Estimate value using learned Q-function
            estimated_value = value_function(context, target_action)
            predicted_values.append(estimated_value)

        if not predicted_values:
            logger.warning("No samples for DM evaluation")
            return PolicyEvaluation(
                policy_name="target",
                value_estimate=0.0,
                std_error=float('inf'),
                confidence_interval=(-float('inf'), float('inf')),
                n_samples=0,
                method='dm'
            )

        # Value estimate
        value_estimate = np.mean(predicted_values)

        # Standard error
        std_error = np.std(predicted_values, ddof=1) / np.sqrt(len(predicted_values))

        # Confidence interval
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        ci_lower = value_estimate - z_score * std_error
        ci_upper = value_estimate + z_score * std_error

        logger.info(
            f"DM Evaluation: value={value_estimate:.2f} ± {std_error:.2f}, "
            f"n={len(predicted_values)}"
        )

        return PolicyEvaluation(
            policy_name="target",
            value_estimate=value_estimate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_samples=len(predicted_values),
            method='dm'
        )

    def doubly_robust(
        self,
        decision_logs: List[Dict],
        target_policy: callable,
        value_function: callable
    ) -> PolicyEvaluation:
        """
        Evaluate policy using Doubly Robust estimation.

        DR combines IPS and DM:
            DR = DM + IPS * (reward - DM)

        Advantages:
        - Unbiased if EITHER model is correct
        - Lower variance than pure IPS
        - More robust than pure DM

        Args:
            decision_logs: Logged decisions
            target_policy: Function (context) -> (action, propensity)
            value_function: Function (context, action) -> expected_reward

        Returns:
            PolicyEvaluation with DR estimate
        """
        dr_values = []

        for log in decision_logs:
            if log.get('reward') is None:
                continue

            context = log['context']
            logged_action = log['action']
            logged_propensity = log['propensity']
            reward = log['reward']

            # Get target policy's action
            target_action, target_propensity = target_policy(context)

            # Direct method component
            dm_estimate = value_function(context, target_action)

            # IPS correction if actions match
            if self._actions_match(logged_action, target_action):
                propensity_ratio = target_propensity / max(logged_propensity, self.clip_propensity)
                ips_correction = propensity_ratio * (reward - dm_estimate)
            else:
                ips_correction = 0

            # Doubly robust estimate for this sample
            dr_value = dm_estimate + ips_correction
            dr_values.append(dr_value)

        if not dr_values:
            logger.warning("No samples for DR evaluation")
            return PolicyEvaluation(
                policy_name="target",
                value_estimate=0.0,
                std_error=float('inf'),
                confidence_interval=(-float('inf'), float('inf')),
                n_samples=0,
                method='dr'
            )

        # Value estimate
        value_estimate = np.mean(dr_values)

        # Standard error
        std_error = np.std(dr_values, ddof=1) / np.sqrt(len(dr_values))

        # Confidence interval
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        ci_lower = value_estimate - z_score * std_error
        ci_upper = value_estimate + z_score * std_error

        logger.info(
            f"DR Evaluation: value={value_estimate:.2f} ± {std_error:.2f}, "
            f"n={len(dr_values)}"
        )

        return PolicyEvaluation(
            policy_name="target",
            value_estimate=value_estimate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_samples=len(dr_values),
            method='dr'
        )

    def compare_policies(
        self,
        decision_logs: List[Dict],
        policies: Dict[str, callable],
        method: str = 'ips',
        value_function: Optional[callable] = None
    ) -> Dict[str, PolicyEvaluation]:
        """
        Compare multiple policies using offline evaluation.

        Args:
            decision_logs: Logged decisions
            policies: Dict mapping policy names to policy functions
            method: Evaluation method ('ips', 'dm', 'dr')
            value_function: Required for 'dm' and 'dr' methods

        Returns:
            Dict mapping policy names to evaluations
        """
        results = {}

        for policy_name, policy_fn in policies.items():
            logger.info(f"Evaluating policy: {policy_name}")

            if method == 'ips':
                eval_result = self.inverse_propensity_score(
                    decision_logs, policy_fn
                )
            elif method == 'dm':
                if value_function is None:
                    raise ValueError("value_function required for DM method")
                eval_result = self.direct_method(
                    decision_logs, policy_fn, value_function
                )
            elif method == 'dr':
                if value_function is None:
                    raise ValueError("value_function required for DR method")
                eval_result = self.doubly_robust(
                    decision_logs, policy_fn, value_function
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            eval_result.policy_name = policy_name
            results[policy_name] = eval_result

        # Rank policies
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].value_estimate,
            reverse=True
        )

        logger.info("\n=== Policy Ranking ===")
        for i, (name, eval_result) in enumerate(ranked):
            logger.info(
                f"{i+1}. {name}: {eval_result.value_estimate:.2f} "
                f"± {eval_result.std_error:.2f}"
            )

        return results

    def statistical_test(
        self,
        eval_a: PolicyEvaluation,
        eval_b: PolicyEvaluation
    ) -> Dict:
        """
        Statistical test comparing two policies.

        Args:
            eval_a: Evaluation for policy A
            eval_b: Evaluation for policy B

        Returns:
            Dict with test results
        """
        # Difference in means
        diff = eval_a.value_estimate - eval_b.value_estimate

        # Standard error of difference
        se_diff = np.sqrt(eval_a.std_error**2 + eval_b.std_error**2)

        # Z-test
        z_stat = diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence interval for difference
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        ci_diff_lower = diff - z_score * se_diff
        ci_diff_upper = diff + z_score * se_diff

        significant = p_value < (1 - self.confidence_level)
        winner = eval_a.policy_name if diff > 0 else eval_b.policy_name

        result = {
            'policy_a': eval_a.policy_name,
            'policy_b': eval_b.policy_name,
            'difference': diff,
            'std_error_diff': se_diff,
            'confidence_interval': (ci_diff_lower, ci_diff_upper),
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': significant,
            'winner': winner if significant else 'tie'
        }

        logger.info(
            f"Statistical test: {eval_a.policy_name} vs {eval_b.policy_name}"
        )
        logger.info(f"  Difference: {diff:.2f} ± {se_diff:.2f}")
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Result: {result['winner']}")

        return result

    def _actions_match(self, action_a: Dict, action_b: Dict) -> bool:
        """
        Check if two actions are the same.

        Args:
            action_a: First action dict
            action_b: Second action dict

        Returns:
            True if actions match
        """
        # Compare decision types
        if action_a.get('decision') != action_b.get('decision'):
            return False

        # Compare policy weights (if present)
        weights_a = action_a.get('policy_weights', {})
        weights_b = action_b.get('policy_weights', {})

        if not weights_a or not weights_b:
            return True  # No weights to compare

        # Check if weights are approximately equal
        for key in weights_a:
            if abs(weights_a.get(key, 0) - weights_b.get(key, 0)) > 0.01:
                return False

        return True


# Utility functions

def create_simple_value_function(decision_logs: List[Dict]) -> callable:
    """
    Create a simple value function from logged data.

    Uses average reward for each (context_type, action_type) pair.

    Args:
        decision_logs: Training data

    Returns:
        value_function(context, action) -> expected_reward
    """
    # Build lookup table
    value_table = {}

    for log in decision_logs:
        if log.get('reward') is None:
            continue

        # Simple hashing of context and action
        context = log['context']
        action = log['action']
        reward = log['reward']

        # Create keys
        context_key = _hash_context(context)
        action_key = action.get('decision', 'unknown')
        key = (context_key, action_key)

        if key not in value_table:
            value_table[key] = []
        value_table[key].append(reward)

    # Average rewards
    avg_value_table = {
        key: np.mean(rewards)
        for key, rewards in value_table.items()
    }

    # Default value (overall mean)
    default_value = np.mean([
        reward
        for log in decision_logs
        if log.get('reward') is not None
        for reward in [log['reward']]
    ])

    def value_function(context: Dict, action: Dict) -> float:
        """Estimate expected reward"""
        context_key = _hash_context(context)
        action_key = action.get('decision', 'unknown')
        key = (context_key, action_key)
        return avg_value_table.get(key, default_value)

    return value_function


def _hash_context(context: Dict) -> str:
    """Simple context hashing for value function"""
    # Bin transaction amount
    amount = context.get('transaction_amount', 0)
    if amount > 500_000:
        amount_bin = 'high'
    elif amount > 100_000:
        amount_bin = 'medium'
    else:
        amount_bin = 'low'

    # Bin quality
    quality = context.get('data_quality_score', 0.5)
    quality_bin = 'high' if quality > 0.8 else 'medium' if quality > 0.5 else 'low'

    urgency = context.get('urgency', 'medium')

    return f"{amount_bin}_{quality_bin}_{urgency}"


if __name__ == "__main__":
    # Test counterfactual evaluation
    logging.basicConfig(level=logging.INFO)

    print("=== Counterfactual Evaluator Test ===\n")

    # Generate synthetic logged data
    np.random.seed(42)

    # Logged policy: Balanced weights
    logged_policy_weights = {
        'alpha': 0.25, 'beta': 0.20, 'gamma': 0.20, 'delta': 0.20, 'lambda': 0.15
    }

    decision_logs = []
    for i in range(200):
        # Random context
        amount = np.random.uniform(10_000, 500_000)
        quality = np.random.uniform(0.4, 0.95)
        urgency = np.random.choice(['low', 'medium', 'high'])

        context = {
            'transaction_amount': amount,
            'data_quality_score': quality,
            'urgency': urgency
        }

        # Logged action (balanced policy)
        action = {
            'decision': 'auto_resolve',
            'policy_weights': logged_policy_weights
        }

        # Simulate reward (higher for high-quality)
        base_reward = 20 if quality > 0.7 else 10
        reward = base_reward + np.random.normal(0, 5)

        decision_logs.append({
            'context': context,
            'action': action,
            'propensity': 0.8,  # Logged policy probability
            'reward': reward
        })

    print(f"Generated {len(decision_logs)} logged decisions\n")

    # Define alternative policies to evaluate

    def balanced_policy(context):
        """Baseline: balanced weights"""
        weights = {'alpha': 0.25, 'beta': 0.20, 'gamma': 0.20, 'delta': 0.20, 'lambda': 0.15}
        action = {'decision': 'auto_resolve', 'policy_weights': weights}
        return action, 0.8

    def ml_heavy_policy(context):
        """ML-heavy: emphasize ML confidence"""
        weights = {'alpha': 0.20, 'beta': 0.20, 'gamma': 0.35, 'delta': 0.15, 'lambda': 0.10}
        action = {'decision': 'auto_resolve', 'policy_weights': weights}
        # Higher propensity for high-quality data
        prop = 0.9 if context.get('data_quality_score', 0.5) > 0.7 else 0.5
        return action, prop

    def rules_heavy_policy(context):
        """Rules-heavy: conservative"""
        weights = {'alpha': 0.40, 'beta': 0.15, 'gamma': 0.15, 'delta': 0.15, 'lambda': 0.15}
        action = {'decision': 'auto_resolve', 'policy_weights': weights}
        return action, 0.7

    # Evaluate policies
    evaluator = CounterfactualEvaluator()

    policies = {
        'balanced': balanced_policy,
        'ml_heavy': ml_heavy_policy,
        'rules_heavy': rules_heavy_policy
    }

    print("Evaluating policies using IPS...\n")
    results = evaluator.compare_policies(decision_logs, policies, method='ips')

    # Display results
    print("\n=== Policy Evaluation Results ===")
    for name, eval_result in results.items():
        ci_lower, ci_upper = eval_result.confidence_interval
        print(f"\n{name}:")
        print(f"  Value: {eval_result.value_estimate:.2f} ± {eval_result.std_error:.2f}")
        print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Samples: {eval_result.n_samples}")

    # Statistical comparison
    print("\n=== Statistical Tests ===")
    test_result = evaluator.statistical_test(
        results['ml_heavy'],
        results['balanced']
    )
    print(f"\nML-heavy vs Balanced:")
    print(f"  Difference: {test_result['difference']:.2f}")
    print(f"  P-value: {test_result['p_value']:.4f}")
    print(f"  Winner: {test_result['winner']}")

    print("\n✓ Counterfactual evaluation test complete")
