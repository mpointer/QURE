"""
Thompson Sampling Multi-Armed Bandit for Policy Weight Optimization

Implements Bayesian multi-armed bandit algorithm to learn optimal policy weights
for different context types through exploration-exploitation tradeoff.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BanditState:
    """State of the Thompson Sampling bandit"""
    means: np.ndarray  # [n_contexts, n_actions] - Mean reward estimates
    stds: np.ndarray   # [n_contexts, n_actions] - Uncertainty (std dev)
    counts: np.ndarray # [n_contexts, n_actions] - Number of trials

    # Action definitions (weight configurations)
    actions: List[Dict[str, float]]

    # Context cluster labels
    context_labels: List[str]

    # Metadata
    last_updated: str
    total_decisions: int


class ThompsonSamplingBandit:
    """
    Thompson Sampling for contextual bandits.

    Uses Bayesian inference to maintain probability distributions over
    expected rewards for each (context, action) pair.

    Algorithm:
    1. For each decision, sample from posterior distributions
    2. Choose action with highest sampled value
    3. Observe reward
    4. Update posterior using Bayesian update
    """

    def __init__(
        self,
        n_contexts: int = 10,
        context_labels: Optional[List[str]] = None,
        actions: Optional[List[Dict[str, float]]] = None,
        prior_mean: float = 0.0,
        prior_std: float = 10.0,
        state_path: Optional[str] = None
    ):
        """
        Initialize Thompson Sampling bandit.

        Args:
            n_contexts: Number of context clusters
            context_labels: Human-readable labels for contexts
            actions: List of weight configurations to test
            prior_mean: Prior mean for all actions (neutral prior)
            prior_std: Prior uncertainty (high = more exploration)
            state_path: Path to save/load bandit state
        """
        self.n_contexts = n_contexts
        self.state_path = state_path or "data/bandit_state.json"

        # Default actions: 5 different weight configurations
        if actions is None:
            actions = self._get_default_actions()
        self.actions = actions
        self.n_actions = len(actions)

        # Context labels
        if context_labels is None:
            context_labels = [f"context_{i}" for i in range(n_contexts)]
        self.context_labels = context_labels

        # Initialize priors
        self.means = np.full((n_contexts, self.n_actions), prior_mean)
        self.stds = np.full((n_contexts, self.n_actions), prior_std)
        self.counts = np.zeros((n_contexts, self.n_actions))

        self.total_decisions = 0

        # Try to load existing state
        self._load_state()

    def _get_default_actions(self) -> List[Dict[str, float]]:
        """
        Default weight configurations to explore.

        Returns 5 strategies:
        1. Balanced: Equal weights
        2. Rules-heavy: Conservative, rule-based
        3. ML-heavy: Data-driven, ML-focused
        4. GenAI-heavy: Semantic understanding emphasis
        5. Assurance-heavy: Risk-averse, high scrutiny
        """
        return [
            # Action 0: Balanced (baseline)
            {
                'alpha': 0.25,   # rules
                'beta': 0.20,    # algorithm
                'gamma': 0.20,   # ml
                'delta': 0.20,   # genai
                'lambda': 0.15   # assurance
            },
            # Action 1: Rules-heavy (conservative)
            {
                'alpha': 0.40,
                'beta': 0.15,
                'gamma': 0.15,
                'delta': 0.15,
                'lambda': 0.15
            },
            # Action 2: ML-heavy (data-driven)
            {
                'alpha': 0.20,
                'beta': 0.20,
                'gamma': 0.35,
                'delta': 0.15,
                'lambda': 0.10
            },
            # Action 3: GenAI-heavy (semantic)
            {
                'alpha': 0.20,
                'beta': 0.15,
                'gamma': 0.15,
                'delta': 0.35,
                'lambda': 0.15
            },
            # Action 4: Assurance-heavy (risk-averse)
            {
                'alpha': 0.25,
                'beta': 0.15,
                'gamma': 0.15,
                'delta': 0.15,
                'lambda': 0.30
            }
        ]

    def select_action(
        self,
        context_id: int,
        explore: bool = True,
        temperature: float = 1.0
    ) -> Tuple[int, Dict[str, float]]:
        """
        Select action using Thompson Sampling.

        Args:
            context_id: Context cluster ID (0 to n_contexts-1)
            explore: If False, use greedy selection (exploit only)
            temperature: Controls exploration (higher = more random)

        Returns:
            (action_id, weights_dict)
        """
        if not explore:
            # Greedy: choose best mean
            action_id = int(np.argmax(self.means[context_id]))
        else:
            # Thompson Sampling: sample from posteriors
            sampled_values = np.random.normal(
                self.means[context_id],
                self.stds[context_id] * temperature
            )
            action_id = int(np.argmax(sampled_values))

        weights = self.actions[action_id]

        logger.info(
            f"Selected action {action_id} for context {context_id} "
            f"({self.context_labels[context_id]}): {weights}"
        )

        return action_id, weights

    def update(
        self,
        context_id: int,
        action_id: int,
        reward: float,
        learning_rate: Optional[float] = None
    ):
        """
        Update bandit posterior after observing reward.

        Uses Bayesian update with Gaussian likelihood:
        - New mean: weighted average of prior and observation
        - New variance: decreases with more observations

        Args:
            context_id: Context cluster ID
            action_id: Action taken
            reward: Observed reward
            learning_rate: If provided, use exponential moving average
        """
        n = self.counts[context_id, action_id]
        old_mean = self.means[context_id, action_id]
        old_std = self.stds[context_id, action_id]

        if learning_rate is not None:
            # Exponential moving average (for non-stationary rewards)
            new_mean = (1 - learning_rate) * old_mean + learning_rate * reward
            # Keep variance constant with EMA
            new_std = old_std
        else:
            # Bayesian update (assumes stationary rewards)
            # Posterior mean: weighted average
            new_mean = (old_mean * n + reward) / (n + 1)

            # Posterior variance: decreases with more samples
            # Using conjugate Gaussian-Gaussian update
            new_var = (old_std ** 2) * n / (n + 1)
            new_std = np.sqrt(max(new_var, 0.1))  # Floor at 0.1 to maintain exploration

        self.means[context_id, action_id] = new_mean
        self.stds[context_id, action_id] = new_std
        self.counts[context_id, action_id] += 1
        self.total_decisions += 1

        logger.debug(
            f"Updated context {context_id}, action {action_id}: "
            f"mean {old_mean:.3f} -> {new_mean:.3f}, "
            f"std {old_std:.3f} -> {new_std:.3f}, "
            f"reward {reward:.3f}"
        )

    def batch_update(
        self,
        contexts: List[int],
        actions: List[int],
        rewards: List[float]
    ):
        """
        Update bandit with batch of observations.

        Args:
            contexts: List of context IDs
            actions: List of action IDs
            rewards: List of observed rewards
        """
        for context_id, action_id, reward in zip(contexts, actions, rewards):
            self.update(context_id, action_id, reward)

        logger.info(f"Batch updated {len(rewards)} observations")

    def get_best_action(self, context_id: int) -> Tuple[int, Dict[str, float]]:
        """
        Get best action for context (greedy, no exploration).

        Args:
            context_id: Context cluster ID

        Returns:
            (action_id, weights_dict)
        """
        return self.select_action(context_id, explore=False)

    def get_statistics(self) -> Dict:
        """
        Get bandit statistics for monitoring.

        Returns:
            Dict with means, uncertainties, counts, and rankings
        """
        stats = {
            'total_decisions': int(self.total_decisions),
            'contexts': []
        }

        for ctx_id in range(self.n_contexts):
            ctx_stats = {
                'id': ctx_id,
                'label': self.context_labels[ctx_id],
                'actions': []
            }

            for act_id in range(self.n_actions):
                ctx_stats['actions'].append({
                    'id': act_id,
                    'weights': self.actions[act_id],
                    'mean_reward': float(self.means[ctx_id, act_id]),
                    'std_reward': float(self.stds[ctx_id, act_id]),
                    'trials': int(self.counts[ctx_id, act_id]),
                    'confidence_interval': [
                        float(self.means[ctx_id, act_id] - 1.96 * self.stds[ctx_id, act_id]),
                        float(self.means[ctx_id, act_id] + 1.96 * self.stds[ctx_id, act_id])
                    ]
                })

            # Rank actions by mean reward
            ranked = sorted(
                ctx_stats['actions'],
                key=lambda x: x['mean_reward'],
                reverse=True
            )
            ctx_stats['best_action'] = ranked[0]
            ctx_stats['actions'] = ranked

            stats['contexts'].append(ctx_stats)

        return stats

    def save_state(self, path: Optional[str] = None):
        """
        Save bandit state to disk.

        Args:
            path: Path to save state (default: self.state_path)
        """
        path = path or self.state_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'means': self.means.tolist(),
            'stds': self.stds.tolist(),
            'counts': self.counts.tolist(),
            'actions': self.actions,
            'context_labels': self.context_labels,
            'total_decisions': self.total_decisions,
            'n_contexts': self.n_contexts,
            'n_actions': self.n_actions
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved bandit state to {path}")

    def _load_state(self):
        """Load bandit state from disk if it exists."""
        if not Path(self.state_path).exists():
            logger.info(f"No existing state found at {self.state_path}, using priors")
            return

        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            self.means = np.array(state['means'])
            self.stds = np.array(state['stds'])
            self.counts = np.array(state['counts'])
            self.actions = state['actions']
            self.context_labels = state['context_labels']
            self.total_decisions = state['total_decisions']
            self.n_contexts = state['n_contexts']
            self.n_actions = state['n_actions']

            logger.info(
                f"Loaded bandit state from {self.state_path} "
                f"({self.total_decisions} total decisions)"
            )
        except Exception as e:
            logger.error(f"Failed to load bandit state: {e}, using priors")


# Utility functions

def simulate_bandit(
    n_contexts: int = 3,
    n_steps: int = 1000,
    true_rewards: Optional[np.ndarray] = None
) -> Dict:
    """
    Simulate Thompson Sampling to verify it works.

    Args:
        n_contexts: Number of contexts
        n_steps: Number of simulation steps
        true_rewards: [n_contexts, n_actions] true mean rewards (if None, random)

    Returns:
        Dict with cumulative regret and selected actions
    """
    bandit = ThompsonSamplingBandit(n_contexts=n_contexts)
    n_actions = bandit.n_actions

    # Generate true reward distributions if not provided
    if true_rewards is None:
        true_rewards = np.random.uniform(-5, 15, (n_contexts, n_actions))

    # Track cumulative regret
    cumulative_regret = 0
    regrets = []
    actions_taken = []

    for step in range(n_steps):
        # Random context
        context_id = np.random.randint(0, n_contexts)

        # Select action
        action_id, _ = bandit.select_action(context_id)

        # Observe reward (noisy)
        true_mean = true_rewards[context_id, action_id]
        reward = np.random.normal(true_mean, 2.0)  # Add noise

        # Update bandit
        bandit.update(context_id, action_id, reward)

        # Compute regret (vs. optimal action)
        optimal_reward = true_rewards[context_id].max()
        regret = optimal_reward - true_mean
        cumulative_regret += regret

        regrets.append(cumulative_regret)
        actions_taken.append(action_id)

    return {
        'cumulative_regret': regrets,
        'actions_taken': actions_taken,
        'true_rewards': true_rewards.tolist(),
        'final_means': bandit.means.tolist(),
        'final_stds': bandit.stds.tolist()
    }


if __name__ == "__main__":
    # Test Thompson Sampling
    logging.basicConfig(level=logging.INFO)

    print("=== Thompson Sampling Bandit Test ===\n")

    # Create bandit with 3 contexts
    context_labels = ['high_value', 'routine', 'low_quality_data']
    bandit = ThompsonSamplingBandit(
        n_contexts=3,
        context_labels=context_labels
    )

    print(f"Initialized bandit with {bandit.n_actions} actions")
    print(f"Actions: {[f'Action {i}' for i in range(bandit.n_actions)]}\n")

    # Simulate 100 decisions
    print("Simulating 100 decisions...\n")

    # True optimal actions (hidden from bandit)
    optimal_actions = {
        0: 1,  # high_value -> Rules-heavy
        1: 2,  # routine -> ML-heavy
        2: 4   # low_quality_data -> Assurance-heavy
    }

    for step in range(100):
        context_id = step % 3

        # Select action
        action_id, weights = bandit.select_action(context_id)

        # Simulate reward (better if optimal, worse if not)
        if action_id == optimal_actions[context_id]:
            reward = np.random.normal(10, 2)  # Good action
        else:
            reward = np.random.normal(5, 2)   # Suboptimal action

        # Update bandit
        bandit.update(context_id, action_id, reward)

    # Show learned strategy
    print("\n=== Learned Strategy ===\n")
    stats = bandit.get_statistics()

    for ctx in stats['contexts']:
        print(f"Context: {ctx['label']}")
        print(f"  Best action: Action {ctx['best_action']['id']}")
        print(f"  Mean reward: {ctx['best_action']['mean_reward']:.2f}")
        print(f"  Trials: {ctx['best_action']['trials']}")
        print(f"  Weights: {ctx['best_action']['weights']}")
        print()

    # Save state
    bandit.save_state('data/test_bandit_state.json')
    print("Saved bandit state to data/test_bandit_state.json")
