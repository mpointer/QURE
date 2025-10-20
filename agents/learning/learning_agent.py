"""
Learning Agent - Thompson Sampling Policy Optimization

Master orchestrator that implements the complete learning loop:
1. Load decision logs from production
2. Cluster contexts into types
3. Compute rewards from outcomes
4. Update Thompson Sampling bandit
5. Check for drift
6. Evaluate counterfactual policies
7. Deploy optimal policy weights

Runs nightly to continuously improve policy performance.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# Import our components
from .thompson_sampling import ThompsonSamplingBandit
from .logging_pipeline import DecisionLogger, merge_outcome_to_decision
from .reward_shaper import RewardShaper, get_reward_shaper
from .context_clusterer import ContextClusterer
from .drift_detector import DriftDetector, run_drift_check
from .counterfactual_evaluator import CounterfactualEvaluator, create_simple_value_function

logger = logging.getLogger(__name__)


class LearningAgent:
    """
    Main learning agent that coordinates Thompson Sampling optimization.

    Architecture:
        Decision Logger → Context Clusterer → Reward Shaper → Bandit → Policy Agent
                                ↓
                        Drift Detector → Alerts
                                ↓
                    Counterfactual Evaluator → A/B Testing
    """

    def __init__(
        self,
        vertical: str = "finance",
        n_clusters: int = 10,
        state_dir: str = "data/learning",
        log_dir: str = "data/logs/decisions",
        reports_dir: str = "data/reports"
    ):
        """
        Initialize Learning Agent.

        Args:
            vertical: Business vertical (finance, insurance, healthcare)
            n_clusters: Number of context clusters
            state_dir: Directory for saved state
            log_dir: Directory for decision logs
            reports_dir: Directory for reports and dashboards
        """
        self.vertical = vertical
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing Learning Agent components...")

        # 1. Decision Logger
        self.decision_logger = DecisionLogger(log_dir=log_dir)
        logger.info("✓ Decision Logger initialized")

        # 2. Context Clusterer
        clusterer_path = self.state_dir / f"context_clusterer_{vertical}.pkl"
        self.context_clusterer = ContextClusterer(
            n_clusters=n_clusters,
            model_path=str(clusterer_path)
        )
        logger.info("✓ Context Clusterer initialized")

        # 3. Reward Shaper (vertical-specific)
        self.reward_shaper = get_reward_shaper(vertical)
        logger.info(f"✓ Reward Shaper initialized (vertical={vertical})")

        # 4. Thompson Sampling Bandit
        bandit_path = self.state_dir / f"bandit_{vertical}.json"
        self.bandit = ThompsonSamplingBandit(
            n_contexts=n_clusters,
            context_labels=self.context_clusterer.cluster_labels,
            state_path=str(bandit_path)
        )
        logger.info("✓ Thompson Sampling Bandit initialized")

        # 5. Drift Detector
        drift_reports_dir = Path(reports_dir) / "drift"
        self.drift_detector = DriftDetector(reports_dir=str(drift_reports_dir))
        logger.info("✓ Drift Detector initialized")

        # 6. Counterfactual Evaluator
        self.counterfactual_evaluator = CounterfactualEvaluator()
        logger.info("✓ Counterfactual Evaluator initialized")

        # Metadata
        self.last_update = None
        self.update_history = []

        logger.info("🎯 Learning Agent ready")

    def log_decision(
        self,
        case_id: str,
        context: Dict,
        policy_weights: Dict,
        decision_type: str,
        utility_score: float
    ) -> str:
        """
        Log a decision made by Policy Agent.

        Args:
            case_id: Unique case identifier
            context: Context features at decision time
            policy_weights: Weights used by policy
            decision_type: auto_resolve, hitl_review, etc.
            utility_score: Computed utility

        Returns:
            log_id: Decision log ID for later outcome update
        """
        # Predict context cluster
        context_id = self.context_clusterer.predict(context)

        # Compute propensity (probability of this action)
        # For now, use fixed propensity (in production, track actual policy prob)
        propensity = 0.8

        action = {
            'policy_weights': policy_weights,
            'decision': decision_type,
            'utility_score': utility_score,
            'context_cluster': context_id
        }

        log_id = self.decision_logger.log_decision(
            case_id=case_id,
            vertical=self.vertical,
            context=context,
            action=action,
            propensity=propensity
        )

        logger.debug(f"Logged decision {log_id} for case {case_id}")

        return log_id

    def update_outcome(
        self,
        log_id: str,
        correct: bool,
        reversed: bool,
        cycle_time_hours: float,
        cost_saved: float,
        context: Optional[Dict] = None,
        action: Optional[Dict] = None,
        error_severity: Optional[str] = None
    ):
        """
        Update decision with observed outcome.

        Args:
            log_id: Decision log ID
            correct: Was the decision correct?
            reversed: Was the decision reversed?
            cycle_time_hours: Time to resolution
            cost_saved: Cost savings vs. manual
            context: Context dict (if not loading from log)
            action: Action dict (if not loading from log)
            error_severity: For incorrect decisions (low/medium/high/critical)
        """
        # Build outcome dict
        outcome = {
            'correct': correct,
            'reversed': reversed,
            'cycle_time_hours': cycle_time_hours,
            'cost_saved': cost_saved,
            'error_severity': error_severity
        }

        # If context/action not provided, load from log
        if context is None or action is None:
            log_entry = self.decision_logger.get_log_by_id(log_id)
            if not log_entry:
                logger.error(f"Log {log_id} not found, cannot update outcome")
                return
            context = log_entry['context']
            action = log_entry['action']

        # Compute reward
        reward = self.reward_shaper.compute_reward(outcome, context, action)

        # Update log
        self.decision_logger.update_outcome(log_id, outcome, reward)

        logger.debug(f"Updated outcome for {log_id}: reward={reward:.2f}")

    def run_update(
        self,
        lookback_days: int = 7,
        min_samples: int = 50,
        explore_rate: float = 0.1
    ) -> Dict:
        """
        Run nightly learning update.

        Workflow:
        1. Load decision logs with outcomes
        2. Fit context clusterer (if needed)
        3. Compute rewards for all outcomes
        4. Update Thompson Sampling bandit
        5. Check for drift
        6. Evaluate counterfactual policies
        7. Save updated state
        8. Generate report

        Args:
            lookback_days: Days of logs to process
            min_samples: Minimum samples needed for update
            explore_rate: Exploration rate for new weights

        Returns:
            Dict with update results
        """
        logger.info("=" * 60)
        logger.info("STARTING LEARNING AGENT UPDATE")
        logger.info("=" * 60)

        start_time = datetime.now()

        # 1. Load decision logs
        logger.info(f"Loading decision logs (last {lookback_days} days)...")
        logs = self.decision_logger.load_logs(
            vertical=self.vertical,
            with_outcomes_only=False
        )

        # Merge outcomes into decisions
        logs = merge_outcome_to_decision(logs)

        # Filter to logs with outcomes
        logs_with_outcomes = [log for log in logs if log.get('outcome')]

        logger.info(f"Loaded {len(logs)} total logs, {len(logs_with_outcomes)} with outcomes")

        if len(logs_with_outcomes) < min_samples:
            logger.warning(
                f"Insufficient samples ({len(logs_with_outcomes)} < {min_samples}), "
                "skipping update"
            )
            return {
                'status': 'skipped',
                'reason': 'insufficient_samples',
                'sample_count': len(logs_with_outcomes)
            }

        # 2. Fit/update context clusterer
        logger.info("Updating context clusterer...")
        contexts = [log['context'] for log in logs]

        if not self.context_clusterer.is_fitted or len(contexts) > 500:
            self.context_clusterer.fit(contexts)
            logger.info("✓ Context clusterer updated")
        else:
            logger.info("✓ Using existing context clusters")

        # Get cluster stats
        cluster_stats = self.context_clusterer.get_cluster_stats(contexts)

        # 3. Update bandit with observations
        logger.info("Updating Thompson Sampling bandit...")

        update_count = 0
        for log in logs_with_outcomes:
            context = log['context']
            action = log['action']
            reward = log.get('reward')

            if reward is None:
                continue

            # Get context cluster
            context_id = self.context_clusterer.predict(context)

            # Get action ID (map weights to action)
            action_id = self._map_weights_to_action(action.get('policy_weights', {}))

            # Update bandit
            self.bandit.update(context_id, action_id, reward)
            update_count += 1

        logger.info(f"✓ Updated bandit with {update_count} observations")

        # 4. Check for drift
        logger.info("Running drift detection...")
        drift_result = run_drift_check(logs_with_outcomes, self.drift_detector)

        # 5. Evaluate counterfactual policies
        logger.info("Evaluating counterfactual policies...")

        # Create policies to evaluate
        policies = {}
        for action_id, weights in enumerate(self.bandit.actions):
            policy_name = self._get_policy_name(weights)
            policies[policy_name] = self._create_policy_function(action_id, weights)

        # Evaluate using IPS
        policy_evaluations = self.counterfactual_evaluator.compare_policies(
            logs_with_outcomes,
            policies,
            method='ips'
        )

        # 6. Get best actions per context
        logger.info("Computing optimal policies per context...")
        best_policies = {}
        for ctx_id in range(self.context_clusterer.n_clusters):
            action_id, weights = self.bandit.get_best_action(ctx_id)
            label = self.context_clusterer.cluster_labels[ctx_id]
            best_policies[label] = {
                'action_id': action_id,
                'weights': weights,
                'mean_reward': float(self.bandit.means[ctx_id, action_id]),
                'std': float(self.bandit.stds[ctx_id, action_id])
            }

        # 7. Save state
        logger.info("Saving updated state...")
        self.bandit.save_state()
        self.context_clusterer.save_model()
        logger.info("✓ State saved")

        # 8. Generate report
        elapsed = (datetime.now() - start_time).total_seconds()

        report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'vertical': self.vertical,
            'data': {
                'total_logs': len(logs),
                'logs_with_outcomes': len(logs_with_outcomes),
                'lookback_days': lookback_days
            },
            'clusters': cluster_stats,
            'drift': drift_result,
            'policy_evaluations': {
                name: {
                    'value': eval.value_estimate,
                    'std_error': eval.std_error,
                    'samples': eval.n_samples
                }
                for name, eval in policy_evaluations.items()
            },
            'best_policies': best_policies,
            'should_retrain': drift_result['should_retrain'],
            'bandit_stats': self.bandit.get_statistics()
        }

        # Save report
        report_path = self.state_dir / f"update_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Report saved to {report_path}")

        # Track update history
        self.last_update = datetime.now()
        self.update_history.append(report)

        logger.info("=" * 60)
        logger.info("LEARNING AGENT UPDATE COMPLETE")
        logger.info(f"Status: {report['status']}")
        logger.info(f"Should retrain: {report['should_retrain']}")
        logger.info(f"Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        return report

    def get_policy_for_context(
        self,
        context: Dict,
        explore: bool = True,
        temperature: float = 1.0
    ) -> Tuple[Dict, float]:
        """
        Get policy weights for a given context.

        Args:
            context: Context features
            explore: If True, use Thompson Sampling (explore)
            temperature: Exploration temperature (higher = more random)

        Returns:
            (policy_weights, propensity)
        """
        # Predict context cluster
        context_id = self.context_clusterer.predict(context)

        # Select action using bandit
        action_id, weights = self.bandit.select_action(
            context_id,
            explore=explore,
            temperature=temperature
        )

        # Compute propensity (for logging)
        # Simplified: use deterministic probability based on mean
        if not explore:
            propensity = 1.0  # Greedy
        else:
            # Probability of selecting this action via Thompson Sampling
            # Approximation: softmax over sampled values
            propensity = 0.8  # Placeholder (can be computed more precisely)

        return weights, propensity

    def get_dashboard_metrics(self) -> Dict:
        """
        Get metrics for monitoring dashboard.

        Returns:
            Dict with key performance indicators
        """
        # Load recent logs
        recent_logs = self.decision_logger.get_recent_logs(hours=24)
        recent_with_outcomes = [log for log in recent_logs if log.get('outcome')]

        if not recent_with_outcomes:
            return {'message': 'no_recent_data'}

        # Compute metrics
        rewards = [log['reward'] for log in recent_with_outcomes if log.get('reward')]
        accuracy = sum(
            1 for log in recent_with_outcomes
            if log.get('outcome', {}).get('correct', False)
        ) / len(recent_with_outcomes)
        reversal_rate = sum(
            1 for log in recent_with_outcomes
            if log.get('outcome', {}).get('reversed', False)
        ) / len(recent_with_outcomes)

        import numpy as np

        return {
            'last_24h': {
                'total_decisions': len(recent_logs),
                'with_outcomes': len(recent_with_outcomes),
                'avg_reward': float(np.mean(rewards)) if rewards else 0,
                'accuracy': accuracy,
                'reversal_rate': reversal_rate
            },
            'bandit': self.bandit.get_statistics(),
            'drift': self.drift_detector.get_drift_summary(),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }

    def _map_weights_to_action(self, weights: Dict) -> int:
        """Map policy weights to action ID"""
        # Find closest action by L2 distance
        import numpy as np

        weight_vec = np.array([
            weights.get('alpha', 0.2),
            weights.get('beta', 0.2),
            weights.get('gamma', 0.2),
            weights.get('delta', 0.2),
            weights.get('lambda', 0.2)
        ])

        best_action = 0
        best_dist = float('inf')

        for action_id, action_weights in enumerate(self.bandit.actions):
            action_vec = np.array([
                action_weights['alpha'],
                action_weights['beta'],
                action_weights['gamma'],
                action_weights['delta'],
                action_weights['lambda']
            ])
            dist = np.linalg.norm(weight_vec - action_vec)
            if dist < best_dist:
                best_dist = dist
                best_action = action_id

        return best_action

    def _get_policy_name(self, weights: Dict) -> str:
        """Get human-readable name for policy weights"""
        alpha = weights.get('alpha', 0.2)
        gamma = weights.get('gamma', 0.2)
        delta = weights.get('delta', 0.2)
        lambda_ = weights.get('lambda', 0.2)

        if alpha > 0.35:
            return 'rules_heavy'
        elif gamma > 0.30:
            return 'ml_heavy'
        elif delta > 0.30:
            return 'genai_heavy'
        elif lambda_ > 0.25:
            return 'assurance_heavy'
        else:
            return 'balanced'

    def _create_policy_function(self, action_id: int, weights: Dict) -> callable:
        """Create policy function for counterfactual evaluation"""
        def policy(context):
            # Predict cluster
            context_id = self.context_clusterer.predict(context)

            # Return action and propensity
            action = {
                'decision': 'auto_resolve',
                'policy_weights': weights
            }

            # Propensity = probability this policy assigns to this action
            # Use softmax over Q-values
            import numpy as np
            q_values = self.bandit.means[context_id]
            exp_q = np.exp(q_values - q_values.max())
            probs = exp_q / exp_q.sum()
            propensity = float(probs[action_id])

            return action, propensity

        return policy


# CLI interface for running updates

def main():
    """Run learning agent update from command line"""
    import argparse

    parser = argparse.ArgumentParser(description='QURE Learning Agent')
    parser.add_argument('--vertical', default='finance', help='Business vertical')
    parser.add_argument('--lookback-days', type=int, default=7, help='Days of logs to process')
    parser.add_argument('--min-samples', type=int, default=50, help='Minimum samples for update')
    parser.add_argument('--dashboard', action='store_true', help='Show dashboard metrics')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Create agent
    agent = LearningAgent(vertical=args.vertical)

    if args.dashboard:
        # Show dashboard
        metrics = agent.get_dashboard_metrics()
        print("\n" + "=" * 60)
        print("LEARNING AGENT DASHBOARD")
        print("=" * 60)
        print(json.dumps(metrics, indent=2))
    else:
        # Run update
        report = agent.run_update(
            lookback_days=args.lookback_days,
            min_samples=args.min_samples
        )

        print("\n" + "=" * 60)
        print("UPDATE REPORT SUMMARY")
        print("=" * 60)
        print(f"Status: {report['status']}")
        print(f"Logs processed: {report['data']['logs_with_outcomes']}")
        print(f"Should retrain: {report['should_retrain']}")
        print(f"Elapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
