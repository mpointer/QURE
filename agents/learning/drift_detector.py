"""
Drift Detector using Evidently AI

Monitors for:
1. Context drift: Distribution changes in input features
2. Prediction drift: Changes in action selection patterns
3. Reward drift: Performance degradation over time
4. Concept drift: Relationships between contexts and rewards changing

Triggers retraining when significant drift is detected.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
        ColumnSummaryMetric
    )
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfDriftedColumns,
        TestShareOfDriftedColumns,
        TestColumnDrift
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently AI not installed. Run: pip install evidently")

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors for drift in context distributions and model performance.

    Uses Evidently AI to detect:
    - Data drift: Input feature distributions changing
    - Target drift: Reward distributions changing
    - Prediction drift: Model behavior changing
    """

    def __init__(
        self,
        reference_window_days: int = 7,
        current_window_days: int = 1,
        drift_threshold: float = 0.5,
        reports_dir: str = "data/drift_reports"
    ):
        """
        Initialize drift detector.

        Args:
            reference_window_days: Days of data to use as reference baseline
            current_window_days: Days of recent data to compare
            drift_threshold: Threshold for drift detection (0-1)
            reports_dir: Directory to save drift reports
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError(
                "Evidently AI is required for drift detection. "
                "Install with: pip install evidently"
            )

        self.reference_window_days = reference_window_days
        self.current_window_days = current_window_days
        self.drift_threshold = drift_threshold
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Track drift history
        self.drift_history = []

    def prepare_dataframe(self, decision_logs: List[Dict]) -> pd.DataFrame:
        """
        Convert decision logs to DataFrame for Evidently.

        Args:
            decision_logs: List of decision log entries

        Returns:
            DataFrame with flattened features
        """
        rows = []

        for log in decision_logs:
            # Skip logs without outcomes
            if not log.get('outcome'):
                continue

            context = log.get('context', {})
            action = log.get('action', {})
            outcome = log.get('outcome', {})

            row = {
                # Metadata
                'timestamp': log.get('timestamp'),
                'case_id': log.get('case_id'),
                'vertical': log.get('vertical'),

                # Context features
                'transaction_amount': context.get('transaction_amount', 0),
                'data_quality_score': context.get('data_quality_score', 0.5),
                'urgency': context.get('urgency', 'medium'),
                'has_swift_reference': int(context.get('has_swift_reference', False)),
                'rule_pass_count': context.get('rule_pass_count', 0),
                'rule_fail_count': context.get('rule_fail_count', 0),
                'ml_confidence': context.get('ml_confidence', 0.5),
                'genai_confidence': context.get('genai_confidence', 0.5),
                'assurance_score': context.get('assurance_score', 0.5),

                # Action (decision made)
                'decision': action.get('decision', 'unknown'),
                'utility_score': action.get('utility_score', 0),
                'propensity': log.get('propensity', 0),

                # Outcome (target)
                'correct': int(outcome.get('correct', False)),
                'reversed': int(outcome.get('reversed', False)),
                'cycle_time_hours': outcome.get('cycle_time_hours', 48),
                'cost_saved': outcome.get('cost_saved', 0),

                # Reward
                'reward': log.get('reward', 0)
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def split_reference_current(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into reference and current windows.

        Args:
            df: Full dataset with timestamps

        Returns:
            (reference_df, current_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Calculate cutoff dates
        max_date = df['timestamp'].max()
        current_cutoff = max_date - timedelta(days=self.current_window_days)
        reference_cutoff = current_cutoff - timedelta(days=self.reference_window_days)

        # Split
        reference_df = df[
            (df['timestamp'] >= reference_cutoff) &
            (df['timestamp'] < current_cutoff)
        ].copy()

        current_df = df[df['timestamp'] >= current_cutoff].copy()

        logger.info(
            f"Split data: {len(reference_df)} reference samples, "
            f"{len(current_df)} current samples"
        )

        return reference_df, current_df

    def detect_drift(
        self,
        decision_logs: List[Dict],
        vertical: Optional[str] = None
    ) -> Dict:
        """
        Detect drift in decision logs.

        Args:
            decision_logs: List of decision log entries
            vertical: Optional vertical filter

        Returns:
            Dict with drift detection results
        """
        # Filter by vertical if specified
        if vertical:
            decision_logs = [
                log for log in decision_logs
                if log.get('vertical') == vertical
            ]

        # Prepare data
        df = self.prepare_dataframe(decision_logs)

        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for drift detection")
            return {
                'drift_detected': False,
                'reason': 'insufficient_data',
                'sample_count': len(df)
            }

        # Split into reference and current
        reference_df, current_df = self.split_reference_current(df)

        if reference_df.empty or current_df.empty:
            logger.warning("Empty reference or current window")
            return {
                'drift_detected': False,
                'reason': 'empty_windows',
                'reference_count': len(reference_df),
                'current_count': len(current_df)
            }

        # Define column mapping
        column_mapping = ColumnMapping()
        column_mapping.target = 'reward'
        column_mapping.prediction = 'utility_score'
        column_mapping.numerical_features = [
            'transaction_amount',
            'data_quality_score',
            'rule_pass_count',
            'rule_fail_count',
            'ml_confidence',
            'genai_confidence',
            'assurance_score',
            'cycle_time_hours',
            'cost_saved'
        ]
        column_mapping.categorical_features = [
            'vertical',
            'urgency',
            'decision'
        ]

        # Create drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            TargetDriftPreset(),
        ])

        # Run report
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )

        # Extract results
        report_dict = report.as_dict()

        # Parse drift detection
        dataset_drift = report_dict['metrics'][0]['result']
        drift_detected = dataset_drift.get('dataset_drift', False)
        drift_share = dataset_drift.get('share_of_drifted_columns', 0)
        drifted_columns = dataset_drift.get('number_of_drifted_columns', 0)

        # Get per-column drift
        column_drifts = {}
        for metric in report_dict['metrics']:
            if metric['metric'] == 'ColumnDriftMetric':
                col_name = metric['result']['column_name']
                drift_score = metric['result'].get('drift_score', 0)
                column_drifts[col_name] = drift_score

        result = {
            'drift_detected': drift_detected,
            'drift_share': drift_share,
            'drifted_columns': drifted_columns,
            'column_drifts': column_drifts,
            'reference_count': len(reference_df),
            'current_count': len(current_df),
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        report_path = self.reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(str(report_path))
        result['report_path'] = str(report_path)

        # Log results
        if drift_detected:
            logger.warning(
                f"⚠️ DRIFT DETECTED: {drift_share:.1%} of columns drifted "
                f"({drifted_columns} columns)"
            )
            logger.warning(f"Drifted columns: {list(column_drifts.keys())}")
        else:
            logger.info("✓ No significant drift detected")

        # Track history
        self.drift_history.append(result)

        return result

    def monitor_reward_performance(
        self,
        decision_logs: List[Dict],
        window_days: int = 7
    ) -> Dict:
        """
        Monitor reward trends over time.

        Args:
            decision_logs: List of decision log entries
            window_days: Days to analyze

        Returns:
            Dict with performance metrics
        """
        df = self.prepare_dataframe(decision_logs)

        if df.empty:
            return {'error': 'no_data'}

        # Filter to window
        cutoff = datetime.now() - timedelta(days=window_days)
        df = df[df['timestamp'] >= cutoff]

        # Compute metrics by day
        df['date'] = df['timestamp'].dt.date
        daily_metrics = df.groupby('date').agg({
            'reward': ['mean', 'std', 'count'],
            'correct': 'mean',
            'reversed': 'mean',
            'cycle_time_hours': 'mean'
        }).reset_index()

        # Trend analysis: is performance declining?
        rewards = daily_metrics[('reward', 'mean')].values
        if len(rewards) >= 3:
            # Simple linear regression
            x = np.arange(len(rewards))
            slope = np.polyfit(x, rewards, 1)[0]
            trend = 'declining' if slope < -0.5 else 'stable' if abs(slope) < 0.5 else 'improving'
        else:
            slope = 0
            trend = 'insufficient_data'

        result = {
            'window_days': window_days,
            'total_decisions': len(df),
            'avg_reward': float(df['reward'].mean()),
            'reward_std': float(df['reward'].std()),
            'accuracy': float(df['correct'].mean()),
            'reversal_rate': float(df['reversed'].mean()),
            'avg_cycle_time': float(df['cycle_time_hours'].mean()),
            'trend': trend,
            'trend_slope': float(slope),
            'daily_metrics': daily_metrics.to_dict(orient='records')
        }

        return result

    def should_retrain(
        self,
        drift_result: Dict,
        performance_result: Dict
    ) -> Tuple[bool, str]:
        """
        Decide if retraining is needed.

        Args:
            drift_result: Output from detect_drift()
            performance_result: Output from monitor_reward_performance()

        Returns:
            (should_retrain, reason)
        """
        reasons = []

        # Check for significant drift
        if drift_result.get('drift_detected'):
            drift_share = drift_result.get('drift_share', 0)
            if drift_share > self.drift_threshold:
                reasons.append(
                    f"significant_drift_{drift_share:.1%}"
                )

        # Check for performance degradation
        trend = performance_result.get('trend')
        if trend == 'declining':
            slope = performance_result.get('trend_slope', 0)
            reasons.append(f"declining_performance_slope={slope:.2f}")

        # Check for low accuracy
        accuracy = performance_result.get('accuracy', 1.0)
        if accuracy < 0.7:
            reasons.append(f"low_accuracy_{accuracy:.1%}")

        # Check for high reversal rate
        reversal_rate = performance_result.get('reversal_rate', 0)
        if reversal_rate > 0.1:
            reasons.append(f"high_reversals_{reversal_rate:.1%}")

        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "no_issues"

        if should_retrain:
            logger.warning(f"🔄 RETRAINING RECOMMENDED: {reason}")

        return should_retrain, reason

    def get_drift_summary(self) -> Dict:
        """
        Get summary of recent drift detections.

        Returns:
            Dict with drift history summary
        """
        if not self.drift_history:
            return {'message': 'no_drift_checks_performed'}

        recent = self.drift_history[-10:]  # Last 10 checks

        drift_count = sum(1 for r in recent if r.get('drift_detected'))

        return {
            'total_checks': len(self.drift_history),
            'recent_checks': len(recent),
            'recent_drift_count': drift_count,
            'recent_drift_rate': drift_count / len(recent) if recent else 0,
            'latest_check': recent[-1] if recent else None
        }


# Utility functions

def run_drift_check(
    decision_logs: List[Dict],
    detector: Optional[DriftDetector] = None
) -> Dict:
    """
    Convenience function to run full drift check.

    Args:
        decision_logs: List of decision log entries
        detector: Optional DriftDetector instance

    Returns:
        Dict with drift and performance results
    """
    if detector is None:
        detector = DriftDetector()

    # Detect drift
    drift_result = detector.detect_drift(decision_logs)

    # Monitor performance
    performance_result = detector.monitor_reward_performance(decision_logs)

    # Check if retraining needed
    should_retrain, reason = detector.should_retrain(drift_result, performance_result)

    return {
        'drift': drift_result,
        'performance': performance_result,
        'should_retrain': should_retrain,
        'retrain_reason': reason,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test drift detection
    logging.basicConfig(level=logging.INFO)

    print("=== Drift Detector Test ===\n")

    if not EVIDENTLY_AVAILABLE:
        print("ERROR: Evidently AI not installed")
        print("Install with: pip install evidently")
        exit(1)

    # Generate synthetic decision logs
    np.random.seed(42)

    # Reference period: stable performance
    reference_logs = []
    for i in range(100):
        timestamp = datetime.now() - timedelta(days=10, hours=i)

        context = {
            'transaction_amount': np.random.uniform(10_000, 200_000),
            'data_quality_score': np.random.uniform(0.7, 0.95),
            'urgency': np.random.choice(['low', 'medium', 'high']),
            'has_swift_reference': np.random.choice([True, False]),
            'rule_pass_count': np.random.randint(5, 12),
            'rule_fail_count': np.random.randint(0, 2),
            'ml_confidence': np.random.uniform(0.7, 0.95),
            'genai_confidence': np.random.uniform(0.7, 0.95),
            'assurance_score': np.random.uniform(0.7, 0.95)
        }

        action = {
            'decision': 'auto_resolve',
            'utility_score': 0.85
        }

        outcome = {
            'correct': np.random.choice([True, False], p=[0.9, 0.1]),
            'reversed': False,
            'cycle_time_hours': np.random.uniform(1, 5),
            'cost_saved': 50
        }

        reward = 30 if outcome['correct'] else -10

        reference_logs.append({
            'timestamp': timestamp.isoformat(),
            'case_id': f'TEST-{i:04d}',
            'vertical': 'finance',
            'context': context,
            'action': action,
            'propensity': 0.9,
            'outcome': outcome,
            'reward': reward
        })

    # Current period: DRIFTED (lower quality, worse performance)
    current_logs = []
    for i in range(50):
        timestamp = datetime.now() - timedelta(hours=i)

        context = {
            'transaction_amount': np.random.uniform(50_000, 500_000),  # DRIFT: Higher amounts
            'data_quality_score': np.random.uniform(0.4, 0.7),  # DRIFT: Lower quality
            'urgency': np.random.choice(['high', 'critical']),  # DRIFT: More urgent
            'has_swift_reference': np.random.choice([True, False]),
            'rule_pass_count': np.random.randint(3, 8),  # DRIFT: Fewer passes
            'rule_fail_count': np.random.randint(1, 5),  # DRIFT: More failures
            'ml_confidence': np.random.uniform(0.5, 0.8),
            'genai_confidence': np.random.uniform(0.5, 0.8),
            'assurance_score': np.random.uniform(0.5, 0.8)
        }

        action = {
            'decision': 'auto_resolve',
            'utility_score': 0.75
        }

        outcome = {
            'correct': np.random.choice([True, False], p=[0.7, 0.3]),  # DRIFT: Worse accuracy
            'reversed': np.random.choice([True, False], p=[0.85, 0.15]),
            'cycle_time_hours': np.random.uniform(2, 10),
            'cost_saved': 30
        }

        reward = 20 if outcome['correct'] else -15  # DRIFT: Lower rewards

        current_logs.append({
            'timestamp': timestamp.isoformat(),
            'case_id': f'TEST-{i+1000:04d}',
            'vertical': 'finance',
            'context': context,
            'action': action,
            'propensity': 0.85,
            'outcome': outcome,
            'reward': reward
        })

    # Combine logs
    all_logs = reference_logs + current_logs

    print(f"Generated {len(all_logs)} synthetic decision logs")
    print(f"  Reference period: {len(reference_logs)} logs (10 days ago)")
    print(f"  Current period: {len(current_logs)} logs (recent)\\n")

    # Run drift check
    detector = DriftDetector(
        reference_window_days=7,
        current_window_days=1,
        drift_threshold=0.3
    )

    result = run_drift_check(all_logs, detector)

    # Display results
    print("\\n=== Drift Detection Results ===")
    drift = result['drift']
    print(f"Drift detected: {drift['drift_detected']}")
    print(f"Drifted columns: {drift['drifted_columns']}")
    print(f"Drift share: {drift['drift_share']:.1%}")
    print(f"Report saved: {drift.get('report_path')}")

    print("\\n=== Performance Monitoring ===")
    perf = result['performance']
    print(f"Average reward: {perf['avg_reward']:.2f}")
    print(f"Accuracy: {perf['accuracy']:.1%}")
    print(f"Reversal rate: {perf['reversal_rate']:.1%}")
    print(f"Trend: {perf['trend']} (slope={perf['trend_slope']:.3f})")

    print("\\n=== Retraining Decision ===")
    print(f"Should retrain: {result['should_retrain']}")
    print(f"Reason: {result['retrain_reason']}")

    print("\\n✓ Drift detection test complete")
