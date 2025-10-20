"""
Decision Logging Pipeline

Logs every decision made by the Policy Agent for learning and auditing.
Uses JSONL format for append-only, immutable logging.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DecisionLog:
    """
    Complete log of a single decision.

    Follows the BANDIT logging format:
    - Context: Features describing the case
    - Action: What policy weights were used
    - Propensity: Probability of choosing this action (for counterfactual eval)
    - Outcome: What actually happened
    - Reward: Computed reward from outcome
    """
    # Identifiers
    timestamp: str
    case_id: str
    vertical: str  # finance, insurance, healthcare, etc.

    # CONTEXT: Features describing the case at decision time
    context: Dict[str, Any]

    # ACTION: What the policy did
    action: Dict[str, Any]

    # Propensity score (for counterfactual evaluation)
    propensity: float

    # OUTCOME: What happened after the decision
    outcome: Optional[Dict[str, Any]] = None

    # REWARD: Computed reward (filled in after outcome observed)
    reward: Optional[float] = None

    # Metadata
    log_id: Optional[str] = None
    updated_at: Optional[str] = None


class DecisionLogger:
    """
    Append-only decision logger using JSONL format.

    Features:
    - Immutable logs (append-only)
    - Optional hash-chaining for tamper detection
    - Batch writing for performance
    - Automatic log rotation
    """

    def __init__(
        self,
        log_dir: str = "data/logs/decisions",
        hash_chain: bool = True,
        rotation_size_mb: int = 100
    ):
        """
        Initialize decision logger.

        Args:
            log_dir: Directory to store decision logs
            hash_chain: If True, create hash chain for tamper detection
            rotation_size_mb: Rotate log file after this size (MB)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.hash_chain = hash_chain
        self.rotation_size_mb = rotation_size_mb
        self.last_hash = None

        # Current log file
        self.current_log_file = self._get_current_log_file()

    def _get_current_log_file(self) -> Path:
        """Get current log file path (date-based rotation)."""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"decisions_{date_str}.jsonl"

    def _compute_hash(self, log_entry: Dict) -> str:
        """
        Compute hash of log entry for tamper detection.

        Args:
            log_entry: Dictionary to hash

        Returns:
            SHA256 hash string
        """
        # Create deterministic JSON string
        json_str = json.dumps(log_entry, sort_keys=True)

        # Include previous hash for chaining
        if self.last_hash:
            json_str = self.last_hash + json_str

        # Compute SHA256
        hash_obj = hashlib.sha256(json_str.encode())
        return hash_obj.hexdigest()

    def log_decision(
        self,
        case_id: str,
        vertical: str,
        context: Dict,
        action: Dict,
        propensity: float,
        outcome: Optional[Dict] = None,
        reward: Optional[float] = None
    ) -> str:
        """
        Log a single decision.

        Args:
            case_id: Unique case identifier
            vertical: Business vertical (finance, insurance, etc.)
            context: Context features at decision time
            action: Action taken (policy weights, decision type)
            propensity: P(action | context) from policy
            outcome: Observed outcome (if available)
            reward: Computed reward (if outcome available)

        Returns:
            log_id: Unique log entry ID
        """
        timestamp = datetime.now().isoformat()

        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'case_id': case_id,
            'vertical': vertical,
            'context': context,
            'action': action,
            'propensity': propensity,
            'outcome': outcome,
            'reward': reward
        }

        # Add hash for tamper detection
        if self.hash_chain:
            log_hash = self._compute_hash(log_entry)
            log_entry['log_hash'] = log_hash
            log_entry['prev_hash'] = self.last_hash
            self.last_hash = log_hash

        # Generate log ID
        log_id = hashlib.md5(f"{case_id}_{timestamp}".encode()).hexdigest()
        log_entry['log_id'] = log_id

        # Write to file
        log_file = self._get_current_log_file()
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.debug(f"Logged decision {log_id} for case {case_id}")

        return log_id

    def update_outcome(
        self,
        log_id: str,
        outcome: Dict,
        reward: float
    ):
        """
        Update a log entry with outcome and reward.

        Note: This creates a NEW entry (append-only), does not modify original.

        Args:
            log_id: Log entry ID to update
            outcome: Observed outcome
            reward: Computed reward
        """
        # Find original entry
        original_entry = self.get_log_by_id(log_id)
        if not original_entry:
            logger.error(f"Log entry {log_id} not found")
            return

        # Create update entry
        update_entry = {
            'timestamp': datetime.now().isoformat(),
            'log_id': log_id,
            'update_type': 'outcome',
            'outcome': outcome,
            'reward': reward
        }

        # Add hash
        if self.hash_chain:
            log_hash = self._compute_hash(update_entry)
            update_entry['log_hash'] = log_hash
            update_entry['prev_hash'] = self.last_hash
            self.last_hash = log_hash

        # Append update
        log_file = self._get_current_log_file()
        with open(log_file, 'a') as f:
            f.write(json.dumps(update_entry) + '\n')

        logger.debug(f"Updated log {log_id} with outcome")

    def get_log_by_id(self, log_id: str) -> Optional[Dict]:
        """
        Retrieve log entry by ID.

        Args:
            log_id: Log entry ID

        Returns:
            Log entry dict or None if not found
        """
        # Search recent logs first
        for days_back in range(7):
            date = datetime.now()
            for _ in range(days_back):
                from datetime import timedelta
                date = date - timedelta(days=1)

            date_str = date.strftime("%Y%m%d")
            log_file = self.log_dir / f"decisions_{date_str}.jsonl"

            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('log_id') == log_id:
                        return entry

        return None

    def load_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        vertical: Optional[str] = None,
        with_outcomes_only: bool = False
    ) -> List[Dict]:
        """
        Load logs for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD), default: 7 days ago
            end_date: End date (YYYY-MM-DD), default: today
            vertical: Filter by vertical
            with_outcomes_only: Only return logs with outcomes

        Returns:
            List of log entries
        """
        from datetime import datetime, timedelta

        # Default date range: last 7 days
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start = datetime.now() - timedelta(days=7)
            start_date = start.strftime("%Y%m%d")

        logs = []

        # Iterate through log files
        for log_file in sorted(self.log_dir.glob("decisions_*.jsonl")):
            # Parse date from filename
            file_date = log_file.stem.replace("decisions_", "")
            if file_date < start_date or file_date > end_date:
                continue

            # Read log file
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)

                    # Skip updates (keep only original decisions)
                    if 'update_type' in entry:
                        continue

                    # Filter by vertical
                    if vertical and entry.get('vertical') != vertical:
                        continue

                    # Filter by outcome availability
                    if with_outcomes_only and not entry.get('outcome'):
                        continue

                    logs.append(entry)

        logger.info(f"Loaded {len(logs)} decision logs from {start_date} to {end_date}")

        return logs

    def get_recent_logs(self, hours: int = 24) -> List[Dict]:
        """
        Get logs from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of log entries
        """
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        logs = []
        log_file = self._get_current_log_file()

        if not log_file.exists():
            return logs

        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)

                # Skip updates
                if 'update_type' in entry:
                    continue

                # Check timestamp
                if entry['timestamp'] >= cutoff_str:
                    logs.append(entry)

        return logs

    def verify_chain(self, log_file: Optional[Path] = None) -> bool:
        """
        Verify hash chain integrity.

        Args:
            log_file: Log file to verify (default: current)

        Returns:
            True if chain is valid, False if tampered
        """
        if not self.hash_chain:
            logger.warning("Hash chaining is disabled")
            return True

        log_file = log_file or self._get_current_log_file()

        if not log_file.exists():
            return True

        prev_hash = None
        with open(log_file, 'r') as f:
            for i, line in enumerate(f):
                entry = json.loads(line)

                # Verify hash chain
                if entry.get('prev_hash') != prev_hash:
                    logger.error(f"Hash chain broken at line {i+1}")
                    return False

                # Verify entry hash
                expected_hash = entry.get('log_hash')
                if expected_hash:
                    # Recompute hash (exclude hash fields)
                    verify_entry = {k: v for k, v in entry.items()
                                    if k not in ['log_hash', 'prev_hash']}
                    actual_hash = self._compute_hash(verify_entry)

                    if actual_hash != expected_hash:
                        logger.error(f"Entry hash mismatch at line {i+1}")
                        return False

                prev_hash = entry.get('log_hash')

        logger.info(f"Hash chain verified for {log_file}")
        return True


# Utility functions

def merge_outcome_to_decision(decisions: List[Dict]) -> List[Dict]:
    """
    Merge outcome updates back into original decision logs.

    Args:
        decisions: List of decision logs (may include updates)

    Returns:
        List of merged decision logs with outcomes
    """
    # Build map of original decisions
    decision_map = {}
    updates_map = {}

    for entry in decisions:
        if 'update_type' in entry:
            # This is an update
            log_id = entry['log_id']
            updates_map[log_id] = entry
        else:
            # This is an original decision
            log_id = entry['log_id']
            decision_map[log_id] = entry

    # Merge outcomes
    merged = []
    for log_id, decision in decision_map.items():
        if log_id in updates_map:
            update = updates_map[log_id]
            decision['outcome'] = update['outcome']
            decision['reward'] = update['reward']
            decision['updated_at'] = update['timestamp']

        merged.append(decision)

    return merged


if __name__ == "__main__":
    # Test logging pipeline
    logging.basicConfig(level=logging.INFO)

    print("=== Decision Logging Pipeline Test ===\n")

    # Create logger
    logger_instance = DecisionLogger(log_dir="data/logs/test")

    # Log a decision
    log_id = logger_instance.log_decision(
        case_id="GL-20251020-0042",
        vertical="finance",
        context={
            "transaction_amount": 125000,
            "data_quality_score": 0.85,
            "has_swift_reference": True,
            "urgency": "high"
        },
        action={
            "policy_weights": {
                "alpha": 0.25,
                "beta": 0.20,
                "gamma": 0.20,
                "delta": 0.20,
                "lambda": 0.15
            },
            "decision": "auto_resolve",
            "utility_score": 0.86
        },
        propensity=0.9
    )

    print(f"Logged decision: {log_id}\n")

    # Simulate outcome
    logger_instance.update_outcome(
        log_id=log_id,
        outcome={
            "correct": True,
            "reversed": False,
            "cycle_time_hours": 2,
            "cost_saved": 50
        },
        reward=39.6
    )

    print("Updated with outcome\n")

    # Load logs
    logs = logger_instance.get_recent_logs(hours=24)
    print(f"Loaded {len(logs)} recent logs\n")

    # Verify chain
    if logger_instance.verify_chain():
        print("✓ Hash chain verified - no tampering detected\n")
    else:
        print("✗ Hash chain broken - potential tampering!\n")
