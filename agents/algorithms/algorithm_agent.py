"""
Algorithm Agent

Executes exact, auditable computations (fuzzy matching, graph traversal, temporal analysis).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from common.schemas import AlgorithmRequest, AlgorithmResponse

logger = logging.getLogger(__name__)


class AlgorithmAgent:
    """
    Algorithm Agent for deterministic computations

    Responsibilities:
    - String similarity (Jaro-Winkler, TF-IDF cosine, Levenshtein)
    - Temporal windowing (±N days, dynamic time warping)
    - Graph algorithms (shortest path, bipartite matching)
    - Financial reconciliation scoring (multi-signal fusion)
    """

    def __init__(self):
        """Initialize Algorithm Agent"""
        logger.info("✅ Algorithm Agent initialized")

    def execute(
        self,
        request: AlgorithmRequest,
    ) -> AlgorithmResponse:
        """
        Execute algorithm

        Args:
            request: AlgorithmRequest with algorithm type and inputs

        Returns:
            AlgorithmResponse with result and score
        """
        algorithm_type = request.algorithm_type
        inputs = request.inputs

        try:
            if algorithm_type == "fuzzy_match":
                result, score, explanation = self._fuzzy_match(inputs)

            elif algorithm_type == "reconciliation_score":
                result, score, explanation = self._reconciliation_score(inputs)

            elif algorithm_type == "date_proximity":
                result, score, explanation = self._date_proximity(inputs)

            elif algorithm_type == "amount_similarity":
                result, score, explanation = self._amount_similarity(inputs)

            elif algorithm_type == "temporal_window":
                result, score, explanation = self._temporal_window(inputs)

            else:
                logger.warning(f"Unknown algorithm type: {algorithm_type}")
                result = None
                score = 0.0
                explanation = f"Unknown algorithm: {algorithm_type}"

            logger.debug(f"Algorithm {algorithm_type} executed: score={score:.3f}")

            return AlgorithmResponse(
                case_id=request.case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                algorithm_type=algorithm_type,
                score=score,
                result=result,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Algorithm execution failed: {e}")
            return AlgorithmResponse(
                case_id=request.case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                algorithm_type=algorithm_type,
                score=0.0,
                result=None,
                explanation=f"Error: {str(e)}",
            )

    def _fuzzy_match(self, inputs: Dict[str, Any]) -> Tuple[Any, float, str]:
        """
        Fuzzy string matching

        Args:
            inputs: Dict with 'string1', 'string2', optional 'method'

        Returns:
            Tuple of (result, score, explanation)
        """
        string1 = str(inputs.get("string1", ""))
        string2 = str(inputs.get("string2", ""))
        method = inputs.get("method", "token_sort_ratio")

        if method == "ratio":
            score = fuzz.ratio(string1, string2) / 100.0
        elif method == "partial_ratio":
            score = fuzz.partial_ratio(string1, string2) / 100.0
        elif method == "token_sort_ratio":
            score = fuzz.token_sort_ratio(string1, string2) / 100.0
        elif method == "token_set_ratio":
            score = fuzz.token_set_ratio(string1, string2) / 100.0
        else:
            score = fuzz.ratio(string1, string2) / 100.0

        explanation = f"Fuzzy match ({method}): '{string1}' vs '{string2}' = {score:.2%}"

        return {"match": score > 0.7}, score, explanation

    def _reconciliation_score(self, inputs: Dict[str, Any]) -> Tuple[Any, float, str]:
        """
        Multi-signal reconciliation scoring for GL↔Bank matching

        Args:
            inputs: Dict with scoring components

        Returns:
            Tuple of (result, score, explanation)
        """
        # Default weights
        weights = {
            "date_proximity": 0.30,
            "amount_similarity": 0.25,
            "memo_similarity": 0.20,
            "payer_match": 0.15,
            "reference_match": 0.10,
        }

        # Get component scores
        date_score = inputs.get("date_proximity_score", 0.0)
        amount_score = inputs.get("amount_similarity_score", 0.0)
        memo_score = inputs.get("memo_similarity_score", 0.0)
        payer_score = inputs.get("payer_match_score", 0.0)
        reference_score = inputs.get("reference_match_score", 0.0)

        # Compute weighted score
        total_score = (
            weights["date_proximity"] * date_score +
            weights["amount_similarity"] * amount_score +
            weights["memo_similarity"] * memo_score +
            weights["payer_match"] * payer_score +
            weights["reference_match"] * reference_score
        )

        # Breakdown for explanation
        breakdown = {
            "date_proximity": f"{date_score:.2f} (weight: {weights['date_proximity']})",
            "amount_similarity": f"{amount_score:.2f} (weight: {weights['amount_similarity']})",
            "memo_similarity": f"{memo_score:.2f} (weight: {weights['memo_similarity']})",
            "payer_match": f"{payer_score:.2f} (weight: {weights['payer_match']})",
            "reference_match": f"{reference_score:.2f} (weight: {weights['reference_match']})",
        }

        explanation = (
            f"Reconciliation score: {total_score:.3f}\n" +
            "\n".join(f"  {k}: {v}" for k, v in breakdown.items())
        )

        result = {
            "total_score": total_score,
            "breakdown": breakdown,
            "match_recommended": total_score >= 0.65,
        }

        return result, total_score, explanation

    def _date_proximity(self, inputs: Dict[str, Any]) -> Tuple[Any, float, str]:
        """
        Calculate date proximity score

        Args:
            inputs: Dict with 'date1', 'date2', optional 'max_days'

        Returns:
            Tuple of (result, score, explanation)
        """
        date1_str = inputs.get("date1")
        date2_str = inputs.get("date2")
        max_days = inputs.get("max_days", 3)

        # Parse dates
        date1 = self._parse_date(date1_str)
        date2 = self._parse_date(date2_str)

        if date1 is None or date2 is None:
            return {"error": "Invalid dates"}, 0.0, "Date parsing failed"

        # Calculate day difference
        delta = abs((date2 - date1).days)

        # Score: 1.0 at 0 days, linear decay to 0.0 at max_days
        if delta <= max_days:
            score = 1.0 - (delta / max_days)
        else:
            score = 0.0

        explanation = f"Date proximity: {delta} days apart (max: {max_days}) = {score:.2%}"

        result = {
            "days_apart": delta,
            "within_window": delta <= max_days,
        }

        return result, score, explanation

    def _amount_similarity(self, inputs: Dict[str, Any]) -> Tuple[Any, float, str]:
        """
        Calculate amount similarity score

        Args:
            inputs: Dict with 'amount1', 'amount2', optional 'tolerance'

        Returns:
            Tuple of (result, score, explanation)
        """
        amount1 = float(inputs.get("amount1", 0.0))
        amount2 = float(inputs.get("amount2", 0.0))
        tolerance = float(inputs.get("tolerance", 0.01))

        # Calculate absolute and relative difference
        abs_diff = abs(amount1 - amount2)
        rel_diff = abs_diff / max(abs(amount1), abs(amount2), 1.0)

        # Score: 1.0 if within tolerance, decay based on relative difference
        if abs_diff <= tolerance:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (rel_diff * 10))  # 10% diff = 0 score

        explanation = (
            f"Amount similarity: ${amount1:.2f} vs ${amount2:.2f}\n"
            f"  Abs diff: ${abs_diff:.2f}, Rel diff: {rel_diff:.1%} = {score:.2%}"
        )

        result = {
            "abs_difference": abs_diff,
            "rel_difference": rel_diff,
            "within_tolerance": abs_diff <= tolerance,
        }

        return result, score, explanation

    def _temporal_window(self, inputs: Dict[str, Any]) -> Tuple[Any, float, str]:
        """
        Check if events fall within temporal window

        Args:
            inputs: Dict with 'event_dates' list and 'window_days'

        Returns:
            Tuple of (result, score, explanation)
        """
        event_dates_str = inputs.get("event_dates", [])
        window_days = inputs.get("window_days", 7)

        # Parse all dates
        event_dates = [self._parse_date(d) for d in event_dates_str]
        event_dates = [d for d in event_dates if d is not None]

        if len(event_dates) < 2:
            return {"error": "Need at least 2 dates"}, 0.0, "Insufficient dates"

        # Find min and max dates
        min_date = min(event_dates)
        max_date = max(event_dates)

        # Calculate span
        span_days = (max_date - min_date).days

        # Score: 1.0 if within window, 0.0 if beyond
        score = 1.0 if span_days <= window_days else 0.0

        explanation = (
            f"Temporal window: {len(event_dates)} events span {span_days} days "
            f"(window: {window_days}) = {'✓ PASS' if score > 0 else '✗ FAIL'}"
        )

        result = {
            "span_days": span_days,
            "within_window": span_days <= window_days,
            "event_count": len(event_dates),
        }

        return result, score, explanation

    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """
        Parse date string to datetime

        Args:
            date_str: Date string or datetime object

        Returns:
            datetime object or None if parsing fails
        """
        if isinstance(date_str, datetime):
            return date_str

        if not date_str:
            return None

        try:
            from dateutil import parser
            return parser.parse(str(date_str))
        except:
            return None


# Singleton instance
_algorithm_agent: Optional[AlgorithmAgent] = None


def get_algorithm_agent() -> AlgorithmAgent:
    """
    Get or create singleton AlgorithmAgent instance

    Returns:
        AlgorithmAgent instance
    """
    global _algorithm_agent

    if _algorithm_agent is None:
        _algorithm_agent = AlgorithmAgent()

    return _algorithm_agent
