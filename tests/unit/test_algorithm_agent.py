"""
Unit tests for Algorithm Agent
"""

import pytest
from datetime import datetime

from agents.algorithms import get_algorithm_agent
from common.schemas import AlgorithmRequest


class TestAlgorithmAgent:
    """Test suite for Algorithm Agent"""

    def setup_method(self):
        """Setup test fixtures"""
        self.agent = get_algorithm_agent()

    def test_fuzzy_match_exact(self):
        """Test fuzzy matching with exact strings"""
        request = AlgorithmRequest(
            case_id="TEST_001",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="fuzzy_match",
            inputs={
                "string1": "Acme Corp",
                "string2": "Acme Corp",
                "method": "ratio",
            },
        )

        response = self.agent.execute(request)

        assert response.score == 1.0
        assert response.result["match"] is True

    def test_fuzzy_match_similar(self):
        """Test fuzzy matching with similar strings"""
        request = AlgorithmRequest(
            case_id="TEST_002",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="fuzzy_match",
            inputs={
                "string1": "Acme Corp",
                "string2": "ACME CORPORATION",
                "method": "token_sort_ratio",
            },
        )

        response = self.agent.execute(request)

        assert response.score > 0.7  # Should be high similarity
        assert response.result["match"] is True

    def test_fuzzy_match_different(self):
        """Test fuzzy matching with different strings"""
        request = AlgorithmRequest(
            case_id="TEST_003",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="fuzzy_match",
            inputs={
                "string1": "Acme Corp",
                "string2": "XYZ Industries",
                "method": "ratio",
            },
        )

        response = self.agent.execute(request)

        assert response.score < 0.5  # Should be low similarity
        assert response.result["match"] is False

    def test_date_proximity_same_day(self):
        """Test date proximity with same day"""
        request = AlgorithmRequest(
            case_id="TEST_004",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="date_proximity",
            inputs={
                "date1": "2024-01-15",
                "date2": "2024-01-15",
                "max_days": 3,
            },
        )

        response = self.agent.execute(request)

        assert response.score == 1.0
        assert response.result["days_apart"] == 0
        assert response.result["within_window"] is True

    def test_date_proximity_within_window(self):
        """Test date proximity within window"""
        request = AlgorithmRequest(
            case_id="TEST_005",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="date_proximity",
            inputs={
                "date1": "2024-01-15",
                "date2": "2024-01-17",
                "max_days": 3,
            },
        )

        response = self.agent.execute(request)

        assert 0 < response.score < 1.0  # Partial score
        assert response.result["days_apart"] == 2
        assert response.result["within_window"] is True

    def test_date_proximity_outside_window(self):
        """Test date proximity outside window"""
        request = AlgorithmRequest(
            case_id="TEST_006",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="date_proximity",
            inputs={
                "date1": "2024-01-15",
                "date2": "2024-01-20",
                "max_days": 3,
            },
        )

        response = self.agent.execute(request)

        assert response.score == 0.0
        assert response.result["days_apart"] == 5
        assert response.result["within_window"] is False

    def test_amount_similarity_exact(self):
        """Test amount similarity with exact match"""
        request = AlgorithmRequest(
            case_id="TEST_007",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="amount_similarity",
            inputs={
                "amount1": 1250.00,
                "amount2": 1250.00,
                "tolerance": 0.01,
            },
        )

        response = self.agent.execute(request)

        assert response.score == 1.0
        assert response.result["within_tolerance"] is True

    def test_amount_similarity_within_tolerance(self):
        """Test amount similarity within tolerance"""
        request = AlgorithmRequest(
            case_id="TEST_008",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="amount_similarity",
            inputs={
                "amount1": 1250.00,
                "amount2": 1250.01,
                "tolerance": 0.01,
            },
        )

        response = self.agent.execute(request)

        assert response.score == 1.0
        assert response.result["within_tolerance"] is True

    def test_amount_similarity_outside_tolerance(self):
        """Test amount similarity outside tolerance"""
        request = AlgorithmRequest(
            case_id="TEST_009",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="amount_similarity",
            inputs={
                "amount1": 1250.00,
                "amount2": 1300.00,
                "tolerance": 0.01,
            },
        )

        response = self.agent.execute(request)

        assert response.score < 1.0
        assert response.result["within_tolerance"] is False

    def test_reconciliation_score(self):
        """Test multi-signal reconciliation scoring"""
        request = AlgorithmRequest(
            case_id="TEST_010",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="reconciliation_score",
            inputs={
                "date_proximity_score": 1.0,
                "amount_similarity_score": 1.0,
                "memo_similarity_score": 0.85,
                "payer_match_score": 0.95,
                "reference_match_score": 1.0,
            },
        )

        response = self.agent.execute(request)

        assert 0.8 < response.score <= 1.0  # High overall score
        assert response.result["match_recommended"] is True
        assert "breakdown" in response.result

    def test_reconciliation_score_low(self):
        """Test reconciliation scoring with low signals"""
        request = AlgorithmRequest(
            case_id="TEST_011",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="reconciliation_score",
            inputs={
                "date_proximity_score": 0.0,
                "amount_similarity_score": 0.5,
                "memo_similarity_score": 0.3,
                "payer_match_score": 0.2,
                "reference_match_score": 0.0,
            },
        )

        response = self.agent.execute(request)

        assert response.score < 0.65  # Below recommendation threshold
        assert response.result["match_recommended"] is False

    def test_temporal_window_within(self):
        """Test temporal window with events within window"""
        request = AlgorithmRequest(
            case_id="TEST_012",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="temporal_window",
            inputs={
                "event_dates": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "window_days": 7,
            },
        )

        response = self.agent.execute(request)

        assert response.score == 1.0
        assert response.result["within_window"] is True
        assert response.result["span_days"] == 2

    def test_temporal_window_outside(self):
        """Test temporal window with events outside window"""
        request = AlgorithmRequest(
            case_id="TEST_013",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="temporal_window",
            inputs={
                "event_dates": ["2024-01-15", "2024-01-25"],
                "window_days": 7,
            },
        )

        response = self.agent.execute(request)

        assert response.score == 0.0
        assert response.result["within_window"] is False
        assert response.result["span_days"] == 10

    def test_unknown_algorithm_type(self):
        """Test handling of unknown algorithm type"""
        request = AlgorithmRequest(
            case_id="TEST_014",
            from_agent="test",
            to_agent="algorithm_agent",
            algorithm_type="unknown_algorithm",
            inputs={},
        )

        response = self.agent.execute(request)

        assert response.score == 0.0
        assert response.result is None
        assert "Unknown algorithm" in response.explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
