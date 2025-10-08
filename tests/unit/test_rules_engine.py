"""
Unit tests for Rules Engine Agent
"""

import pytest

from agents.rules import get_rules_engine
from common.schemas import RulesEvaluationRequest


class TestRulesEngine:
    """Test suite for Rules Engine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = get_rules_engine()

    def test_finance_rules_all_pass(self):
        """Test finance rules with all passing"""
        request = RulesEvaluationRequest(
            case_id="TEST_001",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 1250.00,
                "bank_amount": 1250.00,
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        assert len(response.failed_rules) == 0
        assert len(response.passed_rules) > 0
        assert response.rule_score > 0.8

    def test_amount_mismatch(self):
        """Test rules with amount mismatch"""
        request = RulesEvaluationRequest(
            case_id="TEST_002",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 1250.00,
                "bank_amount": 1300.00,  # Mismatch
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        assert "FR_R1_amount_match" in response.failed_rules
        assert response.rule_score == 0.0  # Mandatory failure

    def test_currency_mismatch(self):
        """Test rules with currency mismatch"""
        request = RulesEvaluationRequest(
            case_id="TEST_003",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 1250.00,
                "bank_amount": 1250.00,
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "EUR",  # Mismatch
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        assert "FR_R7_currency_match" in response.failed_rules
        assert response.rule_score == 0.0  # Mandatory failure

    def test_high_value_no_swift(self):
        """Test SOX compliance rule for high-value transactions"""
        request = RulesEvaluationRequest(
            case_id="TEST_004",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 15000.00,  # > $10k
                "bank_amount": 15000.00,
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": None,  # Missing SWIFT
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        assert "FR_R3_high_value_requires_swift" in response.failed_rules
        assert response.rule_score == 0.0  # Mandatory SOX failure

    def test_duplicate_detection(self):
        """Test duplicate detection rule"""
        request = RulesEvaluationRequest(
            case_id="TEST_005",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 1250.00,
                "bank_amount": 1250.00,
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": True,  # Duplicate
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        assert "FR_R5_duplicate_detection" in response.failed_rules
        assert response.rule_score == 0.0  # Mandatory failure

    def test_optional_rule_failure(self):
        """Test optional rule failure (date proximity)"""
        request = RulesEvaluationRequest(
            case_id="TEST_006",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 1250.00,
                "bank_amount": 1250.00,
                "date_diff_days": 5,  # > 3 days (optional rule)
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        # Optional rule failure should not cause hard fail
        assert response.rule_score > 0.0
        assert "FR_R2_date_proximity" in response.passed_rules  # Rule passes (condition not met but action is "pass")

    def test_manager_approval_required(self):
        """Test manager approval rule for high-value transactions"""
        request = RulesEvaluationRequest(
            case_id="TEST_007",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 60000.00,  # > $50k
                "bank_amount": 60000.00,
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "unreconciled",
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        # Should need evidence (manager approval)
        assert len(response.needs_evidence) > 0
        assert response.rule_score == 0.5  # Partial score when evidence needed

    def test_invalid_status(self):
        """Test invalid GL status"""
        request = RulesEvaluationRequest(
            case_id="TEST_008",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="finance_reconciliation",
            case_data={
                "gl_amount": 1250.00,
                "bank_amount": 1250.00,
                "date_diff_days": 1,
                "gl_payer": "acme corp",
                "bank_payer": "acme corp",
                "swift_ref": "SWIFT123456",
                "gl_currency": "USD",
                "bank_currency": "USD",
                "gl_status": "reconciled",  # Invalid status
                "memo_similarity": 0.85,
                "is_duplicate": False,
                "business_days_elapsed": 1,
            },
        )

        response = self.engine.evaluate(request)

        assert "FR_R8_status_valid" in response.failed_rules
        assert response.rule_score == 0.0  # Mandatory failure

    def test_unknown_rule_set(self):
        """Test handling of unknown rule set"""
        request = RulesEvaluationRequest(
            case_id="TEST_009",
            from_agent="test",
            to_agent="rules_engine",
            rule_set="unknown_rule_set",
            case_data={},
        )

        response = self.engine.evaluate(request)

        assert response.rule_score == 0.0
        assert len(response.passed_rules) == 0
        assert len(response.failed_rules) == 0
        assert "not found" in response.explanations[0]

    def test_get_available_rule_sets(self):
        """Test getting available rule sets"""
        rule_sets = self.engine.get_available_rule_sets()

        assert "finance_reconciliation" in rule_sets

    def test_reload_rules(self):
        """Test rules reload functionality"""
        # Should not raise exception
        self.engine.reload_rules("finance_reconciliation")
        self.engine.reload_rules()  # Reload all


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
