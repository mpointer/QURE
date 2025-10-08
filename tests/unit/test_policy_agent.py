"""
Unit tests for Policy Agent
"""

import pytest

from agents.policy import get_policy_agent, DecisionType


class TestPolicyAgent:
    """Test suite for Policy Agent"""

    def setup_method(self):
        """Setup test fixtures"""
        self.agent = get_policy_agent()

    def test_auto_approve_high_confidence(self):
        """Test auto-approve decision with high confidence"""
        agent_outputs = [
            {"from_agent": "rules_engine", "rule_score": 1.0, "confidence": 1.0},
            {"from_agent": "algorithm_agent", "score": 0.95, "confidence": 0.95},
            {"from_agent": "ml_model_agent", "confidence": 0.92},
            {"from_agent": "genai_reasoner", "confidence": 0.90, "citations": [{"source_id": "doc1"}]},
            {"from_agent": "assurance_agent", "assurance_score": 0.88},
        ]

        fusion_score, breakdown = self.agent._fuse_signals(agent_outputs)

        assert fusion_score > 0.85
        assert "rules" in breakdown
        assert "algorithms" in breakdown

    def test_auto_reject_low_confidence(self):
        """Test auto-reject decision with low confidence"""
        agent_outputs = [
            {"from_agent": "rules_engine", "rule_score": 0.2, "confidence": 0.2},
            {"from_agent": "algorithm_agent", "score": 0.15, "confidence": 0.15},
            {"from_agent": "ml_model_agent", "confidence": 0.18},
            {"from_agent": "genai_reasoner", "confidence": 0.20, "citations": []},
            {"from_agent": "assurance_agent", "assurance_score": 0.25},
        ]

        fusion_score, breakdown = self.agent._fuse_signals(agent_outputs)

        assert fusion_score < 0.30

    def test_mandatory_rule_failure(self):
        """Test decision with mandatory rule failure"""
        agent_outputs = [
            {"from_agent": "rules_engine", "rule_score": 0.0, "confidence": 1.0, "failed_rules": ["FR_R1_amount_match"]},
            {"from_agent": "algorithm_agent", "score": 0.95, "confidence": 0.95},
        ]

        from common.schemas import PolicyDecisionRequest

        request = PolicyDecisionRequest(
            case_id="TEST_001",
            from_agent="test",
            to_agent="policy_agent",
            agent_outputs=agent_outputs,
            business_context={},
        )

        response = self.agent.decide(request)

        assert response.decision == DecisionType.AUTO_REJECT.value
        assert response.confidence == 1.0

    def test_missing_evidence(self):
        """Test decision when evidence is needed"""
        agent_outputs = [
            {
                "from_agent": "rules_engine",
                "rule_score": 0.5,
                "confidence": 0.9,
                "needs_evidence": [{"rule_id": "FR_R9", "required_evidence": ["manager_approval"]}],
            },
        ]

        from common.schemas import PolicyDecisionRequest

        request = PolicyDecisionRequest(
            case_id="TEST_002",
            from_agent="test",
            to_agent="policy_agent",
            agent_outputs=agent_outputs,
            business_context={},
        )

        response = self.agent.decide(request)

        assert response.decision == DecisionType.REQUEST_EVIDENCE.value
        assert response.confidence == 0.9

    def test_hallucination_detected(self):
        """Test decision when hallucination is detected"""
        agent_outputs = [
            {"from_agent": "rules_engine", "rule_score": 0.8, "confidence": 0.8},
            {"from_agent": "assurance_agent", "assurance_score": 0.7, "hallucination_detected": True},
        ]

        from common.schemas import PolicyDecisionRequest

        request = PolicyDecisionRequest(
            case_id="TEST_003",
            from_agent="test",
            to_agent="policy_agent",
            agent_outputs=agent_outputs,
            business_context={},
        )

        response = self.agent.decide(request)

        assert response.decision == DecisionType.HUMAN_REVIEW.value

    def test_high_risk_escalation(self):
        """Test escalation for high-risk cases"""
        agent_outputs = [
            {"from_agent": "rules_engine", "rule_score": 0.7, "confidence": 0.7},
            {"from_agent": "algorithm_agent", "score": 0.75, "confidence": 0.75},
            {"from_agent": "assurance_agent", "assurance_score": 0.72},
        ]

        from common.schemas import PolicyDecisionRequest

        request = PolicyDecisionRequest(
            case_id="TEST_004",
            from_agent="test",
            to_agent="policy_agent",
            agent_outputs=agent_outputs,
            business_context={
                "risk_level": "high",
                "transaction_amount": 75000.00,
            },
        )

        response = self.agent.decide(request)

        assert response.decision == DecisionType.ESCALATE.value

    def test_utility_computation(self):
        """Test utility score computation"""
        fusion_score = 0.80

        # Low risk
        utility_low = self.agent._compute_utility(
            fusion_score=fusion_score,
            agent_outputs=[],
            business_context={"risk_level": "low", "transaction_amount": 500.00, "sla_urgency": 0.5},
        )

        # High risk
        utility_high = self.agent._compute_utility(
            fusion_score=fusion_score,
            agent_outputs=[],
            business_context={"risk_level": "high", "transaction_amount": 75000.00, "sla_urgency": 0.5},
        )

        # Low risk should have higher utility
        assert utility_low > utility_high

    def test_config_update(self):
        """Test policy configuration update"""
        original_config = self.agent.get_config()

        new_config = {
            "thresholds": {
                "auto_approve": 0.90,  # Stricter
                "auto_reject": 0.20,
                "human_review": 0.50,
            }
        }

        self.agent.update_config(new_config)
        updated_config = self.agent.get_config()

        assert updated_config["thresholds"]["auto_approve"] == 0.90

    def test_decision_simulation(self):
        """Test decision simulation"""
        result = self.agent.simulate_decision(
            fusion_score=0.88,
            utility_score=0.90,
            business_context={"risk_level": "low"},
        )

        assert "decision" in result
        assert "confidence" in result
        assert "reasoning" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
