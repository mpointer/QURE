"""
Policy Agent

Decision fusion, utility scoring, and threshold-based action routing.
"""

import logging
from typing import Any, Dict, List, Optional

from common.schemas import DecisionType, PolicyDecisionRequest, PolicyDecisionResponse

logger = logging.getLogger(__name__)


class PolicyAgent:
    """
    Policy Agent for decision-making

    Responsibilities:
    - Fuse signals from multiple reasoning agents
    - Compute utility scores based on business objectives
    - Apply decision thresholds and routing logic
    - Enforce risk-based decision policies
    - Handle edge cases and exceptions
    """

    def __init__(self, policy_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Policy Agent

        Args:
            policy_config: Configuration dict with thresholds and weights
        """
        # Default policy configuration
        self.config = policy_config or self._default_config()

        logger.info("✅ Policy Agent initialized")

    def decide(
        self,
        request: PolicyDecisionRequest,
    ) -> PolicyDecisionResponse:
        """
        Make policy decision based on agent outputs

        Args:
            request: PolicyDecisionRequest with reasoning agent outputs

        Returns:
            PolicyDecisionResponse with decision and routing
        """
        case_id = request.case_id
        scores = request.scores
        uncertainty = request.uncertainty
        constraints = request.constraints

        try:
            # 1. Signal fusion: combine agent scores into unified score
            fusion_score, signal_breakdown = self._fuse_signals(scores)

            # 2. Utility computation: map score to business value
            utility_score = self._compute_utility(
                fusion_score=fusion_score,
                scores=scores,
                constraints=constraints,
            )

            # 3. Decision logic: apply thresholds and routing rules
            decision, confidence, reasoning = self._apply_decision_logic(
                fusion_score=fusion_score,
                utility_score=utility_score,
                uncertainty=uncertainty,
                scores=scores,
                constraints=constraints,
            )

            # 4. Build explanation
            explanation = self._build_explanation(
                decision=decision,
                fusion_score=fusion_score,
                utility_score=utility_score,
                signal_breakdown=signal_breakdown,
                reasoning=reasoning,
            )

            logger.info(
                f"Policy decision for case {case_id}: {decision.value} "
                f"(fusion={fusion_score:.3f}, utility={utility_score:.3f}, conf={confidence:.3f})"
            )

            return PolicyDecisionResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                decision=decision,
                confidence=confidence,
                utility_score=utility_score,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Policy decision failed for case {case_id}: {e}")
            return PolicyDecisionResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                decision=DecisionType.HITL_REVIEW,
                confidence=0.0,
                utility_score=0.0,
                explanation=f"Policy error: {str(e)}",
            )

    def _fuse_signals(
        self,
        scores: Dict[str, float],
    ) -> tuple[float, Dict[str, float]]:
        """
        Fuse signals from multiple reasoning agents

        Args:
            scores: Dict of agent_name -> score

        Returns:
            Tuple of (fusion_score, signal_breakdown)
        """
        # Signal weights (configurable)
        weights = self.config.get("signal_weights", {
            "rules": 0.25,
            "algorithm": 0.20,
            "ml": 0.20,
            "genai": 0.20,
            "assurance": 0.15,
        })

        signal_breakdown = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for agent_name, score in scores.items():
            weight = weights.get(agent_name, 0.0)
            signal_breakdown[agent_name] = score
            weighted_sum += score * weight
            total_weight += weight

        # Normalize by total weight
        fusion_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return fusion_score, signal_breakdown

    def _compute_utility(
        self,
        fusion_score: float,
        scores: Dict[str, float],
        constraints: Dict[str, Any],
    ) -> float:
        """
        Compute utility score based on business objectives

        Args:
            fusion_score: Fused signal score
            scores: Agent scores
            constraints: Business constraints and context

        Returns:
            Utility score
        """
        # Business value factors
        transaction_amount = constraints.get("transaction_amount", 0.0)
        risk_level = constraints.get("risk_level", "medium")
        sla_urgency = constraints.get("sla_urgency", 0.5)

        # Compute base utility from fusion score
        base_utility = fusion_score

        # Adjust based on business factors
        if risk_level == "high":
            # High-risk cases require higher confidence
            base_utility *= 0.8
        elif risk_level == "low":
            # Low-risk cases are more forgiving
            base_utility *= 1.2

        # Factor in transaction amount (higher amounts = more caution)
        if transaction_amount > 50000:
            base_utility *= 0.9
        elif transaction_amount > 10000:
            base_utility *= 0.95

        # Factor in SLA urgency
        base_utility *= (1.0 + sla_urgency * 0.1)

        # Clip to [0, 1]
        utility_score = max(0.0, min(1.0, base_utility))

        return utility_score

    def _apply_decision_logic(
        self,
        fusion_score: float,
        utility_score: float,
        uncertainty: float,
        scores: Dict[str, float],
        constraints: Dict[str, Any],
    ) -> tuple[DecisionType, float, List[str]]:
        """
        Apply decision logic with thresholds

        Args:
            fusion_score: Fused signal score
            utility_score: Utility score
            uncertainty: Uncertainty score
            scores: Agent scores
            constraints: Business constraints

        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        reasoning = []

        # Get thresholds from config
        thresholds = self.config.get("thresholds", {
            "auto_resolve": 0.85,
            "reject": 0.30,
            "hitl_review": 0.50,
        })

        # Get assurance score
        assurance_score = scores.get("assurance", 0.0)

        # Decision based on utility score and assurance
        if utility_score >= thresholds["auto_resolve"] and assurance_score >= 0.7:
            reasoning.append(f"High utility ({utility_score:.2%}) and assurance ({assurance_score:.2%})")
            return DecisionType.AUTO_RESOLVE, utility_score, reasoning

        elif utility_score < thresholds["reject"]:
            reasoning.append(f"Low utility score ({utility_score:.2%})")
            return DecisionType.REJECT, 1.0 - utility_score, reasoning

        elif utility_score >= thresholds["hitl_review"]:
            # Medium confidence - check business context
            risk_level = constraints.get("risk_level", "medium")
            transaction_amount = constraints.get("transaction_amount", 0.0)

            if risk_level == "high" or transaction_amount > 50000:
                reasoning.append(f"High-risk or high-value case requires review")
                return DecisionType.HITL_REVIEW, 0.8, reasoning
            else:
                reasoning.append(f"Medium confidence ({utility_score:.2%})")
                return DecisionType.HITL_REVIEW, utility_score, reasoning

        else:
            reasoning.append(f"Score ({utility_score:.2%}) below review threshold")
            return DecisionType.REJECT, 0.7, reasoning

    def _build_explanation(
        self,
        decision: DecisionType,
        fusion_score: float,
        utility_score: float,
        signal_breakdown: Dict[str, float],
        reasoning: List[str],
    ) -> str:
        """
        Build human-readable explanation

        Args:
            decision: Decision type
            fusion_score: Fusion score
            utility_score: Utility score
            signal_breakdown: Signal breakdown dict
            reasoning: Reasoning steps

        Returns:
            Explanation string
        """
        lines = [
            f"DECISION: {decision.value.replace('_', ' ').title()}",
            f"",
            f"Scores:",
            f"  Fusion Score: {fusion_score:.2%}",
            f"  Utility Score: {utility_score:.2%}",
            f"",
            f"Signal Breakdown:",
        ]

        for signal, score in sorted(signal_breakdown.items()):
            lines.append(f"  {signal.title()}: {score:.2%}")

        lines.append(f"")
        lines.append(f"Reasoning:")
        for step in reasoning:
            lines.append(f"  • {step}")

        return "\n".join(lines)

    def _default_config(self) -> Dict[str, Any]:
        """
        Get default policy configuration

        Returns:
            Default config dict
        """
        return {
            "signal_weights": {
                "rules": 0.25,
                "algorithm": 0.20,
                "ml": 0.20,
                "genai": 0.20,
                "assurance": 0.15,
            },
            "thresholds": {
                "auto_resolve": 0.85,
                "reject": 0.30,
                "hitl_review": 0.50,
            },
            "risk_adjustments": {
                "high": 0.8,
                "medium": 1.0,
                "low": 1.2,
            },
        }

    def update_config(self, config: Dict[str, Any]):
        """
        Update policy configuration

        Args:
            config: New configuration dict
        """
        self.config.update(config)
        logger.info("Policy configuration updated")

    def get_config(self) -> Dict[str, Any]:
        """
        Get current policy configuration

        Returns:
            Config dict
        """
        return self.config.copy()

    def simulate_decision(
        self,
        fusion_score: float,
        utility_score: float,
        business_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate decision for testing/debugging

        Args:
            fusion_score: Fusion score
            utility_score: Utility score
            business_context: Business context

        Returns:
            Simulation result dict
        """
        if business_context is None:
            business_context = {}

        decision, confidence, reasoning = self._apply_decision_logic(
            fusion_score=fusion_score,
            utility_score=utility_score,
            agent_outputs=[],
            business_context=business_context,
        )

        return {
            "decision": decision.value,
            "confidence": confidence,
            "reasoning": reasoning,
            "fusion_score": fusion_score,
            "utility_score": utility_score,
        }


# Singleton instance
_policy_agent: Optional[PolicyAgent] = None


def get_policy_agent(policy_config: Optional[Dict[str, Any]] = None) -> PolicyAgent:
    """
    Get or create singleton PolicyAgent instance

    Args:
        policy_config: Policy configuration dict

    Returns:
        PolicyAgent instance
    """
    global _policy_agent

    if _policy_agent is None:
        _policy_agent = PolicyAgent(policy_config=policy_config)

    return _policy_agent
