"""
Reconciliation agents implementing QRU pattern

Specialized agents for financial reconciliation:
- QuestionAgent: Analyzes discrepancies and formulates questions
- ReasonAgent: Provides detailed reasoning about the reconciliation
- UpdateAgent: Makes final decision on reconciliation action
"""

from typing import Dict, Any
import time

from backend.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentDecision,
    DecisionType
)


class QuestionAgent(BaseAgent):
    """
    Question Agent - First phase of QRU

    Analyzes the reconciliation case and identifies:
    - Key discrepancies between transactions
    - Missing information
    - Potential matching criteria
    - Questions that need to be answered
    """

    def __init__(self, llm_service=None):
        super().__init__(
            name="Question Agent",
            role=AgentRole.QUESTION,
            llm_service=llm_service
        )

    def get_system_prompt(self) -> str:
        return """You are the Question Agent in a financial reconciliation system.

Your role is to:
1. Analyze two transactions that may or may not match
2. Identify key similarities and differences
3. Determine what questions need to be answered to reconcile them
4. Assess initial confidence in potential match

Focus on:
- Amount matching (exact or within tolerance)
- Date proximity
- Party/entity matching
- Reference number alignment
- Transaction type compatibility

Output your analysis as JSON with this structure:
{
    "decision": null,  # Question agent doesn't make final decision
    "confidence_score": 0.0-1.0,  # How confident you are that these CAN be reconciled
    "reasoning": "Detailed analysis of similarities and differences",
    "rules_applied": ["rule1", "rule2"],
    "evidence": {
        "amount_match": true/false,
        "date_within_tolerance": true/false,
        "party_match": true/false,
        "discrepancies": ["list", "of", "issues"],
        "key_questions": ["What about X?", "Is Y acceptable?"]
    }
}

Be thorough but concise. Your analysis will guide the Reason agent."""

    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute Question agent logic"""
        start_time = time.time()

        # Build prompts
        system_prompt = self.get_system_prompt()
        user_prompt = self._build_user_prompt(context)

        # Call LLM
        response = await self._call_llm(system_prompt, user_prompt)

        # Parse response
        parsed = self._parse_llm_response(response)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Build decision
        decision = AgentDecision(
            agent_name=self.name,
            agent_role=self.role,
            decision=None,  # Question agent doesn't make final decision
            confidence_score=parsed.get("confidence_score", 0.5),
            reasoning=parsed.get("reasoning", ""),
            rules_applied=parsed.get("rules_applied", []),
            evidence=parsed.get("evidence", {}),
            execution_time_ms=execution_time_ms
        )

        return decision


class ReasonAgent(BaseAgent):
    """
    Reason Agent - Second phase of QRU

    Provides detailed reasoning about the reconciliation:
    - Evaluates evidence from Question agent
    - Applies business rules
    - Determines if auto-resolution is appropriate
    - Identifies need for human review
    """

    def __init__(self, llm_service=None):
        super().__init__(
            name="Reason Agent",
            role=AgentRole.REASON,
            llm_service=llm_service
        )

    def get_system_prompt(self) -> str:
        return """You are the Reason Agent in a financial reconciliation system.

Your role is to:
1. Review the Question agent's analysis
2. Apply business rules and reconciliation logic
3. Determine the appropriate action path
4. Provide detailed reasoning for your recommendation

Decision types:
- AUTO_RESOLVE: Clear match, can reconcile automatically
- HITL_REVIEW: Needs human review (close match but uncertain)
- REJECT: Clear mismatch, should not reconcile
- ESCALATE: Complex case requiring expert review
- REQUEST_EVIDENCE: Need additional documentation

Consider:
- Materiality thresholds (e.g., $0.01 tolerance for amounts)
- Time windows (e.g., 3-day settlement periods)
- Business rule compliance
- Risk levels based on amounts
- Historical patterns

Output your analysis as JSON:
{
    "decision": "AUTO_RESOLVE|HITL_REVIEW|REJECT|ESCALATE|REQUEST_EVIDENCE",
    "confidence_score": 0.0-1.0,
    "reasoning": "Detailed explanation of why this decision is appropriate",
    "rules_applied": ["Amount tolerance rule", "Date window rule"],
    "evidence": {
        "match_score": 0.0-1.0,
        "risk_level": "low|medium|high",
        "supporting_factors": ["factor1", "factor2"],
        "concerns": ["concern1", "concern2"]
    }
}

Be conservative with AUTO_RESOLVE - only use when extremely confident."""

    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute Reason agent logic"""
        start_time = time.time()

        # Build prompts
        system_prompt = self.get_system_prompt()
        user_prompt = self._build_user_prompt(context)

        # Call LLM
        response = await self._call_llm(system_prompt, user_prompt)

        # Parse response
        parsed = self._parse_llm_response(response)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Validate decision type
        decision_str = parsed.get("decision")
        try:
            decision = DecisionType(decision_str) if decision_str else None
        except ValueError:
            decision = DecisionType.HITL_REVIEW  # Default to human review if invalid

        # Build decision
        decision_obj = AgentDecision(
            agent_name=self.name,
            agent_role=self.role,
            decision=decision,
            confidence_score=parsed.get("confidence_score", 0.5),
            reasoning=parsed.get("reasoning", ""),
            rules_applied=parsed.get("rules_applied", []),
            evidence=parsed.get("evidence", {}),
            execution_time_ms=execution_time_ms
        )

        return decision_obj


class UpdateAgent(BaseAgent):
    """
    Update Agent - Final phase of QRU

    Makes the final decision on reconciliation:
    - Reviews all previous agent decisions
    - Applies final validation checks
    - Determines final action
    - Provides executive summary
    """

    def __init__(self, llm_service=None):
        super().__init__(
            name="Update Agent",
            role=AgentRole.UPDATE,
            llm_service=llm_service
        )

    def get_system_prompt(self) -> str:
        return """You are the Update Agent in a financial reconciliation system.

Your role is to:
1. Review decisions from Question and Reason agents
2. Validate their recommendations
3. Make the final decision
4. Provide executive summary

You are the final decision-maker. Consider:
- Do you agree with the Reason agent's recommendation?
- Is the confidence score appropriate?
- Are there any red flags that were missed?
- Should this case be escalated despite recommendations?

Decision types:
- AUTO_RESOLVE: Approve automatic reconciliation
- HITL_REVIEW: Send to human reviewer
- REJECT: Reject reconciliation
- ESCALATE: Escalate to senior reviewer
- REQUEST_EVIDENCE: Request additional documentation

Output your final decision as JSON:
{
    "decision": "AUTO_RESOLVE|HITL_REVIEW|REJECT|ESCALATE|REQUEST_EVIDENCE",
    "confidence_score": 0.0-1.0,
    "reasoning": "Executive summary explaining final decision",
    "rules_applied": ["Final validation rule"],
    "evidence": {
        "agreement_with_reason_agent": true/false,
        "final_match_score": 0.0-1.0,
        "key_decision_factors": ["factor1", "factor2"],
        "recommended_action": "Specific next steps"
    }
}

Your decision is final and will be recorded in the system."""

    async def execute(self, context: AgentContext) -> AgentDecision:
        """Execute Update agent logic"""
        start_time = time.time()

        # Build prompts
        system_prompt = self.get_system_prompt()
        user_prompt = self._build_user_prompt(context)

        # Call LLM
        response = await self._call_llm(system_prompt, user_prompt)

        # Parse response
        parsed = self._parse_llm_response(response)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Validate decision type
        decision_str = parsed.get("decision")
        try:
            decision = DecisionType(decision_str) if decision_str else DecisionType.HITL_REVIEW
        except ValueError:
            decision = DecisionType.HITL_REVIEW  # Default to human review if invalid

        # Build decision
        decision_obj = AgentDecision(
            agent_name=self.name,
            agent_role=self.role,
            decision=decision,
            confidence_score=parsed.get("confidence_score", 0.5),
            reasoning=parsed.get("reasoning", ""),
            rules_applied=parsed.get("rules_applied", []),
            evidence=parsed.get("evidence", {}),
            execution_time_ms=execution_time_ms
        )

        return decision_obj


def create_reconciliation_agents(llm_service) -> tuple[QuestionAgent, ReasonAgent, UpdateAgent]:
    """
    Factory function to create all reconciliation agents

    Args:
        llm_service: LLM service instance

    Returns:
        Tuple of (QuestionAgent, ReasonAgent, UpdateAgent)
    """
    question_agent = QuestionAgent(llm_service=llm_service)
    reason_agent = ReasonAgent(llm_service=llm_service)
    update_agent = UpdateAgent(llm_service=llm_service)

    return question_agent, reason_agent, update_agent
