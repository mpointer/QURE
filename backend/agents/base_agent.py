"""
Base agent classes for QRU (Question-Reason-Update) pattern

Implements the core agent abstraction used throughout QURE.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Agent roles in QRU pattern"""
    QUESTION = "question"
    REASON = "reason"
    UPDATE = "update"


class DecisionType(str, Enum):
    """Types of decisions agents can make"""
    AUTO_RESOLVE = "auto_resolve"
    HITL_REVIEW = "hitl_review"
    REJECT = "reject"
    ESCALATE = "escalate"
    REQUEST_EVIDENCE = "request_evidence"


class AgentContext(BaseModel):
    """Context passed to agents during execution"""
    case_id: str
    vertical: str
    data1: Dict[str, Any]  # GL/Source transaction
    data2: Dict[str, Any]  # Bank/Target transaction
    business_rules: List[Dict[str, Any]] = Field(default_factory=list)
    previous_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    """Agent decision output"""
    agent_name: str
    agent_role: AgentRole
    decision: Optional[DecisionType] = None
    confidence_score: float
    reasoning: str
    rules_applied: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseAgent(ABC):
    """
    Base agent class implementing QRU pattern

    All agents must implement:
    - execute(): Main decision-making logic
    - get_system_prompt(): Agent-specific instructions
    """

    def __init__(self, name: str, role: AgentRole, llm_service=None):
        self.name = name
        self.role = role
        self.llm_service = llm_service

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentDecision:
        """
        Execute agent logic and return decision

        Args:
            context: Agent execution context

        Returns:
            AgentDecision with reasoning and confidence score
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get agent-specific system prompt

        Returns:
            System prompt for LLM
        """
        pass

    def _build_user_prompt(self, context: AgentContext) -> str:
        """
        Build user prompt from context

        Args:
            context: Agent execution context

        Returns:
            Formatted user prompt
        """
        prompt_parts = [
            f"# Reconciliation Case: {context.case_id}",
            f"**Vertical:** {context.vertical}",
            "",
            "## Transaction 1 (GL/Source)",
            self._format_transaction(context.data1),
            "",
            "## Transaction 2 (Bank/Target)",
            self._format_transaction(context.data2),
        ]

        if context.business_rules:
            prompt_parts.extend([
                "",
                "## Applicable Business Rules",
                self._format_business_rules(context.business_rules)
            ])

        if context.previous_decisions:
            prompt_parts.extend([
                "",
                "## Previous Agent Decisions",
                self._format_previous_decisions(context.previous_decisions)
            ])

        return "\n".join(prompt_parts)

    def _format_transaction(self, data: Dict[str, Any]) -> str:
        """Format transaction data for prompt"""
        lines = []
        for key, value in data.items():
            lines.append(f"- **{key}:** {value}")
        return "\n".join(lines)

    def _format_business_rules(self, rules: List[Dict[str, Any]]) -> str:
        """Format business rules for prompt"""
        lines = []
        for rule in rules:
            lines.append(f"- **{rule.get('name', 'Unknown')}:** {rule.get('description', 'N/A')}")
        return "\n".join(lines)

    def _format_previous_decisions(self, decisions: List[Dict[str, Any]]) -> str:
        """Format previous decisions for prompt"""
        lines = []
        for decision in decisions:
            lines.append(
                f"- **{decision.get('agent_name')}** ({decision.get('agent_role')}): "
                f"{decision.get('reasoning', 'N/A')} "
                f"[Confidence: {decision.get('confidence_score', 0):.2%}]"
            )
        return "\n".join(lines)

    async def _call_llm(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Call LLM service with prompts

        Args:
            system_prompt: System instructions
            user_prompt: User query
            **kwargs: Additional LLM parameters

        Returns:
            LLM response text
        """
        if not self.llm_service:
            raise ValueError("LLM service not configured for agent")

        return await self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            **kwargs
        )

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse structured response from LLM

        Expects JSON format with:
        - decision: DecisionType
        - confidence_score: float
        - reasoning: str
        - rules_applied: List[str]
        - evidence: Dict[str, Any]

        Args:
            response: Raw LLM response

        Returns:
            Parsed response dictionary
        """
        import json

        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # No JSON found, return minimal structure
                return {
                    "decision": None,
                    "confidence_score": 0.5,
                    "reasoning": response,
                    "rules_applied": [],
                    "evidence": {}
                }
        except json.JSONDecodeError:
            # Invalid JSON, return minimal structure
            return {
                "decision": None,
                "confidence_score": 0.5,
                "reasoning": response,
                "rules_applied": [],
                "evidence": {}
            }


class QRUOrchestrator:
    """
    Orchestrates the Question-Reason-Update agent workflow

    Coordinates multiple agents following the QRU pattern:
    1. Question agent analyzes the reconciliation case
    2. Reason agent provides detailed reasoning
    3. Update agent makes final decision
    """

    def __init__(
        self,
        question_agent: BaseAgent,
        reason_agent: BaseAgent,
        update_agent: BaseAgent
    ):
        self.question_agent = question_agent
        self.reason_agent = reason_agent
        self.update_agent = update_agent

    async def execute(self, context: AgentContext) -> List[AgentDecision]:
        """
        Execute QRU workflow

        Args:
            context: Agent execution context

        Returns:
            List of agent decisions in execution order
        """
        decisions = []

        # Question phase
        q_decision = await self.question_agent.execute(context)
        decisions.append(q_decision)

        # Update context with Question decision
        context.previous_decisions.append(q_decision.dict())

        # Reason phase
        r_decision = await self.reason_agent.execute(context)
        decisions.append(r_decision)

        # Update context with Reason decision
        context.previous_decisions.append(r_decision.dict())

        # Update phase
        u_decision = await self.update_agent.execute(context)
        decisions.append(u_decision)

        return decisions

    def get_final_decision(self, decisions: List[AgentDecision]) -> AgentDecision:
        """
        Get final decision from QRU workflow

        Args:
            decisions: List of agent decisions

        Returns:
            Final decision (from Update agent)
        """
        # Return last decision (Update agent)
        return decisions[-1] if decisions else None
