"""
Reconciliation service orchestrating QRU agents

Handles:
- Case creation and management
- Agent execution
- Result persistence
- Audit logging
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database.models import (
    ReconciliationCase,
    AgentDecision as DBAgentDecision,
    AuditLog,
    ReconciliationStatus,
    DecisionType
)
from backend.agents.base_agent import AgentContext, QRUOrchestrator
from backend.agents.reconciliation_agents import create_reconciliation_agents
from backend.services.llm_service import LLMService
from backend.config.settings import Settings


class ReconciliationService:
    """
    Service for managing reconciliation workflow

    Coordinates:
    - Agent execution via QRU orchestrator
    - Database persistence
    - Audit trail
    - Business rule application
    """

    def __init__(self, db_session: AsyncSession, settings: Settings):
        self.db = db_session
        self.settings = settings
        self.llm_service = LLMService(settings)

        # Create agents
        q_agent, r_agent, u_agent = create_reconciliation_agents(self.llm_service)
        self.orchestrator = QRUOrchestrator(q_agent, r_agent, u_agent)

    async def create_case(
        self,
        vertical: str,
        data1: Dict[str, Any],
        data2: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> ReconciliationCase:
        """
        Create new reconciliation case

        Args:
            vertical: Business vertical (e.g., "finance", "payroll")
            data1: First transaction data (GL/Source)
            data2: Second transaction data (Bank/Target)
            case_id: Optional custom case ID

        Returns:
            Created ReconciliationCase
        """
        # Generate case ID if not provided
        if not case_id:
            case_id = f"{vertical}_{uuid.uuid4().hex[:12]}"

        # Create case
        case = ReconciliationCase(
            case_id=case_id,
            vertical=vertical,
            data1=data1,
            data2=data2,
            status=ReconciliationStatus.PENDING,
            llm_provider=str(self.settings.llm_provider.value)
        )

        self.db.add(case)
        await self.db.commit()
        await self.db.refresh(case)

        # Audit log
        await self._log_audit(
            action="create_case",
            entity_type="reconciliation_case",
            entity_id=case_id,
            details={"vertical": vertical},
            outcome="success"
        )

        return case

    async def reconcile(
        self,
        case_id: str,
        business_rules: Optional[List[Dict[str, Any]]] = None
    ) -> ReconciliationCase:
        """
        Execute reconciliation workflow for case

        Args:
            case_id: Case identifier
            business_rules: Optional business rules to apply

        Returns:
            Updated ReconciliationCase with results
        """
        # Fetch case
        result = await self.db.execute(
            select(ReconciliationCase).where(ReconciliationCase.case_id == case_id)
        )
        case = result.scalar_one_or_none()

        if not case:
            raise ValueError(f"Case not found: {case_id}")

        # Update status
        case.status = ReconciliationStatus.IN_PROGRESS
        await self.db.commit()

        start_time = datetime.utcnow()

        try:
            # Build context
            context = AgentContext(
                case_id=case_id,
                vertical=case.vertical,
                data1=case.data1,
                data2=case.data2,
                business_rules=business_rules or [],
                previous_decisions=[],
                metadata={}
            )

            # Execute QRU workflow
            agent_decisions = await self.orchestrator.execute(context)

            # Get final decision
            final_decision = self.orchestrator.get_final_decision(agent_decisions)

            # Persist agent decisions
            for decision in agent_decisions:
                db_decision = DBAgentDecision(
                    case_id=case.id,
                    agent_name=decision.agent_name,
                    agent_role=decision.agent_role.value,
                    decision=decision.decision.value if decision.decision else None,
                    confidence_score=decision.confidence_score,
                    reasoning=decision.reasoning,
                    rules_applied=decision.rules_applied,
                    evidence=decision.evidence,
                    execution_time_ms=decision.execution_time_ms
                )
                self.db.add(db_decision)

            # Update case with final results
            case.decision = final_decision.decision if final_decision else None
            case.confidence_score = final_decision.confidence_score if final_decision else None
            case.match_score = final_decision.evidence.get("final_match_score") if final_decision else None
            case.agent_reasoning = {
                "question": agent_decisions[0].dict() if len(agent_decisions) > 0 else None,
                "reason": agent_decisions[1].dict() if len(agent_decisions) > 1 else None,
                "update": agent_decisions[2].dict() if len(agent_decisions) > 2 else None
            }
            case.status = ReconciliationStatus.COMPLETED
            case.completed_at = datetime.utcnow()
            case.processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            await self.db.commit()
            await self.db.refresh(case)

            # Audit log
            await self._log_audit(
                action="reconcile_case",
                entity_type="reconciliation_case",
                entity_id=case_id,
                details={
                    "decision": case.decision.value if case.decision else None,
                    "confidence_score": case.confidence_score,
                    "processing_time_ms": case.processing_time_ms
                },
                outcome="success"
            )

            return case

        except Exception as e:
            # Mark as failed
            case.status = ReconciliationStatus.FAILED
            await self.db.commit()

            # Audit log
            await self._log_audit(
                action="reconcile_case",
                entity_type="reconciliation_case",
                entity_id=case_id,
                details={},
                outcome="failed",
                error_message=str(e)
            )

            raise

    async def get_case(self, case_id: str) -> Optional[ReconciliationCase]:
        """
        Retrieve case by ID

        Args:
            case_id: Case identifier

        Returns:
            ReconciliationCase or None
        """
        result = await self.db.execute(
            select(ReconciliationCase).where(ReconciliationCase.case_id == case_id)
        )
        return result.scalar_one_or_none()

    async def list_cases(
        self,
        vertical: Optional[str] = None,
        status: Optional[ReconciliationStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ReconciliationCase]:
        """
        List cases with optional filters

        Args:
            vertical: Filter by vertical
            status: Filter by status
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of ReconciliationCase
        """
        query = select(ReconciliationCase)

        if vertical:
            query = query.where(ReconciliationCase.vertical == vertical)
        if status:
            query = query.where(ReconciliationCase.status == status)

        query = query.limit(limit).offset(offset).order_by(ReconciliationCase.created_at.desc())

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_agent_decisions(self, case_id: str) -> List[DBAgentDecision]:
        """
        Get all agent decisions for a case

        Args:
            case_id: Case identifier

        Returns:
            List of AgentDecision
        """
        # Get case
        result = await self.db.execute(
            select(ReconciliationCase).where(ReconciliationCase.case_id == case_id)
        )
        case = result.scalar_one_or_none()

        if not case:
            return []

        # Get decisions
        result = await self.db.execute(
            select(DBAgentDecision)
            .where(DBAgentDecision.case_id == case.id)
            .order_by(DBAgentDecision.created_at)
        )
        return result.scalars().all()

    async def _log_audit(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        details: Dict[str, Any],
        outcome: str,
        error_message: Optional[str] = None
    ):
        """
        Log audit entry

        Args:
            action: Action performed
            entity_type: Type of entity
            entity_id: Entity identifier
            details: Additional details
            outcome: Outcome (success/failed)
            error_message: Optional error message
        """
        audit = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            outcome=outcome,
            error_message=error_message
        )
        self.db.add(audit)
        await self.db.commit()

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get reconciliation statistics

        Returns:
            Statistics dictionary
        """
        from sqlalchemy import func

        # Total cases
        total_result = await self.db.execute(
            select(func.count(ReconciliationCase.id))
        )
        total_cases = total_result.scalar()

        # Cases by status
        status_result = await self.db.execute(
            select(
                ReconciliationCase.status,
                func.count(ReconciliationCase.id)
            ).group_by(ReconciliationCase.status)
        )
        status_counts = {status: count for status, count in status_result.all()}

        # Cases by decision
        decision_result = await self.db.execute(
            select(
                ReconciliationCase.decision,
                func.count(ReconciliationCase.id)
            ).group_by(ReconciliationCase.decision)
        )
        decision_counts = {
            decision.value if decision else "none": count
            for decision, count in decision_result.all()
        }

        # Average confidence score
        avg_confidence_result = await self.db.execute(
            select(func.avg(ReconciliationCase.confidence_score))
            .where(ReconciliationCase.confidence_score.isnot(None))
        )
        avg_confidence = avg_confidence_result.scalar() or 0.0

        # Average processing time
        avg_time_result = await self.db.execute(
            select(func.avg(ReconciliationCase.processing_time_ms))
            .where(ReconciliationCase.processing_time_ms.isnot(None))
        )
        avg_processing_time = avg_time_result.scalar() or 0

        return {
            "total_cases": total_cases,
            "by_status": status_counts,
            "by_decision": decision_counts,
            "avg_confidence_score": float(avg_confidence),
            "avg_processing_time_ms": int(avg_processing_time)
        }
