"""
FastAPI routes for QURE reconciliation API

Endpoints:
- POST /api/cases - Create new reconciliation case
- POST /api/cases/{case_id}/reconcile - Execute reconciliation
- GET /api/cases/{case_id} - Get case details
- GET /api/cases - List cases
- GET /api/statistics - Get system statistics
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.database.models import ReconciliationStatus, DecisionType
from backend.services.reconciliation_service import ReconciliationService
from backend.config.settings import settings


# Pydantic models for API
class CreateCaseRequest(BaseModel):
    """Request to create new reconciliation case"""
    vertical: str = Field(..., description="Business vertical (e.g., finance, payroll)")
    data1: Dict[str, Any] = Field(..., description="First transaction (GL/Source)")
    data2: Dict[str, Any] = Field(..., description="Second transaction (Bank/Target)")
    case_id: Optional[str] = Field(None, description="Optional custom case ID")


class ReconcileRequest(BaseModel):
    """Request to execute reconciliation"""
    business_rules: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional business rules to apply"
    )


class CaseResponse(BaseModel):
    """Response with case details"""
    id: int
    case_id: str
    vertical: str
    status: str
    decision: Optional[str]
    confidence_score: Optional[float]
    match_score: Optional[float]
    data1: Dict[str, Any]
    data2: Dict[str, Any]
    agent_reasoning: Optional[Dict[str, Any]]
    created_at: str
    updated_at: Optional[str]
    completed_at: Optional[str]
    processing_time_ms: Optional[int]
    llm_provider: Optional[str]

    class Config:
        from_attributes = True


class AgentDecisionResponse(BaseModel):
    """Response with agent decision details"""
    id: int
    agent_name: str
    agent_role: str
    decision: Optional[str]
    confidence_score: float
    reasoning: Optional[str]
    rules_applied: Optional[List[str]]
    evidence: Optional[Dict[str, Any]]
    execution_time_ms: Optional[int]
    created_at: str

    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Response with system statistics"""
    total_cases: int
    by_status: Dict[str, int]
    by_decision: Dict[str, int]
    avg_confidence_score: float
    avg_processing_time_ms: int


# Router
router = APIRouter(prefix="/api", tags=["reconciliation"])


# Dependency to get database session
async def get_db():
    """Get database session"""
    from backend.database.database import async_session_maker

    async with async_session_maker() as session:
        yield session


# Dependency to get reconciliation service
async def get_reconciliation_service(
    db: AsyncSession = Depends(get_db)
) -> ReconciliationService:
    """Get reconciliation service instance"""
    return ReconciliationService(db, settings)


@router.post("/cases", response_model=CaseResponse, status_code=201)
async def create_case(
    request: CreateCaseRequest,
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """
    Create new reconciliation case

    Creates a case with pending status. Use /reconcile endpoint to execute.
    """
    case = await service.create_case(
        vertical=request.vertical,
        data1=request.data1,
        data2=request.data2,
        case_id=request.case_id
    )

    return CaseResponse(
        id=case.id,
        case_id=case.case_id,
        vertical=case.vertical,
        status=case.status.value,
        decision=case.decision.value if case.decision else None,
        confidence_score=case.confidence_score,
        match_score=case.match_score,
        data1=case.data1,
        data2=case.data2,
        agent_reasoning=case.agent_reasoning,
        created_at=case.created_at.isoformat(),
        updated_at=case.updated_at.isoformat() if case.updated_at else None,
        completed_at=case.completed_at.isoformat() if case.completed_at else None,
        processing_time_ms=case.processing_time_ms,
        llm_provider=case.llm_provider
    )


@router.post("/cases/{case_id}/reconcile", response_model=CaseResponse)
async def reconcile_case(
    case_id: str,
    request: ReconcileRequest = ReconcileRequest(),
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """
    Execute reconciliation for case

    Runs QRU agent workflow and returns results.
    """
    try:
        case = await service.reconcile(
            case_id=case_id,
            business_rules=request.business_rules
        )

        return CaseResponse(
            id=case.id,
            case_id=case.case_id,
            vertical=case.vertical,
            status=case.status.value,
            decision=case.decision.value if case.decision else None,
            confidence_score=case.confidence_score,
            match_score=case.match_score,
            data1=case.data1,
            data2=case.data2,
            agent_reasoning=case.agent_reasoning,
            created_at=case.created_at.isoformat(),
            updated_at=case.updated_at.isoformat() if case.updated_at else None,
            completed_at=case.completed_at.isoformat() if case.completed_at else None,
            processing_time_ms=case.processing_time_ms,
            llm_provider=case.llm_provider
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {str(e)}")


@router.get("/cases/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """
    Get case details by ID
    """
    case = await service.get_case(case_id)

    if not case:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    return CaseResponse(
        id=case.id,
        case_id=case.case_id,
        vertical=case.vertical,
        status=case.status.value,
        decision=case.decision.value if case.decision else None,
        confidence_score=case.confidence_score,
        match_score=case.match_score,
        data1=case.data1,
        data2=case.data2,
        agent_reasoning=case.agent_reasoning,
        created_at=case.created_at.isoformat(),
        updated_at=case.updated_at.isoformat() if case.updated_at else None,
        completed_at=case.completed_at.isoformat() if case.completed_at else None,
        processing_time_ms=case.processing_time_ms,
        llm_provider=case.llm_provider
    )


@router.get("/cases", response_model=List[CaseResponse])
async def list_cases(
    vertical: Optional[str] = Query(None, description="Filter by vertical"),
    status: Optional[ReconciliationStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """
    List reconciliation cases with optional filters
    """
    cases = await service.list_cases(
        vertical=vertical,
        status=status,
        limit=limit,
        offset=offset
    )

    return [
        CaseResponse(
            id=case.id,
            case_id=case.case_id,
            vertical=case.vertical,
            status=case.status.value,
            decision=case.decision.value if case.decision else None,
            confidence_score=case.confidence_score,
            match_score=case.match_score,
            data1=case.data1,
            data2=case.data2,
            agent_reasoning=case.agent_reasoning,
            created_at=case.created_at.isoformat(),
            updated_at=case.updated_at.isoformat() if case.updated_at else None,
            completed_at=case.completed_at.isoformat() if case.completed_at else None,
            processing_time_ms=case.processing_time_ms,
            llm_provider=case.llm_provider
        )
        for case in cases
    ]


@router.get("/cases/{case_id}/decisions", response_model=List[AgentDecisionResponse])
async def get_agent_decisions(
    case_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """
    Get all agent decisions for a case
    """
    decisions = await service.get_agent_decisions(case_id)

    return [
        AgentDecisionResponse(
            id=decision.id,
            agent_name=decision.agent_name,
            agent_role=decision.agent_role,
            decision=decision.decision,
            confidence_score=decision.confidence_score,
            reasoning=decision.reasoning,
            rules_applied=decision.rules_applied,
            evidence=decision.evidence,
            execution_time_ms=decision.execution_time_ms,
            created_at=decision.created_at.isoformat()
        )
        for decision in decisions
    ]


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    service: ReconciliationService = Depends(get_reconciliation_service)
):
    """
    Get system statistics
    """
    stats = await service.get_statistics()

    return StatisticsResponse(
        total_cases=stats["total_cases"],
        by_status=stats["by_status"],
        by_decision=stats["by_decision"],
        avg_confidence_score=stats["avg_confidence_score"],
        avg_processing_time_ms=stats["avg_processing_time_ms"]
    )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "QURE Reconciliation API",
        "version": settings.app_version
    }
