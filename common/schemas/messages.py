"""
Inter-agent message schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .base import (
    AgentType,
    CaseStatus,
    DecisionType,
    Document,
    Evidence,
    FeatureVector,
    TextSpan,
    VerticalType,
)


class AgentMessage(BaseModel):
    """Base message for inter-agent communication"""
    message_id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., description="Case being processed")
    from_agent: AgentType
    to_agent: Optional[AgentType] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaseRequest(AgentMessage):
    """Initial case submission"""
    vertical: VerticalType
    case_data: Dict[str, Any] = Field(..., description="Raw case data")
    priority: int = Field(1, ge=1, le=10)


class RetrievalRequest(AgentMessage):
    """Request to retrieve data"""
    source_ids: List[str] = Field(..., description="List of source IDs to retrieve")
    source_types: List[str] = Field(default_factory=list, description="Types (s3, db, api)")


class RetrievalResponse(AgentMessage):
    """Response with retrieved documents"""
    documents: List[Document]
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict)


class DataProcessingRequest(AgentMessage):
    """Request to process/normalize data"""
    documents: List[Document]
    processing_config: Dict[str, Any] = Field(default_factory=dict)


class DataProcessingResponse(AgentMessage):
    """Response with processed data"""
    documents: List[Document] = Field(..., description="Enriched documents")
    graph_updates: List[Dict[str, Any]] = Field(default_factory=list)
    feature_vectors: List[FeatureVector] = Field(default_factory=list)
    embeddings_stored: int = 0


class RulesEvaluationRequest(AgentMessage):
    """Request rules evaluation"""
    case_data: Dict[str, Any]
    rule_set: str = Field(..., description="Rule set name (e.g., 'finance_reconciliation')")


class RulesEvaluationResponse(AgentMessage):
    """Response with rules evaluation"""
    passed_rules: List[str] = Field(default_factory=list)
    failed_rules: List[str] = Field(default_factory=list)
    needs_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    rule_score: float = Field(0.0, ge=0.0, le=1.0)
    explanations: List[str] = Field(default_factory=list)


class AlgorithmRequest(AgentMessage):
    """Request algorithmic computation"""
    algorithm_type: str = Field(..., description="e.g., 'fuzzy_match', 'graph_path'")
    inputs: Dict[str, Any]


class AlgorithmResponse(AgentMessage):
    """Response with algorithm results"""
    algorithm_type: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    result: Any
    explanation: str


class MLPredictionRequest(AgentMessage):
    """Request ML prediction"""
    model_name: str = Field(..., description="Model identifier")
    features: FeatureVector


class MLPredictionResponse(AgentMessage):
    """Response with ML prediction"""
    model_name: str
    prediction: Any
    probability: float = Field(0.0, ge=0.0, le=1.0)
    confidence_interval: Optional[tuple[float, float]] = None
    explanation: Optional[str] = None


class GenAIRequest(AgentMessage):
    """Request GenAI reasoning"""
    task: str = Field(..., description="Task type (extract_evidence, entailment, generate)")
    prompt: str
    documents: List[Document] = Field(default_factory=list)
    structured_output_schema: Optional[Dict[str, Any]] = None


class GenAIResponse(AgentMessage):
    """Response with GenAI output"""
    task: str
    output: Any
    evidence: List[Evidence] = Field(default_factory=list)
    citations: List[TextSpan] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    grounding_rate: float = Field(0.0, ge=0.0, le=1.0, description="% with valid citations")


class AssuranceRequest(AgentMessage):
    """Request uncertainty scoring"""
    ml_variance: Optional[float] = None
    llm_confidence: Optional[float] = None
    rule_conflicts: List[str] = Field(default_factory=list)
    algorithm_noise: Optional[float] = None


class AssuranceResponse(AgentMessage):
    """Response with uncertainty score"""
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list, description="Warning flags")


class PolicyRequest(AgentMessage):
    """Request policy decision"""
    scores: Dict[str, float] = Field(..., description="Scores from each modality")
    uncertainty: float
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PolicyResponse(AgentMessage):
    """Response with policy decision"""
    decision: DecisionType
    utility_score: float
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list)


class ActionRequest(AgentMessage):
    """Request action execution"""
    action_type: str = Field(..., description="e.g., 'write_back', 'generate_letter'")
    action_params: Dict[str, Any]
    dry_run: bool = Field(False, description="If true, don't execute")


class ActionResponse(AgentMessage):
    """Response with action results"""
    action_type: str
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class FeedbackMessage(AgentMessage):
    """Feedback for learning loop"""
    decision: DecisionType
    outcome: str = Field(..., description="actual outcome")
    reward: float = Field(..., description="Computed reward")
    context: Dict[str, Any] = Field(default_factory=dict)
    propensity: float = Field(..., ge=0.0, le=1.0, description="Action selection probability")


class CaseCompletionMessage(AgentMessage):
    """Final case completion message"""
    case_id: str
    status: CaseStatus
    decision: DecisionType
    final_score: float
    execution_time_seconds: float
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)
