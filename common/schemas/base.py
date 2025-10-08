"""
Base schemas and types used across QURE agents
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    """Case processing status"""
    PENDING = "pending"
    RETRIEVING = "retrieving"
    PROCESSING = "processing"
    REASONING = "reasoning"
    DECIDING = "deciding"
    ACTING = "acting"
    COMPLETED = "completed"
    FAILED = "failed"
    HITL_REVIEW = "hitl_review"
    REQUEST_INFO = "request_info"


class DecisionType(str, Enum):
    """Final decision types"""
    AUTO_RESOLVE = "auto_resolve"
    HITL_REVIEW = "hitl_review"
    REQUEST_INFO = "request_info"
    REJECT = "reject"


class VerticalType(str, Enum):
    """Business vertical types"""
    FINANCE = "finance"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"


class AgentType(str, Enum):
    """Agent types in the system"""
    ORCHESTRATION = "orchestration"
    RETRIEVER = "retriever"
    DATA = "data"
    RULES = "rules"
    ALGORITHM = "algorithm"
    ML = "ml"
    GENAI = "genai"
    ASSURANCE = "assurance"
    POLICY = "policy"
    ACTION = "action"
    LEARNING = "learning"


class BaseEntity(BaseModel):
    """Base entity with common fields"""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextSpan(BaseModel):
    """Text span for evidence citation"""
    text: str = Field(..., description="Exact quoted text")
    start_char: int = Field(..., description="Start character offset")
    end_char: int = Field(..., description="End character offset")
    source_id: str = Field(..., description="Source document ID")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")


class Evidence(BaseModel):
    """Evidence with citation"""
    claim: str = Field(..., description="The claim being made")
    span: TextSpan = Field(..., description="Source text span")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    agent_type: AgentType
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Entity(BaseModel):
    """Extracted entity"""
    entity_type: str = Field(..., description="Entity type (e.g., PERSON, ORG, DATE, MONEY)")
    value: str = Field(..., description="Entity value")
    normalized_value: Optional[str] = Field(None, description="Normalized form")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source_span: Optional[TextSpan] = None


class Document(BaseModel):
    """Document representation"""
    id: str = Field(..., description="Unique document ID")
    content: str = Field(..., description="Document content")
    doc_type: str = Field(..., description="Document type (e.g., email, pdf, csv)")
    source: str = Field(..., description="Source system or path")
    entities: List[Entity] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class GraphNode(BaseModel):
    """Graph node representation"""
    id: str = Field(..., description="Node ID")
    label: str = Field(..., description="Node label/type")
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Graph edge representation"""
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict)


class Feature(BaseModel):
    """Feature for ML models"""
    name: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Feature value")
    feature_type: str = Field(..., description="Feature type (numeric, categorical, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Point-in-time")


class FeatureVector(BaseModel):
    """Collection of features for a single entity"""
    entity_id: str = Field(..., description="Entity being described")
    features: List[Feature]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
