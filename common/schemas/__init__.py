"""
QURE Common Schemas
"""

from .base import (
    AgentType,
    BaseEntity,
    CaseStatus,
    DecisionType,
    Document,
    Entity,
    Evidence,
    Feature,
    FeatureVector,
    GraphEdge,
    GraphNode,
    TextSpan,
    VerticalType,
)
from .messages import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    AlgorithmRequest,
    AlgorithmResponse,
    AssuranceRequest,
    AssuranceResponse,
    CaseCompletionMessage,
    CaseRequest,
    DataProcessingRequest,
    DataProcessingResponse,
    FeedbackMessage,
    GenAIRequest,
    GenAIResponse,
    MLPredictionRequest,
    MLPredictionResponse,
    PolicyRequest,
    PolicyResponse,
    RetrievalRequest,
    RetrievalResponse,
    RulesEvaluationRequest,
    RulesEvaluationResponse,
)

# Aliases for backward compatibility
GenAIReasoningRequest = GenAIRequest
GenAIReasoningResponse = GenAIResponse
PolicyDecisionRequest = PolicyRequest
PolicyDecisionResponse = PolicyResponse

__all__ = [
    # Base types
    "AgentType",
    "BaseEntity",
    "CaseStatus",
    "DecisionType",
    "Document",
    "Entity",
    "Evidence",
    "Feature",
    "FeatureVector",
    "GraphEdge",
    "GraphNode",
    "TextSpan",
    "VerticalType",
    # Messages
    "ActionRequest",
    "ActionResponse",
    "AgentMessage",
    "AlgorithmRequest",
    "AlgorithmResponse",
    "AssuranceRequest",
    "AssuranceResponse",
    "CaseCompletionMessage",
    "CaseRequest",
    "DataProcessingRequest",
    "DataProcessingResponse",
    "FeedbackMessage",
    "GenAIRequest",
    "GenAIResponse",
    "GenAIReasoningRequest",
    "GenAIReasoningResponse",
    "MLPredictionRequest",
    "MLPredictionResponse",
    "PolicyRequest",
    "PolicyResponse",
    "PolicyDecisionRequest",
    "PolicyDecisionResponse",
    "RetrievalRequest",
    "RetrievalResponse",
    "RulesEvaluationRequest",
    "RulesEvaluationResponse",
]
