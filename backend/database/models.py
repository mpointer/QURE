"""
Database models for QURE reconciliation system

Stores:
- Reconciliation cases and their results
- Agent decisions with confidence scores
- Audit trail of all agent reasoning
"""

from datetime import datetime
from typing import Optional
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, JSON,
    ForeignKey, Enum, Boolean, Index
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class ReconciliationStatus(str, PyEnum):
    """Status of reconciliation case"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionType(str, PyEnum):
    """Types of decisions agents can make"""
    AUTO_RESOLVE = "auto_resolve"
    HITL_REVIEW = "hitl_review"
    REJECT = "reject"
    ESCALATE = "escalate"
    REQUEST_EVIDENCE = "request_evidence"


class ReconciliationCase(Base):
    """Main reconciliation case"""
    __tablename__ = "reconciliation_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(100), unique=True, nullable=False, index=True)

    # Vertical/domain
    vertical = Column(String(50), nullable=False, index=True)

    # Transaction data
    data1 = Column(JSON, nullable=False)  # GL/Source transaction
    data2 = Column(JSON, nullable=False)  # Bank/Target transaction

    # Results
    status = Column(Enum(ReconciliationStatus), default=ReconciliationStatus.PENDING, index=True)
    decision = Column(Enum(DecisionType), nullable=True)
    confidence_score = Column(Float, nullable=True)
    match_score = Column(Float, nullable=True)

    # Agent reasoning
    agent_reasoning = Column(JSON, nullable=True)  # Full QRU breakdown

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Processing metadata
    llm_provider = Column(String(50), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    retry_count = Column(Integer, default=0)

    # Relationships
    agent_decisions = relationship("AgentDecision", back_populates="case", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('ix_case_status_created', 'status', 'created_at'),
        Index('ix_case_vertical_decision', 'vertical', 'decision'),
    )

    def __repr__(self):
        return f"<ReconciliationCase(case_id={self.case_id}, status={self.status}, decision={self.decision})>"


class AgentDecision(Base):
    """Individual agent decisions within a case (QRU pattern)"""
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("reconciliation_cases.id"), nullable=False)

    # Agent info
    agent_name = Column(String(100), nullable=False)  # e.g., "Q", "R", "U"
    agent_role = Column(String(100), nullable=False)  # e.g., "question", "reason", "update"

    # Decision
    decision = Column(Enum(DecisionType), nullable=True)
    confidence_score = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)

    # Evidence and rules applied
    rules_applied = Column(JSON, nullable=True)  # List of business rules used
    evidence = Column(JSON, nullable=True)  # Supporting data points

    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    execution_time_ms = Column(Integer, nullable=True)

    # Relationships
    case = relationship("ReconciliationCase", back_populates="agent_decisions")

    # Indexes
    __table_args__ = (
        Index('ix_agent_case_role', 'case_id', 'agent_role'),
    )

    def __repr__(self):
        return f"<AgentDecision(agent={self.agent_name}, confidence={self.confidence_score})>"


class AuditLog(Base):
    """Audit trail of all system actions"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    # What happened
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(100), nullable=True)

    # Details
    details = Column(JSON, nullable=True)
    outcome = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)

    # Context
    user_id = Column(String(100), nullable=True)  # For future auth
    ip_address = Column(String(45), nullable=True)

    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    def __repr__(self):
        return f"<AuditLog(action={self.action}, entity={self.entity_type}, timestamp={self.timestamp})>"


class BusinessRule(Base):
    """Business rules that agents can apply"""
    __tablename__ = "business_rules"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(String(100), unique=True, nullable=False, index=True)

    # Rule definition
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    vertical = Column(String(50), nullable=False, index=True)

    # Rule logic
    rule_type = Column(String(50), nullable=False)  # e.g., "amount_threshold", "date_range"
    conditions = Column(JSON, nullable=False)
    actions = Column(JSON, nullable=False)

    # Configuration
    priority = Column(Integer, default=100)
    is_active = Column(Boolean, default=True, index=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(100), nullable=True)

    __table_args__ = (
        Index('ix_rule_vertical_active', 'vertical', 'is_active'),
    )

    def __repr__(self):
        return f"<BusinessRule(rule_id={self.rule_id}, name={self.name}, active={self.is_active})>"
