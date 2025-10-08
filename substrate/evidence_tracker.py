"""
Evidence Tracker

Links claims/decisions to source text spans for auditability.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from common.schemas import AgentType, Evidence, TextSpan

logger = logging.getLogger(__name__)


class EvidenceTracker:
    """
    Track evidence linking for claims and decisions

    Features:
    - Link claims to source documents with span offsets
    - Track evidence provenance (which agent generated it)
    - Validate that spans exist in source documents
    - Generate audit trails
    - Compute grounding rate (% claims with valid evidence)
    """

    def __init__(self):
        """Initialize evidence tracker"""
        # In-memory store for now; could be backed by database
        self._evidence_store: Dict[str, List[Evidence]] = {}
        logger.info("✅ Evidence tracker initialized")

    def add_evidence(
        self,
        case_id: str,
        evidence: Evidence,
    ) -> None:
        """
        Add evidence for a case

        Args:
            case_id: Case ID
            evidence: Evidence object with claim and citation
        """
        if case_id not in self._evidence_store:
            self._evidence_store[case_id] = []

        self._evidence_store[case_id].append(evidence)
        logger.debug(f"Added evidence for case {case_id} from {evidence.agent_type}")

    def add_evidence_batch(
        self,
        case_id: str,
        evidence_list: List[Evidence],
    ) -> None:
        """
        Add multiple evidence items in batch

        Args:
            case_id: Case ID
            evidence_list: List of Evidence objects
        """
        if case_id not in self._evidence_store:
            self._evidence_store[case_id] = []

        self._evidence_store[case_id].extend(evidence_list)
        logger.info(f"Added {len(evidence_list)} evidence items for case {case_id}")

    def get_evidence(
        self,
        case_id: str,
        agent_type: Optional[AgentType] = None,
    ) -> List[Evidence]:
        """
        Get evidence for a case

        Args:
            case_id: Case ID
            agent_type: Optional filter by agent type

        Returns:
            List of Evidence objects
        """
        evidence_list = self._evidence_store.get(case_id, [])

        if agent_type:
            evidence_list = [e for e in evidence_list if e.agent_type == agent_type]

        return evidence_list

    def validate_span(
        self,
        span: TextSpan,
        source_document: str,
    ) -> bool:
        """
        Validate that a text span exists in the source document

        Args:
            span: TextSpan to validate
            source_document: Source document content

        Returns:
            True if span text matches document at given offsets
        """
        try:
            # Extract text at specified offsets
            extracted_text = source_document[span.start_char:span.end_char]

            # Check if it matches the claimed text
            if extracted_text.strip() == span.text.strip():
                return True

            # Allow for minor whitespace differences
            if extracted_text.replace(" ", "") == span.text.replace(" ", ""):
                return True

            logger.warning(
                f"Span validation failed for source {span.source_id}: "
                f"Expected '{span.text}' but found '{extracted_text}'"
            )
            return False

        except IndexError:
            logger.error(f"Span offsets out of bounds for source {span.source_id}")
            return False

    def validate_evidence(
        self,
        case_id: str,
        source_documents: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Validate all evidence for a case against source documents

        Args:
            case_id: Case ID
            source_documents: Dict mapping source_id to document content

        Returns:
            Dict with validation stats (grounding_rate, valid_count, total_count)
        """
        evidence_list = self.get_evidence(case_id)

        if not evidence_list:
            return {
                "grounding_rate": 0.0,
                "valid_count": 0,
                "total_count": 0,
            }

        valid_count = 0
        total_count = len(evidence_list)

        for evidence in evidence_list:
            source_doc = source_documents.get(evidence.span.source_id)

            if source_doc and self.validate_span(evidence.span, source_doc):
                valid_count += 1

        grounding_rate = valid_count / total_count if total_count > 0 else 0.0

        logger.info(
            f"Evidence validation for case {case_id}: "
            f"{valid_count}/{total_count} valid ({grounding_rate:.1%})"
        )

        return {
            "grounding_rate": grounding_rate,
            "valid_count": valid_count,
            "total_count": total_count,
        }

    def get_evidence_by_agent(
        self,
        case_id: str,
    ) -> Dict[AgentType, List[Evidence]]:
        """
        Group evidence by agent type

        Args:
            case_id: Case ID

        Returns:
            Dict mapping agent_type to list of Evidence
        """
        evidence_list = self.get_evidence(case_id)
        grouped: Dict[AgentType, List[Evidence]] = {}

        for evidence in evidence_list:
            if evidence.agent_type not in grouped:
                grouped[evidence.agent_type] = []
            grouped[evidence.agent_type].append(evidence)

        return grouped

    def generate_audit_trail(
        self,
        case_id: str,
    ) -> List[Dict]:
        """
        Generate audit trail for evidence

        Args:
            case_id: Case ID

        Returns:
            List of audit trail entries
        """
        evidence_list = self.get_evidence(case_id)

        audit_trail = []
        for evidence in evidence_list:
            entry = {
                "timestamp": evidence.timestamp.isoformat(),
                "agent": evidence.agent_type.value,
                "claim": evidence.claim,
                "source": evidence.span.source_id,
                "span_text": evidence.span.text,
                "span_offset": [evidence.span.start_char, evidence.span.end_char],
                "confidence": evidence.span.confidence,
                "relevance": evidence.relevance_score,
            }
            audit_trail.append(entry)

        return sorted(audit_trail, key=lambda x: x["timestamp"])

    def get_citation_summary(
        self,
        case_id: str,
    ) -> Dict:
        """
        Get summary of citations for a case

        Args:
            case_id: Case ID

        Returns:
            Summary dict with counts and stats
        """
        evidence_list = self.get_evidence(case_id)

        if not evidence_list:
            return {
                "total_claims": 0,
                "total_citations": 0,
                "avg_confidence": 0.0,
                "avg_relevance": 0.0,
                "sources": [],
                "agents": [],
            }

        sources = set(e.span.source_id for e in evidence_list)
        agents = set(e.agent_type for e in evidence_list)

        avg_confidence = sum(e.span.confidence for e in evidence_list) / len(evidence_list)
        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)

        return {
            "total_claims": len(evidence_list),
            "total_citations": len(evidence_list),  # 1:1 for now
            "avg_confidence": avg_confidence,
            "avg_relevance": avg_relevance,
            "sources": list(sources),
            "agents": [a.value for a in agents],
        }

    def clear_case(self, case_id: str) -> int:
        """
        Clear all evidence for a case

        Args:
            case_id: Case ID

        Returns:
            Number of evidence items cleared
        """
        if case_id in self._evidence_store:
            count = len(self._evidence_store[case_id])
            del self._evidence_store[case_id]
            logger.debug(f"Cleared {count} evidence items for case {case_id}")
            return count
        return 0

    def get_stats(self) -> Dict:
        """
        Get global evidence tracker statistics

        Returns:
            Dict with stats across all cases
        """
        total_cases = len(self._evidence_store)
        total_evidence = sum(len(ev) for ev in self._evidence_store.values())

        agent_counts = {}
        for evidence_list in self._evidence_store.values():
            for evidence in evidence_list:
                agent_type = evidence.agent_type.value
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1

        return {
            "total_cases": total_cases,
            "total_evidence": total_evidence,
            "evidence_by_agent": agent_counts,
        }


# Singleton instance
_evidence_tracker: Optional[EvidenceTracker] = None


def get_evidence_tracker() -> EvidenceTracker:
    """
    Get or create singleton EvidenceTracker instance

    Returns:
        EvidenceTracker instance
    """
    global _evidence_tracker

    if _evidence_tracker is None:
        _evidence_tracker = EvidenceTracker()

    return _evidence_tracker
