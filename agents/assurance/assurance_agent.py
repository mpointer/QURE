"""
Assurance Agent

Uncertainty quantification, grounding validation, and confidence calibration.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from common.schemas import AssuranceRequest, AssuranceResponse

try:
    from substrate import get_evidence_tracker
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AssuranceAgent:
    """
    Assurance Agent for uncertainty quantification

    Responsibilities:
    - Compute uncertainty scores (epistemic, aleatoric)
    - Validate grounding (citation coverage, span accuracy)
    - Calibrate confidence scores
    - Detect hallucinations
    - Multi-agent consensus checking
    """

    def __init__(self):
        """Initialize Assurance Agent"""
        self.evidence_tracker = None

        if SUBSTRATE_AVAILABLE:
            try:
                self.evidence_tracker = get_evidence_tracker()
            except Exception as e:
                logger.warning(f"Could not initialize evidence tracker: {e}. Grounding validation disabled.")

        logger.info("✅ Assurance Agent initialized")

    def evaluate(
        self,
        request: AssuranceRequest,
    ) -> AssuranceResponse:
        """
        Evaluate assurance for a case

        Args:
            request: AssuranceRequest with agent outputs and evidence

        Returns:
            AssuranceResponse with uncertainty and grounding scores
        """
        case_id = request.case_id
        agent_outputs = request.agent_outputs
        source_documents = request.source_documents

        try:
            # 1. Uncertainty quantification
            uncertainty_score = self._compute_uncertainty(agent_outputs)

            # 2. Grounding validation
            grounding_score = self._validate_grounding(
                case_id, source_documents
            )

            # 3. Confidence calibration
            calibrated_confidence = self._calibrate_confidence(
                agent_outputs, uncertainty_score, grounding_score
            )

            # 4. Consensus checking
            consensus_score = self._check_consensus(agent_outputs)

            # 5. Hallucination detection
            hallucination_detected = self._detect_hallucination(
                agent_outputs, grounding_score
            )

            # Overall assurance score (weighted combination)
            assurance_score = (
                0.30 * (1.0 - uncertainty_score) +  # Lower uncertainty = higher assurance
                0.30 * grounding_score +
                0.20 * calibrated_confidence +
                0.20 * consensus_score
            )

            # Build explanation
            explanation_parts = [
                f"Uncertainty: {uncertainty_score:.2%}",
                f"Grounding: {grounding_score:.2%}",
                f"Calibrated Confidence: {calibrated_confidence:.2%}",
                f"Consensus: {consensus_score:.2%}",
                f"Overall Assurance: {assurance_score:.2%}",
            ]

            if hallucination_detected:
                explanation_parts.append("⚠️ Potential hallucination detected")

            explanation = "\n".join(explanation_parts)

            logger.info(
                f"Assurance evaluation for case {case_id}: "
                f"assurance={assurance_score:.3f}, uncertainty={uncertainty_score:.3f}, "
                f"grounding={grounding_score:.3f}"
            )

            return AssuranceResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                uncertainty_score=uncertainty_score,
                grounding_score=grounding_score,
                calibrated_confidence=calibrated_confidence,
                consensus_score=consensus_score,
                assurance_score=assurance_score,
                hallucination_detected=hallucination_detected,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Assurance evaluation failed for case {case_id}: {e}")
            return AssuranceResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                uncertainty_score=1.0,
                grounding_score=0.0,
                calibrated_confidence=0.0,
                consensus_score=0.0,
                assurance_score=0.0,
                hallucination_detected=True,
                explanation=f"Assurance error: {str(e)}",
            )

    def _compute_uncertainty(
        self,
        agent_outputs: List[Dict[str, Any]],
    ) -> float:
        """
        Compute uncertainty score across agent outputs

        Args:
            agent_outputs: List of agent output dicts with 'confidence' field

        Returns:
            Uncertainty score (0.0 = certain, 1.0 = maximum uncertainty)
        """
        if not agent_outputs:
            return 1.0  # Maximum uncertainty if no outputs

        confidences = []
        for output in agent_outputs:
            conf = output.get("confidence", 0.5)
            confidences.append(conf)

        if not confidences:
            return 1.0

        # Epistemic uncertainty: variance in confidence scores
        epistemic = float(np.var(confidences))

        # Aleatoric uncertainty: average of (1 - confidence)
        aleatoric = float(np.mean([1.0 - c for c in confidences]))

        # Combined uncertainty (equal weighting)
        uncertainty = 0.5 * epistemic + 0.5 * aleatoric

        return min(1.0, uncertainty)

    def _validate_grounding(
        self,
        case_id: str,
        source_documents: Dict[str, str],
    ) -> float:
        """
        Validate grounding using evidence tracker

        Args:
            case_id: Case ID
            source_documents: Map of document_id -> content

        Returns:
            Grounding score (0.0 = no grounding, 1.0 = fully grounded)
        """
        try:
            validation_result = self.evidence_tracker.validate_evidence(
                case_id=case_id,
                source_documents=source_documents,
            )

            grounding_score = validation_result["grounding_rate"]
            return grounding_score

        except Exception as e:
            logger.warning(f"Grounding validation failed: {e}")
            return 0.0

    def _calibrate_confidence(
        self,
        agent_outputs: List[Dict[str, Any]],
        uncertainty_score: float,
        grounding_score: float,
    ) -> float:
        """
        Calibrate confidence scores

        Args:
            agent_outputs: Agent outputs
            uncertainty_score: Computed uncertainty
            grounding_score: Grounding validation score

        Returns:
            Calibrated confidence score
        """
        if not agent_outputs:
            return 0.0

        # Get raw confidence scores
        confidences = [
            output.get("confidence", 0.5)
            for output in agent_outputs
        ]

        if not confidences:
            return 0.0

        # Average confidence
        avg_confidence = float(np.mean(confidences))

        # Calibration: adjust based on uncertainty and grounding
        calibrated = (
            avg_confidence *
            (1.0 - uncertainty_score) *  # Penalize high uncertainty
            grounding_score  # Penalize poor grounding
        )

        return min(1.0, calibrated)

    def _check_consensus(
        self,
        agent_outputs: List[Dict[str, Any]],
    ) -> float:
        """
        Check consensus across agent outputs

        Args:
            agent_outputs: Agent outputs with predictions/answers

        Returns:
            Consensus score (0.0 = no consensus, 1.0 = full consensus)
        """
        if len(agent_outputs) < 2:
            return 1.0  # Single agent = full consensus by definition

        # Extract predictions/answers
        predictions = []
        for output in agent_outputs:
            pred = output.get("prediction") or output.get("answer")
            if pred is not None:
                predictions.append(pred)

        if not predictions:
            return 0.0

        # Count unique predictions
        unique_predictions = len(set(str(p) for p in predictions))

        # Consensus = 1.0 if all agree, decreases with more disagreement
        consensus = 1.0 - ((unique_predictions - 1) / len(predictions))

        return max(0.0, consensus)

    def _detect_hallucination(
        self,
        agent_outputs: List[Dict[str, Any]],
        grounding_score: float,
    ) -> bool:
        """
        Detect potential hallucinations

        Args:
            agent_outputs: Agent outputs
            grounding_score: Grounding validation score

        Returns:
            True if hallucination detected
        """
        # Simple heuristics for hallucination detection:
        # 1. High confidence but low grounding
        # 2. GenAI output with no citations

        for output in agent_outputs:
            agent_type = output.get("from_agent", "")
            confidence = output.get("confidence", 0.0)
            citations = output.get("citations", [])

            # GenAI output without citations is suspicious
            if "genai" in agent_type.lower() and not citations:
                return True

            # High confidence but low grounding
            if confidence > 0.8 and grounding_score < 0.3:
                return True

        return False

    def quantify_epistemic_uncertainty(
        self,
        predictions: List[float],
    ) -> float:
        """
        Quantify epistemic uncertainty (model uncertainty)

        Args:
            predictions: List of prediction values from multiple models/runs

        Returns:
            Epistemic uncertainty score
        """
        if not predictions:
            return 1.0

        # Use variance as proxy for epistemic uncertainty
        return float(np.var(predictions))

    def quantify_aleatoric_uncertainty(
        self,
        probabilities: List[float],
    ) -> float:
        """
        Quantify aleatoric uncertainty (data uncertainty)

        Args:
            probabilities: Predicted probabilities

        Returns:
            Aleatoric uncertainty score
        """
        if not probabilities:
            return 1.0

        # Use entropy as proxy for aleatoric uncertainty
        # H(p) = -sum(p * log(p))
        probs = np.array(probabilities)
        probs = probs / probs.sum()  # Normalize

        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize entropy to [0, 1]
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def compute_confidence_intervals(
        self,
        predictions: List[float],
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute confidence intervals for predictions

        Args:
            predictions: List of prediction values
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dict with 'mean', 'lower', 'upper'
        """
        if not predictions:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

        preds = np.array(predictions)
        mean = float(np.mean(preds))
        std = float(np.std(preds))

        # Compute z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        margin = z_score * std / np.sqrt(len(preds))

        return {
            "mean": mean,
            "lower": mean - margin,
            "upper": mean + margin,
        }

    def validate_model_calibration(
        self,
        predicted_probs: List[float],
        true_labels: List[int],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Validate model calibration using reliability diagrams

        Args:
            predicted_probs: Predicted probabilities
            true_labels: True binary labels (0 or 1)
            n_bins: Number of bins for calibration curve

        Returns:
            Dict with calibration metrics
        """
        if len(predicted_probs) != len(true_labels):
            raise ValueError("Length mismatch between predictions and labels")

        probs = np.array(predicted_probs)
        labels = np.array(true_labels)

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1

        # Compute calibration for each bin
        bin_true_probs = []
        bin_pred_probs = []

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_pred_probs.append(probs[mask].mean())
                bin_true_probs.append(labels[mask].mean())

        # Compute Expected Calibration Error (ECE)
        ece = 0.0
        for pred_prob, true_prob in zip(bin_pred_probs, bin_true_probs):
            ece += abs(pred_prob - true_prob)
        ece /= len(bin_pred_probs) if bin_pred_probs else 1.0

        return {
            "expected_calibration_error": float(ece),
            "bin_predicted_probs": bin_pred_probs,
            "bin_true_probs": bin_true_probs,
            "is_well_calibrated": ece < 0.1,  # Threshold: ECE < 0.1
        }

    def ensemble_predictions(
        self,
        predictions: List[Dict[str, Any]],
        method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """
        Ensemble multiple predictions

        Args:
            predictions: List of prediction dicts with 'value' and 'confidence'
            method: "weighted_average", "majority_vote", or "max_confidence"

        Returns:
            Dict with ensembled prediction and confidence
        """
        if not predictions:
            return {"value": None, "confidence": 0.0}

        if method == "weighted_average":
            # Weight by confidence
            total_weight = sum(p["confidence"] for p in predictions)
            if total_weight == 0:
                return {"value": predictions[0]["value"], "confidence": 0.0}

            weighted_sum = sum(
                p["value"] * p["confidence"]
                for p in predictions
                if isinstance(p["value"], (int, float))
            )
            ensembled_value = weighted_sum / total_weight
            ensembled_confidence = total_weight / len(predictions)

            return {
                "value": ensembled_value,
                "confidence": ensembled_confidence,
            }

        elif method == "majority_vote":
            # Count votes
            from collections import Counter
            values = [p["value"] for p in predictions]
            counter = Counter(values)
            most_common = counter.most_common(1)[0]

            ensembled_value = most_common[0]
            vote_count = most_common[1]
            ensembled_confidence = vote_count / len(predictions)

            return {
                "value": ensembled_value,
                "confidence": ensembled_confidence,
            }

        elif method == "max_confidence":
            # Pick prediction with highest confidence
            best = max(predictions, key=lambda p: p["confidence"])
            return {
                "value": best["value"],
                "confidence": best["confidence"],
            }

        else:
            raise ValueError(f"Unknown ensemble method: {method}")


# Singleton instance
_assurance_agent: Optional[AssuranceAgent] = None


def get_assurance_agent() -> AssuranceAgent:
    """
    Get or create singleton AssuranceAgent instance

    Returns:
        AssuranceAgent instance
    """
    global _assurance_agent

    if _assurance_agent is None:
        _assurance_agent = AssuranceAgent()

    return _assurance_agent
