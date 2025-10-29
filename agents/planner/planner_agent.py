"""
Planner QRU (Planning Agent) - Business-Aware Meta-Orchestrator

The Planner QRU has a strong business identity and leads efforts to solve specific
classes of business problems across verticals. It understands:
- Finance: Reconciliation, GL/Bank matching, SOX compliance
- Healthcare: Prior authorization, medical necessity, claims processing
- Insurance: Subrogation recovery, liability assessment, claims validation
- Retail: Inventory reconciliation, shrinkage analysis, returns processing
- Manufacturing: Purchase order matching, receiving discrepancies, quality issues

The Planner dynamically determines which specialized QRUs should be invoked based on
the business problem class, data quality, and required reasoning depth.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Complexity(str, Enum):
    """Case complexity levels"""
    SIMPLE = "Simple"
    MODERATE = "Moderate"
    COMPLEX = "Complex"
    HIGHLY_COMPLEX = "Highly Complex"


class DataQuality(str, Enum):
    """Data quality assessment"""
    COMPLETE = "Complete"
    INCOMPLETE = "Incomplete"
    AMBIGUOUS = "Ambiguous"
    CONFLICTING = "Conflicting"


class ReasoningType(str, Enum):
    """Required reasoning approach"""
    RULES_ONLY = "Rules-only"
    ALGORITHMIC = "Algorithmic"
    ML_REQUIRED = "ML Required"
    CONTEXTUAL_LLM = "Contextual (LLM)"


class BusinessProblemClass(str, Enum):
    """Business problem classifications by vertical"""
    # Finance
    FINANCE_RECONCILIATION = "Finance: GL/Bank Reconciliation"
    FINANCE_SOX_COMPLIANCE = "Finance: SOX Compliance Review"
    FINANCE_INTERCOMPANY = "Finance: Intercompany Matching"

    # Healthcare
    HEALTHCARE_PRIOR_AUTH = "Healthcare: Prior Authorization"
    HEALTHCARE_MEDICAL_NECESSITY = "Healthcare: Medical Necessity"
    HEALTHCARE_CLAIMS_ADJUDICATION = "Healthcare: Claims Adjudication"

    # Insurance
    INSURANCE_SUBROGATION = "Insurance: Subrogation Recovery"
    INSURANCE_LIABILITY = "Insurance: Liability Assessment"
    INSURANCE_FRAUD = "Insurance: Fraud Detection"

    # Retail
    RETAIL_INVENTORY_RECON = "Retail: Inventory Reconciliation"
    RETAIL_SHRINKAGE = "Retail: Shrinkage Analysis"
    RETAIL_RETURNS = "Retail: Returns Processing"

    # Manufacturing
    MFG_PO_MATCHING = "Manufacturing: PO Matching"
    MFG_RECEIVING = "Manufacturing: Receiving Discrepancies"
    MFG_QUALITY = "Manufacturing: Quality Issues"


@dataclass
class Classification:
    """Case classification result"""
    complexity: Complexity
    data_quality: DataQuality
    reasoning_type: ReasoningType
    business_problem: BusinessProblemClass
    confidence: float
    complexity_score: int
    missing_fields: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


@dataclass
class QRUSelection:
    """Selected QRU with metadata"""
    qru_name: str
    priority: int
    required: bool
    reason: str
    estimated_cost: float
    estimated_time_seconds: float


@dataclass
class ExecutionPlan:
    """Complete execution plan for a case"""
    plan_id: str
    case_id: str
    timestamp: datetime
    classification: Classification
    selected_qrus: List[QRUSelection]
    skipped_qrus: List[str]
    estimated_total_cost: float
    estimated_total_time_seconds: float
    reasoning: str
    business_context: str


class PlannerQRU:
    """
    Business-Aware Planner QRU

    The Planner QRU acts as a business consultant that understands the domain-specific
    nuances of each vertical and can intelligently plan the resolution strategy.
    """

    # QRU cost model ($/invocation)
    QRU_COSTS = {
        "Retriever": 0.001,
        "Data": 0.002,
        "Rules": 0.0005,
        "Algorithm": 0.001,
        "ML Model": 0.01,
        "GenAI": 0.05,
        "Assurance": 0.01,
        "Orchestration": 0.002,
        "Policy": 0.005,
        "Action": 0.001,
        "Learning": 0.003
    }

    # QRU time model (seconds)
    QRU_TIMES = {
        "Retriever": 0.5,
        "Data": 1.0,
        "Rules": 0.2,
        "Algorithm": 0.5,
        "ML Model": 2.0,
        "GenAI": 5.0,
        "Assurance": 1.5,
        "Orchestration": 0.5,
        "Policy": 1.0,
        "Action": 0.5,
        "Learning": 1.0
    }

    def __init__(self, knowledge_substrate=None):
        """
        Initialize Planner QRU

        Args:
            knowledge_substrate: Optional knowledge substrate for historical lookups
        """
        self.knowledge = knowledge_substrate
        self.plan_counter = 0
        logger.info("Planner QRU initialized with business problem awareness")

    def analyze_case(self, case_data: dict, vertical: str = "Finance") -> ExecutionPlan:
        """
        Main entry point: Analyze case and generate execution plan

        Args:
            case_data: Case data dictionary
            vertical: Business vertical (Finance, Healthcare, Insurance, Retail, Manufacturing)

        Returns:
            ExecutionPlan with selected QRUs and reasoning
        """
        self.plan_counter += 1
        plan_id = f"PLAN_{datetime.now().strftime('%Y%m%d')}_{self.plan_counter:04d}"

        logger.info(f"Planner QRU analyzing case {case_data.get('case_id', 'UNKNOWN')} for {vertical}")

        # Step 1: Classify the business problem
        classification = self._classify_case(case_data, vertical)

        # Step 2: Generate business context understanding
        business_context = self._generate_business_context(classification, case_data, vertical)

        # Step 3: Select QRUs based on business problem and classification
        selected_qrus = self._select_qrus(classification, case_data, vertical)

        # Step 4: Determine skipped QRUs
        all_qrus = set(self.QRU_COSTS.keys()) - {"Policy", "Action"}  # These always run
        selected_qru_names = {qru.qru_name for qru in selected_qrus if qru.qru_name not in ["Policy", "Action"]}
        skipped_qrus = list(all_qrus - selected_qru_names)

        # Step 5: Calculate cost and time estimates
        estimated_cost = sum(qru.estimated_cost for qru in selected_qrus)
        estimated_time = sum(qru.estimated_time_seconds for qru in selected_qrus)

        # Step 6: Generate reasoning explanation
        reasoning = self._generate_reasoning(classification, selected_qrus, skipped_qrus, vertical)

        plan = ExecutionPlan(
            plan_id=plan_id,
            case_id=case_data.get("case_id", "UNKNOWN"),
            timestamp=datetime.now(),
            classification=classification,
            selected_qrus=selected_qrus,
            skipped_qrus=skipped_qrus,
            estimated_total_cost=estimated_cost,
            estimated_total_time_seconds=estimated_time,
            reasoning=reasoning,
            business_context=business_context
        )

        logger.info(f"Plan {plan_id}: {len(selected_qrus)} QRUs selected, {len(skipped_qrus)} skipped. "
                   f"Est. cost: ${estimated_cost:.4f}, time: {estimated_time:.1f}s")

        return plan

    def _classify_case(self, case_data: dict, vertical: str) -> Classification:
        """
        Classify case by complexity, data quality, business problem, etc.

        This method embodies the Planner's business expertise - it understands
        what makes a Finance reconciliation complex vs. a Healthcare prior auth.
        """
        # Step 1: Assess data completeness
        required_fields = self._get_required_fields(vertical)
        missing_fields = [f for f in required_fields if not case_data.get(f)]

        # Step 2: Detect data conflicts
        conflicts = self._detect_conflicts(case_data, vertical)

        # Step 3: Calculate complexity score (0-11 points)
        complexity_score = 0

        # Factor 1: Missing fields (0-3 points)
        if len(missing_fields) == 0:
            complexity_score += 0
        elif len(missing_fields) <= 2:
            complexity_score += 1
        elif len(missing_fields) <= 5:
            complexity_score += 2
        else:
            complexity_score += 3

        # Factor 2: Data conflicts (0-3 points)
        complexity_score += min(len(conflicts), 3)

        # Factor 3: Historical precedent (0-2 points)
        if self.knowledge:
            similar_count = self._count_similar_cases(case_data, vertical)
            if similar_count == 0:
                complexity_score += 2  # Never seen before
            elif similar_count < 5:
                complexity_score += 1  # Few precedents

        # Factor 4: Business-specific complexity (0-3 points)
        business_complexity = self._assess_business_complexity(case_data, vertical)
        complexity_score += business_complexity

        # Map score to complexity level
        if complexity_score <= 2:
            complexity = Complexity.SIMPLE
        elif complexity_score <= 5:
            complexity = Complexity.MODERATE
        elif complexity_score <= 8:
            complexity = Complexity.COMPLEX
        else:
            complexity = Complexity.HIGHLY_COMPLEX

        # Step 4: Assess data quality
        if len(conflicts) > 0:
            data_quality = DataQuality.CONFLICTING
        elif len(missing_fields) > 0 and self._has_ambiguous_values(case_data):
            data_quality = DataQuality.AMBIGUOUS
        elif len(missing_fields) > 0:
            data_quality = DataQuality.INCOMPLETE
        else:
            data_quality = DataQuality.COMPLETE

        # Step 5: Determine reasoning type needed
        reasoning_type = self._determine_reasoning_type(complexity, data_quality)

        # Step 6: Classify business problem
        business_problem = self._classify_business_problem(case_data, vertical)

        # Step 7: Calculate confidence in classification
        confidence = self._calculate_classification_confidence(
            complexity_score, len(missing_fields), len(conflicts)
        )

        return Classification(
            complexity=complexity,
            data_quality=data_quality,
            reasoning_type=reasoning_type,
            business_problem=business_problem,
            confidence=confidence,
            complexity_score=complexity_score,
            missing_fields=missing_fields,
            conflicts=conflicts
        )

    def _get_required_fields(self, vertical: str) -> List[str]:
        """Get required fields for each vertical's business problems"""
        field_map = {
            "Finance": ["amount", "date", "description", "account"],
            "Healthcare": ["patient_id", "procedure_code", "diagnosis", "provider"],
            "Insurance": ["claim_id", "incident_date", "loss_amount", "liability"],
            "Retail": ["item_id", "quantity", "location", "timestamp"],
            "Manufacturing": ["po_number", "part_number", "quantity", "delivery_date"]
        }
        return field_map.get(vertical, ["id", "date", "amount"])

    def _detect_conflicts(self, case_data: dict, vertical: str) -> List[str]:
        """Detect data conflicts specific to business vertical"""
        conflicts = []

        if vertical == "Finance":
            # Amount mismatch
            gl_amount = case_data.get("gl_amount")
            bank_amount = case_data.get("bank_amount")
            if gl_amount and bank_amount and abs(float(gl_amount) - float(bank_amount)) > 0.01:
                conflicts.append("Amount mismatch between GL and Bank")

            # Date variance > 7 days
            gl_date = case_data.get("gl_date")
            bank_date = case_data.get("bank_date")
            if gl_date and bank_date:
                # Simplified check - would parse dates in production
                if gl_date != bank_date:
                    conflicts.append("Date mismatch between GL and Bank")

        return conflicts

    def _count_similar_cases(self, case_data: dict, vertical: str) -> int:
        """Count similar historical cases (stub for knowledge substrate integration)"""
        # TODO: Query knowledge substrate for similar cases
        # For now, return 0 to indicate no historical data
        return 0

    def _assess_business_complexity(self, case_data: dict, vertical: str) -> int:
        """
        Assess business-specific complexity factors

        This is where the Planner's business identity shines - it understands
        what makes a case complex from a business perspective.
        """
        complexity_points = 0

        if vertical == "Finance":
            # High-value transactions need more scrutiny
            amount = float(case_data.get("gl_amount", 0) or case_data.get("bank_amount", 0) or 0)
            if amount > 100000:
                complexity_points += 2  # SOX compliance threshold
            elif amount > 10000:
                complexity_points += 1

            # Foreign currency adds complexity
            if case_data.get("currency") and case_data.get("currency") != "USD":
                complexity_points += 1

        elif vertical == "Healthcare":
            # Experimental procedures need more review
            if case_data.get("experimental", False):
                complexity_points += 2

            # Out-of-network adds complexity
            if not case_data.get("in_network", True):
                complexity_points += 1

        elif vertical == "Insurance":
            # Multi-party liability is complex
            if case_data.get("liable_parties", 1) > 1:
                complexity_points += 2

            # Large claims need more scrutiny
            loss_amount = float(case_data.get("loss_amount", 0) or 0)
            if loss_amount > 50000:
                complexity_points += 1

        return min(complexity_points, 3)  # Cap at 3

    def _has_ambiguous_values(self, case_data: dict) -> bool:
        """Check for ambiguous values that need interpretation"""
        # Look for partial matches, unclear descriptions, etc.
        description = case_data.get("description", "")
        if description and len(description.split()) < 3:
            return True  # Very short descriptions are ambiguous
        return False

    def _determine_reasoning_type(self, complexity: Complexity, data_quality: DataQuality) -> ReasoningType:
        """Determine what type of reasoning is required"""
        if complexity == Complexity.SIMPLE and data_quality == DataQuality.COMPLETE:
            return ReasoningType.RULES_ONLY
        elif complexity in [Complexity.SIMPLE, Complexity.MODERATE]:
            return ReasoningType.ALGORITHMIC
        elif complexity == Complexity.COMPLEX:
            return ReasoningType.ML_REQUIRED
        else:
            return ReasoningType.CONTEXTUAL_LLM

    def _classify_business_problem(self, case_data: dict, vertical: str) -> BusinessProblemClass:
        """Classify the specific business problem"""
        # Map vertical to primary business problem
        # In production, would analyze case_data to determine specific problem type
        problem_map = {
            "Finance": BusinessProblemClass.FINANCE_RECONCILIATION,
            "Healthcare": BusinessProblemClass.HEALTHCARE_PRIOR_AUTH,
            "Insurance": BusinessProblemClass.INSURANCE_SUBROGATION,
            "Retail": BusinessProblemClass.RETAIL_INVENTORY_RECON,
            "Manufacturing": BusinessProblemClass.MFG_PO_MATCHING
        }
        return problem_map.get(vertical, BusinessProblemClass.FINANCE_RECONCILIATION)

    def _calculate_classification_confidence(self, score: int, missing: int, conflicts: int) -> float:
        """Calculate confidence in the classification"""
        # High confidence when we have complete data
        if missing == 0 and conflicts == 0:
            return 0.95
        # Medium confidence with some missing data
        elif missing <= 2 and conflicts == 0:
            return 0.85
        # Low confidence with many missing fields or conflicts
        elif missing > 5 or conflicts > 2:
            return 0.60
        else:
            return 0.75

    def _generate_business_context(self, classification: Classification,
                                   case_data: dict, vertical: str) -> str:
        """
        Generate business context statement that shows the Planner's understanding
        of the business problem at hand.
        """
        problem = classification.business_problem.value
        complexity = classification.complexity.value

        context = f"Business Problem: {problem}\n"
        context += f"Complexity Level: {complexity}\n"
        context += f"Data Quality: {classification.data_quality.value}\n"

        # Add vertical-specific business insights
        if vertical == "Finance":
            amount = float(case_data.get("gl_amount", 0) or case_data.get("bank_amount", 0) or 0)
            context += f"Transaction Amount: ${amount:,.2f}\n"
            if amount > 100000:
                context += "⚠️ SOX Compliance Review Required (>$100K)\n"

        elif vertical == "Healthcare":
            context += f"Clinical Review: {classification.reasoning_type.value}\n"

        elif vertical == "Insurance":
            loss = float(case_data.get("loss_amount", 0) or 0)
            context += f"Loss Amount: ${loss:,.2f}\n"

        return context

    def _select_qrus(self, classification: Classification,
                    case_data: dict, vertical: str) -> List[QRUSelection]:
        """
        Select which QRUs to invoke based on business problem and classification

        This is the core intelligence of the Planner - it knows which specialists
        (QRUs) are needed for each business problem.
        """
        selected = []

        # Always required: Retriever and Data (data ingestion pipeline)
        selected.append(QRUSelection(
            qru_name="Retriever",
            priority=1,
            required=True,
            reason="Fetch source data from systems",
            estimated_cost=self.QRU_COSTS["Retriever"],
            estimated_time_seconds=self.QRU_TIMES["Retriever"]
        ))

        selected.append(QRUSelection(
            qru_name="Data",
            priority=2,
            required=True,
            reason="Entity extraction and normalization",
            estimated_cost=self.QRU_COSTS["Data"],
            estimated_time_seconds=self.QRU_TIMES["Data"]
        ))

        # Rules QRU for compliance and business rules
        if classification.complexity in [Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX]:
            reason = "Business rule validation"
            if vertical == "Finance":
                reason = "SOX compliance and accounting rules"
            elif vertical == "Healthcare":
                reason = "Medical policy and coverage rules"
            elif vertical == "Insurance":
                reason = "Policy terms and liability rules"

            selected.append(QRUSelection(
                qru_name="Rules",
                priority=3,
                required=True,
                reason=reason,
                estimated_cost=self.QRU_COSTS["Rules"],
                estimated_time_seconds=self.QRU_TIMES["Rules"]
            ))

        # Algorithm QRU for deterministic matching
        if classification.complexity in [Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX]:
            reason = "Fuzzy matching algorithms"
            if vertical == "Finance":
                reason = "GL/Bank amount and date matching"
            elif vertical == "Retail":
                reason = "POS and physical inventory reconciliation"
            elif vertical == "Manufacturing":
                reason = "PO and receiving document matching"

            selected.append(QRUSelection(
                qru_name="Algorithm",
                priority=4,
                required=True,
                reason=reason,
                estimated_cost=self.QRU_COSTS["Algorithm"],
                estimated_time_seconds=self.QRU_TIMES["Algorithm"]
            ))

        # ML Model QRU for pattern recognition
        if classification.complexity in [Complexity.MODERATE, Complexity.COMPLEX, Complexity.HIGHLY_COMPLEX]:
            reason = "ML-based pattern matching and prediction"
            if vertical == "Healthcare":
                reason = "Medical necessity prediction from historical approvals"
            elif vertical == "Insurance":
                reason = "Fraud detection and subrogation likelihood"

            selected.append(QRUSelection(
                qru_name="ML Model",
                priority=5,
                required=True,
                reason=reason,
                estimated_cost=self.QRU_COSTS["ML Model"],
                estimated_time_seconds=self.QRU_TIMES["ML Model"]
            ))

        # GenAI QRU for contextual reasoning
        if classification.data_quality in [DataQuality.AMBIGUOUS, DataQuality.CONFLICTING] or \
           classification.complexity in [Complexity.COMPLEX, Complexity.HIGHLY_COMPLEX]:
            reason = "LLM-based contextual reasoning"
            if vertical == "Finance":
                reason = "Interpret ambiguous transaction descriptions"
            elif vertical == "Healthcare":
                reason = "Assess medical necessity from clinical notes"
            elif vertical == "Insurance":
                reason = "Evaluate liability from incident narratives"

            selected.append(QRUSelection(
                qru_name="GenAI",
                priority=6,
                required=True,
                reason=reason,
                estimated_cost=self.QRU_COSTS["GenAI"],
                estimated_time_seconds=self.QRU_TIMES["GenAI"]
            ))

            # If using GenAI, also need Assurance for hallucination detection
            selected.append(QRUSelection(
                qru_name="Assurance",
                priority=7,
                required=True,
                reason="Uncertainty quantification and grounding validation",
                estimated_cost=self.QRU_COSTS["Assurance"],
                estimated_time_seconds=self.QRU_TIMES["Assurance"]
            ))

        # Always required: Policy and Action (decision and execution)
        selected.append(QRUSelection(
            qru_name="Policy",
            priority=98,
            required=True,
            reason="Multi-signal fusion and final decision",
            estimated_cost=self.QRU_COSTS["Policy"],
            estimated_time_seconds=self.QRU_TIMES["Policy"]
        ))

        selected.append(QRUSelection(
            qru_name="Action",
            priority=99,
            required=True,
            reason="Execute resolution action",
            estimated_cost=self.QRU_COSTS["Action"],
            estimated_time_seconds=self.QRU_TIMES["Action"]
        ))

        # Learning QRU only for highly complex or new scenarios
        if classification.complexity == Complexity.HIGHLY_COMPLEX or \
           self._is_new_scenario(case_data, vertical):
            selected.append(QRUSelection(
                qru_name="Learning",
                priority=100,
                required=False,
                reason="Update models from resolution outcome",
                estimated_cost=self.QRU_COSTS["Learning"],
                estimated_time_seconds=self.QRU_TIMES["Learning"]
            ))

        return sorted(selected, key=lambda x: x.priority)

    def _is_new_scenario(self, case_data: dict, vertical: str) -> bool:
        """Determine if this is a novel scenario that needs learning"""
        # Stub - would check knowledge substrate
        return False

    def _generate_reasoning(self, classification: Classification,
                           selected_qrus: List[QRUSelection],
                           skipped_qrus: List[str], vertical: str) -> str:
        """
        Generate human-readable reasoning for the execution plan

        This shows the Planner's thought process and business acumen.
        """
        reasoning = []

        # Opening statement about the business problem
        reasoning.append(f"Business Problem: {classification.business_problem.value}")
        reasoning.append(f"Classification: {classification.complexity.value} complexity, "
                        f"{classification.data_quality.value} data quality")

        # Explain QRU selection strategy
        if classification.complexity == Complexity.SIMPLE:
            reasoning.append("Strategy: Simple case with complete data - using rules and algorithms only")
            reasoning.append("Skipping expensive ML/GenAI QRUs to optimize cost")
        elif classification.complexity == Complexity.MODERATE:
            reasoning.append("Strategy: Moderate complexity - adding ML for fuzzy matching")
            reasoning.append("Skipping GenAI to save costs while maintaining accuracy")
        elif classification.complexity == Complexity.COMPLEX:
            reasoning.append("Strategy: Complex case requires full multi-agent reasoning")
            reasoning.append("Using GenAI for contextual understanding with Assurance validation")
        else:
            reasoning.append("Strategy: Highly complex - full pipeline with learning enabled")

        # Cost optimization statement
        baseline_cost = sum(self.QRU_COSTS.values())
        actual_cost = sum(qru.estimated_cost for qru in selected_qrus)
        savings = ((baseline_cost - actual_cost) / baseline_cost) * 100
        reasoning.append(f"Cost Optimization: {savings:.1f}% savings vs. full pipeline "
                        f"(${actual_cost:.4f} vs ${baseline_cost:.4f})")

        # Business-specific insights
        if vertical == "Finance" and "GenAI" in skipped_qrus:
            reasoning.append("Finance Insight: Transaction data is structured and complete - "
                           "deterministic matching is sufficient")
        elif vertical == "Healthcare" and "GenAI" in selected_qrus:
            reasoning.append("Healthcare Insight: Medical necessity requires clinical judgment - "
                           "LLM needed for contextual reasoning")

        return "\n".join(reasoning)

    def update_from_outcome(self, plan_id: str, outcome_data: dict):
        """
        Learning loop: Update Planner based on actual outcomes

        Args:
            plan_id: The execution plan ID
            outcome_data: Actual outcome including confidence, accuracy, etc.
        """
        # TODO: Implement learning loop
        # - Track if we under-estimated complexity (low confidence outcome)
        # - Track if we over-estimated complexity (high confidence with simple QRUs)
        # - Update classification model weights
        logger.info(f"Planner learning from outcome of plan {plan_id}")
        pass
