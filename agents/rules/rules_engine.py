"""
Rules Engine Agent

Applies deterministic business logic, compliance checks, and validation rules.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.schemas import RulesEvaluationRequest, RulesEvaluationResponse

logger = logging.getLogger(__name__)


class RulesEngine:
    """
    Rules Engine Agent for deterministic logic

    Responsibilities:
    - Evaluate rules DSL
    - Return pass/fail/needs_evidence with rationale
    - Enforce hard constraints (SOX, HIPAA, compliance)
    - Explain which rules passed/failed
    - Support rule versioning
    """

    def __init__(self, rules_dir: Optional[Path] = None):
        """
        Initialize Rules Engine

        Args:
            rules_dir: Directory containing rule definition files
        """
        if rules_dir is None:
            rules_dir = Path(__file__).parent / "rule_library"

        self.rules_dir = rules_dir
        self.rules_cache: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(f"✅ Rules Engine initialized with rules from {rules_dir}")

    def evaluate(
        self,
        request: RulesEvaluationRequest,
    ) -> RulesEvaluationResponse:
        """
        Evaluate rules against case data

        Args:
            request: RulesEvaluationRequest with case data and rule set

        Returns:
            RulesEvaluationResponse with results
        """
        # Load rules for the specified set
        rules = self._load_rule_set(request.rule_set)

        if not rules:
            logger.warning(f"No rules found for rule set: {request.rule_set}")
            return RulesEvaluationResponse(
                case_id=request.case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                passed_rules=[],
                failed_rules=[],
                needs_evidence=[],
                rule_score=0.0,
                explanations=[f"Rule set '{request.rule_set}' not found"],
            )

        passed_rules = []
        failed_rules = []
        needs_evidence = []
        explanations = []

        # Evaluate each rule
        for rule in rules:
            result = self._evaluate_single_rule(rule, request.case_data)

            rule_id = rule.get("rule_id", "unknown")

            if result["status"] == "pass":
                passed_rules.append(rule_id)
                explanations.append(f"✓ {rule_id}: {result['explanation']}")

            elif result["status"] == "fail":
                failed_rules.append(rule_id)
                explanations.append(f"✗ {rule_id}: {result['explanation']}")

            elif result["status"] == "needs_evidence":
                needs_evidence.append({
                    "rule_id": rule_id,
                    "required_evidence": result.get("required_evidence", []),
                    "explanation": result["explanation"],
                })
                explanations.append(f"? {rule_id}: {result['explanation']}")

        # Compute overall score
        total_rules = len(rules)
        mandatory_failures = sum(
            1 for rule_id in failed_rules
            if self._is_mandatory_rule(rule_id, rules)
        )

        if mandatory_failures > 0:
            rule_score = 0.0  # Hard fail on mandatory rules
        elif needs_evidence:
            rule_score = 0.5  # Partial score if evidence needed
        else:
            rule_score = len(passed_rules) / total_rules if total_rules > 0 else 1.0

        logger.info(
            f"Rules evaluation for {request.case_id}: "
            f"{len(passed_rules)} passed, {len(failed_rules)} failed, "
            f"{len(needs_evidence)} need evidence (score: {rule_score:.2f})"
        )

        return RulesEvaluationResponse(
            case_id=request.case_id,
            from_agent=request.to_agent or request.from_agent,
            to_agent=None,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            needs_evidence=needs_evidence,
            rule_score=rule_score,
            explanations=explanations,
        )

    def _evaluate_single_rule(
        self,
        rule: Dict[str, Any],
        case_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single rule

        Args:
            rule: Rule definition
            case_data: Case data to evaluate against

        Returns:
            Dict with status, explanation, and optional required_evidence
        """
        rule_id = rule.get("rule_id", "unknown")
        condition = rule.get("condition", "")
        action = rule.get("action", "pass")
        requires_evidence = rule.get("requires_evidence", [])

        try:
            # Evaluate condition
            if self._evaluate_condition(condition, case_data):
                # Condition is true
                if requires_evidence:
                    # Check if evidence is present
                    has_evidence = self._check_evidence(requires_evidence, case_data)

                    if has_evidence:
                        return {
                            "status": "pass",
                            "explanation": f"Condition met and evidence provided",
                        }
                    else:
                        return {
                            "status": "needs_evidence",
                            "explanation": f"Condition met but missing evidence",
                            "required_evidence": requires_evidence,
                        }
                else:
                    return {
                        "status": "pass",
                        "explanation": f"Condition met",
                    }
            else:
                # Condition is false
                if action == "fail":
                    return {
                        "status": "fail",
                        "explanation": f"Condition not met (required)",
                    }
                else:
                    return {
                        "status": "pass",
                        "explanation": f"Condition not met (optional)",
                    }

        except Exception as e:
            logger.error(f"Error evaluating rule {rule_id}: {e}")
            return {
                "status": "fail",
                "explanation": f"Rule evaluation error: {str(e)}",
            }

    def _evaluate_condition(
        self,
        condition: str,
        case_data: Dict[str, Any],
    ) -> bool:
        """
        Evaluate a rule condition

        Args:
            condition: Condition expression (e.g., "amount > 1000 AND date_diff < 7")
            case_data: Case data

        Returns:
            True if condition is met
        """
        try:
            # Simple expression evaluator
            # In production, use a proper DSL parser (e.g., pyparsing, lark)

            # Replace field names with values from case_data
            expr = condition

            # Handle common operators
            for key, value in case_data.items():
                # Replace field references
                expr = expr.replace(key, repr(value))

            # Handle "in" operator
            expr = expr.replace(" IN ", " in ")
            expr = expr.replace(" AND ", " and ")
            expr = expr.replace(" OR ", " or ")

            # Evaluate (DANGER: use ast.literal_eval in production!)
            result = eval(expr, {"__builtins__": {}}, {})

            return bool(result)

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    def _check_evidence(
        self,
        required_evidence: List[str],
        case_data: Dict[str, Any],
    ) -> bool:
        """
        Check if required evidence fields are present in case data

        Args:
            required_evidence: List of required evidence field names
            case_data: Case data

        Returns:
            True if all evidence is present
        """
        for evidence_field in required_evidence:
            if evidence_field not in case_data or not case_data[evidence_field]:
                return False

        return True

    def _is_mandatory_rule(
        self,
        rule_id: str,
        rules: List[Dict[str, Any]],
    ) -> bool:
        """
        Check if a rule is mandatory

        Args:
            rule_id: Rule ID
            rules: List of all rules

        Returns:
            True if rule is mandatory
        """
        for rule in rules:
            if rule.get("rule_id") == rule_id:
                return rule.get("mandatory", False)

        return False

    def _load_rule_set(self, rule_set_name: str) -> List[Dict[str, Any]]:
        """
        Load rules from a rule set file

        Args:
            rule_set_name: Name of the rule set (e.g., "finance_reconciliation")

        Returns:
            List of rule definitions
        """
        # Check cache first
        if rule_set_name in self.rules_cache:
            return self.rules_cache[rule_set_name]

        # Load from file
        rule_file = self.rules_dir / f"{rule_set_name}.json"

        if not rule_file.exists():
            logger.error(f"Rule file not found: {rule_file}")
            return []

        try:
            with open(rule_file, "r") as f:
                rules = json.load(f)

            # Cache the rules
            self.rules_cache[rule_set_name] = rules

            logger.debug(f"Loaded {len(rules)} rules from {rule_set_name}")
            return rules

        except Exception as e:
            logger.error(f"Failed to load rules from {rule_file}: {e}")
            return []

    def get_available_rule_sets(self) -> List[str]:
        """
        Get list of available rule sets

        Returns:
            List of rule set names
        """
        if not self.rules_dir.exists():
            return []

        rule_files = list(self.rules_dir.glob("*.json"))
        return [f.stem for f in rule_files]

    def reload_rules(self, rule_set_name: Optional[str] = None):
        """
        Reload rules from disk

        Args:
            rule_set_name: Specific rule set to reload (or all if None)
        """
        if rule_set_name:
            if rule_set_name in self.rules_cache:
                del self.rules_cache[rule_set_name]
            logger.info(f"Reloaded rules: {rule_set_name}")
        else:
            self.rules_cache.clear()
            logger.info("Reloaded all rules")


# Singleton instance
_rules_engine: Optional[RulesEngine] = None


def get_rules_engine(rules_dir: Optional[Path] = None) -> RulesEngine:
    """
    Get or create singleton RulesEngine instance

    Args:
        rules_dir: Directory containing rule files

    Returns:
        RulesEngine instance
    """
    global _rules_engine

    if _rules_engine is None:
        _rules_engine = RulesEngine(rules_dir=rules_dir)

    return _rules_engine
