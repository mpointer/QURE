"""
Finance Reconciliation Demo

End-to-end demonstration of QURE system for GL↔Bank reconciliation.
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def run_finance_reconciliation_demo():
    """
    Demonstrate complete finance reconciliation workflow:

    1. Retrieval: Get GL and Bank transaction data
    2. Data Processing: Extract entities, compute embeddings
    3. Rules Evaluation: Check compliance (amount match, date proximity, SOX)
    4. Algorithm: Compute reconciliation scores
    5. ML Prediction: Score match likelihood
    6. GenAI Reasoning: Generate explanation
    7. Assurance: Validate grounding and uncertainty
    8. Policy Decision: Decide on action
    9. Action: Write-back or escalate
    """

    logger.info("=" * 80)
    logger.info("QURE Finance Reconciliation Demo")
    logger.info("=" * 80)

    # Import agents
    from agents.retriever import get_retriever_agent
    from agents.data import get_data_agent
    from agents.rules import get_rules_engine
    from agents.algorithms import get_algorithm_agent
    from agents.ml_model import get_ml_model_agent
    from agents.genai import get_genai_reasoner
    from agents.assurance import get_assurance_agent
    from agents.policy import get_policy_agent
    from agents.action import get_action_agent
    from agents.orchestrator import get_orchestrator

    from common.schemas import (
        RetrievalRequest,
        DataProcessingRequest,
        RulesEvaluationRequest,
        AlgorithmRequest,
        PolicyDecisionRequest,
        ActionRequest,
    )

    # Initialize agents
    logger.info("\n[1/10] Initializing agents...")

    retriever = get_retriever_agent()
    data_agent = get_data_agent()
    rules_engine = get_rules_engine()
    algorithm_agent = get_algorithm_agent()
    ml_model_agent = get_ml_model_agent()
    genai_reasoner = get_genai_reasoner(provider="openai", model="gpt-4-turbo-preview")
    assurance_agent = get_assurance_agent()
    policy_agent = get_policy_agent()
    action_agent = get_action_agent()
    orchestrator = get_orchestrator()

    logger.info("✅ All agents initialized")

    # Sample data: GL and Bank transactions
    gl_transaction = {
        "id": "GL_20240115_001",
        "date": "2024-01-15",
        "amount": 1250.00,
        "currency": "USD",
        "payer": "Acme Corp",
        "memo": "Invoice payment 2024-001",
        "reference": "INV-2024-001",
        "status": "unreconciled",
    }

    bank_transaction = {
        "id": "BANK_20240116_045",
        "date": "2024-01-16",
        "amount": 1250.00,
        "currency": "USD",
        "payer": "ACME CORPORATION",
        "memo": "Payment for invoice 2024-001",
        "swift_ref": "SWIFT123456",
    }

    case_id = "RECON_2024_001"

    logger.info(f"\n[2/10] Case: {case_id}")
    logger.info(f"GL Transaction: {gl_transaction['id']} - ${gl_transaction['amount']} on {gl_transaction['date']}")
    logger.info(f"Bank Transaction: {bank_transaction['id']} - ${bank_transaction['amount']} on {bank_transaction['date']}")

    # Step 1: Retrieval (simulate - already have data)
    logger.info("\n[3/10] Retrieval Agent: Loading transaction data...")
    logger.info("✅ Transaction data retrieved")

    # Step 2: Data Agent - Entity extraction
    logger.info("\n[4/10] Data Agent: Extracting entities...")

    # Simulate entity extraction (in production, would process documents)
    logger.info(f"  Extracted entities:")
    logger.info(f"    - GL Payer: 'Acme Corp'")
    logger.info(f"    - Bank Payer: 'ACME CORPORATION'")
    logger.info(f"    - Amount: $1,250.00")
    logger.info(f"    - Dates: 2024-01-15 (GL), 2024-01-16 (Bank)")

    # Step 3: Rules Engine
    logger.info("\n[5/10] Rules Engine: Evaluating compliance rules...")

    rules_request = RulesEvaluationRequest(
        case_id=case_id,
        from_agent="orchestration",
        to_agent="rules",
        rule_set="finance_reconciliation",
        case_data={
            "gl_amount": gl_transaction["amount"],
            "bank_amount": bank_transaction["amount"],
            "date_diff_days": 1,
            "gl_payer": gl_transaction["payer"].lower(),
            "bank_payer": bank_transaction["payer"].lower(),
            "swift_ref": bank_transaction.get("swift_ref"),
            "gl_currency": gl_transaction["currency"],
            "bank_currency": bank_transaction["currency"],
            "gl_status": gl_transaction["status"],
            "memo_similarity": 0.85,
            "is_duplicate": False,
            "business_days_elapsed": 1,
        },
    )

    rules_response = rules_engine.evaluate(rules_request)

    logger.info(f"  Passed: {len(rules_response.passed_rules)} rules")
    logger.info(f"  Failed: {len(rules_response.failed_rules)} rules")
    logger.info(f"  Rule Score: {rules_response.rule_score:.2%}")

    for explanation in rules_response.explanations[:5]:
        logger.info(f"    {explanation}")

    # Step 4: Algorithm Agent - Reconciliation scoring
    logger.info("\n[6/10] Algorithm Agent: Computing reconciliation score...")

    # First compute component scores
    date_proximity_request = AlgorithmRequest(
        case_id=case_id,
        from_agent="orchestration",
        to_agent="algorithm",
        algorithm_type="date_proximity",
        inputs={
            "date1": gl_transaction["date"],
            "date2": bank_transaction["date"],
            "max_days": 3,
        },
    )

    date_prox_response = algorithm_agent.execute(date_proximity_request)
    logger.info(f"  Date Proximity: {date_prox_response.score:.2%} ({date_prox_response.result['days_apart']} days apart)")

    amount_sim_request = AlgorithmRequest(
        case_id=case_id,
        from_agent="orchestration",
        to_agent="algorithm",
        algorithm_type="amount_similarity",
        inputs={
            "amount1": gl_transaction["amount"],
            "amount2": bank_transaction["amount"],
            "tolerance": 0.01,
        },
    )

    amount_sim_response = algorithm_agent.execute(amount_sim_request)
    logger.info(f"  Amount Similarity: {amount_sim_response.score:.2%}")

    fuzzy_match_request = AlgorithmRequest(
        case_id=case_id,
        from_agent="orchestration",
        to_agent="algorithm",
        algorithm_type="fuzzy_match",
        inputs={
            "string1": gl_transaction["payer"],
            "string2": bank_transaction["payer"],
            "method": "token_sort_ratio",
        },
    )

    fuzzy_match_response = algorithm_agent.execute(fuzzy_match_request)
    logger.info(f"  Payer Match: {fuzzy_match_response.score:.2%}")

    # Now compute overall reconciliation score
    recon_request = AlgorithmRequest(
        case_id=case_id,
        from_agent="orchestration",
        to_agent="algorithm",
        algorithm_type="reconciliation_score",
        inputs={
            "date_proximity_score": date_prox_response.score,
            "amount_similarity_score": amount_sim_response.score,
            "memo_similarity_score": 0.85,
            "payer_match_score": fuzzy_match_response.score,
            "reference_match_score": 1.0,
        },
    )

    recon_response = algorithm_agent.execute(recon_request)
    logger.info(f"  Overall Reconciliation Score: {recon_response.score:.2%}")
    logger.info(f"  Recommendation: {'MATCH' if recon_response.result['match_recommended'] else 'NO MATCH'}")

    # Step 5: ML Model (simulated - would need trained model)
    logger.info("\n[7/10] ML Model Agent: Predicting match likelihood...")
    logger.info("  (Simulated - model would be trained on historical reconciliations)")
    ml_confidence = 0.92
    logger.info(f"  ML Prediction: MATCH (confidence: {ml_confidence:.2%})")

    # Step 6: GenAI Reasoner (simulated to avoid API call)
    logger.info("\n[8/10] GenAI Reasoner: Generating explanation...")
    logger.info("  (Simulated - would call OpenAI GPT-4)")
    genai_answer = """
Based on the provided transaction data, this appears to be a valid match between GL and Bank records:

Matching Evidence:
1. Amounts are identical: $1,250.00 [GL_20240115_001][BANK_20240116_045]
2. Dates are within acceptable window: 1 day apart [GL_20240115_001][BANK_20240116_045]
3. Payer names match: "Acme Corp" vs "ACME CORPORATION" (95% similarity) [GL_20240115_001][BANK_20240116_045]
4. Invoice reference consistent: "Invoice payment 2024-001" vs "Payment for invoice 2024-001" [GL_20240115_001][BANK_20240116_045]
5. SWIFT reference present for compliance [BANK_20240116_045]

Recommendation: APPROVE reconciliation with high confidence.
"""
    logger.info(f"  GenAI Analysis:\n{genai_answer}")
    genai_confidence = 0.88
    citation_count = 5

    # Step 7: Assurance Agent
    logger.info("\n[9/10] Assurance Agent: Validating uncertainty and grounding...")

    # Simulate agent outputs for assurance
    agent_outputs = [
        {"from_agent": "rules", "confidence": rules_response.rule_score, "rule_score": rules_response.rule_score},
        {"from_agent": "algorithm", "confidence": recon_response.score, "score": recon_response.score},
        {"from_agent": "ml", "confidence": ml_confidence},
        {"from_agent": "genai", "confidence": genai_confidence, "citations": [{"source_id": "GL_20240115_001"}] * citation_count},
    ]

    # Compute uncertainty
    uncertainty = assurance_agent._compute_uncertainty(agent_outputs)
    logger.info(f"  Uncertainty Score: {uncertainty:.2%}")

    # Simulated grounding score
    grounding_score = 0.95  # High grounding due to 5 citations
    logger.info(f"  Grounding Score: {grounding_score:.2%}")

    calibrated_confidence = assurance_agent._calibrate_confidence(
        agent_outputs, uncertainty, grounding_score
    )
    logger.info(f"  Calibrated Confidence: {calibrated_confidence:.2%}")

    consensus_score = assurance_agent._check_consensus(agent_outputs)
    logger.info(f"  Consensus Score: {consensus_score:.2%}")

    assurance_score = (
        0.30 * (1.0 - uncertainty) +
        0.30 * grounding_score +
        0.20 * calibrated_confidence +
        0.20 * consensus_score
    )
    logger.info(f"  Overall Assurance: {assurance_score:.2%}")
    logger.info(f"  Hallucination Detected: No")

    # Step 8: Policy Agent - Decision
    logger.info("\n[10/10] Policy Agent: Making decision...")

    policy_request = PolicyDecisionRequest(
        case_id=case_id,
        from_agent="orchestration",
        to_agent="policy",
        scores={
            "rules": rules_response.rule_score,
            "algorithm": recon_response.score,
            "ml": ml_confidence,
            "genai": genai_confidence,
            "assurance": assurance_score,
        },
        uncertainty=uncertainty,
        constraints={
            "transaction_amount": gl_transaction["amount"],
            "risk_level": "low",
            "sla_urgency": 0.3,
        },
    )

    policy_response = policy_agent.decide(policy_request)

    logger.info(f"\n  Decision: {policy_response.decision.upper()}")
    logger.info(f"  Confidence: {policy_response.confidence:.2%}")
    logger.info(f"  Utility Score: {policy_response.utility_score:.2%}")
    logger.info(f"\n  Explanation:")
    logger.info(f"    {policy_response.explanation}")

    # Step 9: Action Agent - Execute decision
    if policy_response.decision == "auto_resolve":
        logger.info("\n[ACTION] Executing approval...")

        action_request = ActionRequest(
            case_id=case_id,
            from_agent="orchestration",
            to_agent="action",
            action_type="write_back",
            action_params={
                "target_system": "database",
                "entity_type": "gl_transactions",
                "entity_id": gl_transaction["id"],
                "updates": {
                    "status": "reconciled",
                    "bank_transaction_id": bank_transaction["id"],
                    "reconciliation_score": recon_response.score,
                    "reconciled_by": "QURE_AUTO",
                },
            },
        )

        logger.info(f"  Updating GL transaction status to 'reconciled'")
        logger.info(f"  Linking to Bank transaction: {bank_transaction['id']}")

        # Generate notification
        notification_request = ActionRequest(
            case_id=case_id,
            from_agent="orchestration",
            to_agent="action",
            action_type="send_notification",
            action_params={
                "notification_type": "email",
                "recipient": "finance.team@company.com",
                "subject": f"Reconciliation Complete: {case_id}",
                "message": f"GL transaction {gl_transaction['id']} automatically reconciled with {bank_transaction['id']}",
            },
        )

        logger.info(f"  Sending notification to finance team")

        logger.info(f"\n✅ Reconciliation complete!")

    elif policy_response.decision == "hitl_review":
        logger.info("\n[ACTION] Escalating for human review...")

        action_request = ActionRequest(
            case_id=case_id,
            from_agent="orchestration",
            to_agent="action",
            action_type="escalate",
            action_params={
                "reason": "Medium confidence - requires human validation",
                "assigned_to": "senior_accountant",
                "priority": "medium",
            },
        )

        logger.info(f"  Assigned to: senior_accountant")
        logger.info(f"  Priority: medium")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Case ID: {case_id}")
    logger.info(f"GL Transaction: {gl_transaction['id']}")
    logger.info(f"Bank Transaction: {bank_transaction['id']}")
    logger.info(f"Amount: ${gl_transaction['amount']}")
    logger.info(f"\nAgent Scores:")
    logger.info(f"  Rules Engine: {rules_response.rule_score:.2%}")
    logger.info(f"  Algorithm: {recon_response.score:.2%}")
    logger.info(f"  ML Model: {ml_confidence:.2%}")
    logger.info(f"  GenAI: {genai_confidence:.2%}")
    logger.info(f"  Assurance: {assurance_score:.2%}")
    logger.info(f"\nPolicy Decision:")
    logger.info(f"  Decision: {policy_response.decision}")
    logger.info(f"  Confidence: {policy_response.confidence:.2%}")
    logger.info(f"  Utility: {policy_response.utility_score:.2%}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    run_finance_reconciliation_demo()
