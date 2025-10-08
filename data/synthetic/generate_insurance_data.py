"""
Synthetic Insurance Data Generator

Generates realistic subrogation claims test cases with various scenarios.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)


class InsuranceDataGenerator:
    """Generate synthetic subrogation claims and recovery data"""

    def __init__(self):
        self.claim_types = [
            {"type": "Auto Collision", "typical_cost": 5000},
            {"type": "Property Damage", "typical_cost": 8000},
            {"type": "Medical Expenses", "typical_cost": 12000},
            {"type": "Water Damage", "typical_cost": 6000},
            {"type": "Fire Damage", "typical_cost": 25000},
            {"type": "Theft", "typical_cost": 3000},
            {"type": "Liability", "typical_cost": 15000},
        ]

        self.liability_indicators = [
            "Police report confirms",
            "Witness statements support",
            "Photographic evidence shows",
            "Expert analysis indicates",
            "Adjuster report validates",
        ]

        self.recovery_statuses = [
            "Unrecovered",
            "Partial Recovery",
            "Full Recovery Pending",
            "Under Negotiation",
        ]

    def generate_dataset(self, n_cases: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate dataset with various scenarios

        Args:
            n_cases: Number of test cases

        Returns:
            Dict with 'claim_submissions', 'recovery_evidence', 'expected_matches'
        """
        claim_submissions = []
        recovery_evidence = []
        expected_matches = []

        # Case types distribution
        auto_approve = int(n_cases * 0.35)  # 35% auto-approve (clear liability)
        hitl_review = int(n_cases * 0.40)   # 40% HITL review (partial liability, disputed)
        request_info = int(n_cases * 0.15)  # 15% request more info (missing evidence)
        reject = n_cases - auto_approve - hitl_review - request_info  # 10% reject (no liability)

        case_id = 1

        # Generate auto-approve cases
        for i in range(auto_approve):
            claim, evidence, match = self._generate_auto_approve(case_id)
            claim_submissions.append(claim)
            recovery_evidence.append(evidence)
            expected_matches.append(match)
            case_id += 1

        # Generate HITL review cases
        for i in range(hitl_review):
            claim, evidence, match = self._generate_hitl_review(case_id)
            claim_submissions.append(claim)
            recovery_evidence.append(evidence)
            expected_matches.append(match)
            case_id += 1

        # Generate request info cases
        for i in range(request_info):
            claim, evidence, match = self._generate_request_info(case_id)
            claim_submissions.append(claim)
            recovery_evidence.append(evidence)
            expected_matches.append(match)
            case_id += 1

        # Generate reject cases
        for i in range(reject):
            claim, evidence, match = self._generate_reject(case_id)
            claim_submissions.append(claim)
            recovery_evidence.append(evidence)
            expected_matches.append(match)
            case_id += 1

        return {
            "claim_submissions": claim_submissions,
            "recovery_evidence": recovery_evidence,
            "expected_matches": expected_matches,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_cases": n_cases,
                "auto_approve": auto_approve,
                "hitl_review": hitl_review,
                "request_info": request_info,
                "reject": reject,
            }
        }

    def _generate_auto_approve(self, case_id: int) -> tuple:
        """Generate auto-approve scenario - clear liability"""
        incident_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        claim_type = random.choice(self.claim_types)
        amount = round(claim_type["typical_cost"] * random.uniform(0.8, 1.2), 2)

        # Claim submission
        claim = {
            "id": f"CLM_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "submission_date": (incident_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
            "incident_date": incident_date.strftime("%Y-%m-%d"),
            "claim_type": claim_type["type"],
            "paid_amount": amount,
            "currency": "USD",
            "insured_name": f"Policy Holder {case_id}",
            "policy_number": f"POL-{2024000 + case_id}",
            "responsible_party": f"Third Party LLC {case_id}",
            "responsible_party_insurance": f"Carrier Insurance Co.",
            "status": "Paid - Subrogation Pending",
        }

        # Recovery evidence
        evidence = {
            "id": f"EVD_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_date": (incident_date + timedelta(days=random.randint(10, 20))).strftime("%Y-%m-%d"),
            "liability_score": random.uniform(0.90, 1.0),
            "police_report": True,
            "witness_statements": random.randint(2, 4),
            "photographic_evidence": True,
            "expert_opinion": True,
            "liability_summary": f"{random.choice(self.liability_indicators)} 100% liability of third party",
            "recovery_potential": "High",
            "legal_review": "Favorable",
        }

        # Expected match
        match = {
            "case_id": f"SUBRO_2024_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_id": evidence["id"],
            "expected_decision": "auto_resolve",
            "match_score": 0.95,
            "notes": "Clear liability with strong evidence - proceed with subrogation recovery",
        }

        return claim, evidence, match

    def _generate_hitl_review(self, case_id: int) -> tuple:
        """Generate HITL review scenario - partial liability or disputed"""
        incident_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        claim_type = random.choice(self.claim_types)
        amount = round(claim_type["typical_cost"] * random.uniform(0.8, 1.2), 2)

        review_reason = random.choice([
            "partial_liability",
            "disputed_facts",
            "conflicting_evidence",
            "comparative_negligence",
        ])

        claim = {
            "id": f"CLM_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "submission_date": (incident_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
            "incident_date": incident_date.strftime("%Y-%m-%d"),
            "claim_type": claim_type["type"],
            "paid_amount": amount,
            "currency": "USD",
            "insured_name": f"Policy Holder {case_id}",
            "policy_number": f"POL-{2024000 + case_id}",
            "responsible_party": f"Third Party LLC {case_id}",
            "responsible_party_insurance": f"Carrier Insurance Co.",
            "status": "Paid - Subrogation Pending",
        }

        # Adjust evidence based on review reason
        if review_reason == "partial_liability":
            liability_score = random.uniform(0.60, 0.75)
            liability_summary = "Shared liability - third party 60-70% at fault"
            recovery_potential = "Medium"
        elif review_reason == "disputed_facts":
            liability_score = random.uniform(0.65, 0.80)
            liability_summary = "Disputed facts - conflicting witness statements"
            recovery_potential = "Medium"
        elif review_reason == "conflicting_evidence":
            liability_score = random.uniform(0.55, 0.70)
            liability_summary = "Conflicting evidence requires legal review"
            recovery_potential = "Medium-Low"
        else:  # comparative_negligence
            liability_score = random.uniform(0.50, 0.65)
            liability_summary = "Comparative negligence - both parties contributed"
            recovery_potential = "Low-Medium"

        evidence = {
            "id": f"EVD_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_date": (incident_date + timedelta(days=random.randint(10, 20))).strftime("%Y-%m-%d"),
            "liability_score": liability_score,
            "police_report": random.choice([True, False]),
            "witness_statements": random.randint(0, 2),
            "photographic_evidence": random.choice([True, False]),
            "expert_opinion": random.choice([True, False]),
            "liability_summary": liability_summary,
            "recovery_potential": recovery_potential,
            "legal_review": "Requires Review",
        }

        match = {
            "case_id": f"SUBRO_2024_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_id": evidence["id"],
            "expected_decision": "hitl_review",
            "match_score": 0.65,
            "notes": f"Requires human review - {review_reason.replace('_', ' ')}",
        }

        return claim, evidence, match

    def _generate_request_info(self, case_id: int) -> tuple:
        """Generate request info scenario - missing evidence"""
        incident_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        claim_type = random.choice(self.claim_types)
        amount = round(claim_type["typical_cost"] * random.uniform(0.8, 1.2), 2)

        missing_element = random.choice([
            "police_report",
            "witness_statements",
            "photographic_evidence",
            "expert_opinion",
        ])

        claim = {
            "id": f"CLM_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "submission_date": (incident_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
            "incident_date": incident_date.strftime("%Y-%m-%d"),
            "claim_type": claim_type["type"],
            "paid_amount": amount,
            "currency": "USD",
            "insured_name": f"Policy Holder {case_id}",
            "policy_number": f"POL-{2024000 + case_id}",
            "responsible_party": f"Third Party LLC {case_id}",
            "responsible_party_insurance": f"Carrier Insurance Co.",
            "status": "Paid - Subrogation Pending",
        }

        evidence = {
            "id": f"EVD_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_date": (incident_date + timedelta(days=random.randint(10, 20))).strftime("%Y-%m-%d"),
            "liability_score": random.uniform(0.40, 0.60),
            "police_report": missing_element != "police_report",
            "witness_statements": 0 if missing_element == "witness_statements" else random.randint(1, 2),
            "photographic_evidence": missing_element != "photographic_evidence",
            "expert_opinion": missing_element != "expert_opinion",
            "liability_summary": f"Insufficient evidence - missing {missing_element.replace('_', ' ')}",
            "recovery_potential": "Unknown",
            "legal_review": "Pending Evidence",
        }

        match = {
            "case_id": f"SUBRO_2024_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_id": evidence["id"],
            "expected_decision": "request_evidence",
            "match_score": 0.45,
            "notes": f"Insufficient evidence - request {missing_element.replace('_', ' ')}",
        }

        return claim, evidence, match

    def _generate_reject(self, case_id: int) -> tuple:
        """Generate reject scenario - no viable subrogation"""
        incident_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        claim_type = random.choice(self.claim_types)
        amount = round(claim_type["typical_cost"] * random.uniform(0.8, 1.2), 2)

        reject_reason = random.choice([
            "no_third_party_liability",
            "insured_at_fault",
            "uninsured_responsible_party",
            "statute_of_limitations",
        ])

        claim = {
            "id": f"CLM_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "submission_date": (incident_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
            "incident_date": incident_date.strftime("%Y-%m-%d"),
            "claim_type": claim_type["type"],
            "paid_amount": amount,
            "currency": "USD",
            "insured_name": f"Policy Holder {case_id}",
            "policy_number": f"POL-{2024000 + case_id}",
            "responsible_party": f"Third Party LLC {case_id}" if reject_reason != "no_third_party_liability" else "N/A",
            "responsible_party_insurance": "Unknown" if reject_reason == "uninsured_responsible_party" else "Carrier Insurance Co.",
            "status": "Paid - Subrogation Pending",
        }

        if reject_reason == "no_third_party_liability":
            liability_summary = "No third party liability identified - single-vehicle incident"
            liability_score = 0.0
        elif reject_reason == "insured_at_fault":
            liability_summary = "Insured party determined to be 100% at fault"
            liability_score = 0.0
        elif reject_reason == "uninsured_responsible_party":
            liability_summary = "Third party uninsured and no assets for recovery"
            liability_score = 0.85
        else:  # statute_of_limitations
            liability_summary = "Statute of limitations expired - cannot pursue"
            liability_score = 0.75

        evidence = {
            "id": f"EVD_{incident_date.strftime('%Y%m%d')}_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_date": (incident_date + timedelta(days=random.randint(10, 20))).strftime("%Y-%m-%d"),
            "liability_score": liability_score,
            "police_report": True,
            "witness_statements": random.randint(0, 2),
            "photographic_evidence": True,
            "expert_opinion": False,
            "liability_summary": liability_summary,
            "recovery_potential": "None",
            "legal_review": "Not Viable",
        }

        match = {
            "case_id": f"SUBRO_2024_{case_id:03d}",
            "claim_id": claim["id"],
            "evidence_id": evidence["id"],
            "expected_decision": "reject",
            "match_score": 0.15,
            "notes": f"Subrogation not viable - {reject_reason.replace('_', ' ')}",
        }

        return claim, evidence, match

    def save_dataset(self, dataset: Dict[str, Any], output_dir: Path):
        """Save dataset to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save claim submissions
        claims_file = output_dir / "claim_submissions.json"
        with open(claims_file, "w") as f:
            json.dump(dataset["claim_submissions"], f, indent=2)

        # Save recovery evidence
        evidence_file = output_dir / "recovery_evidence.json"
        with open(evidence_file, "w") as f:
            json.dump(dataset["recovery_evidence"], f, indent=2)

        # Save expected matches
        matches_file = output_dir / "expected_matches.json"
        with open(matches_file, "w") as f:
            json.dump(dataset["expected_matches"], f, indent=2)

        # Save metadata
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

        print(f"[OK] Dataset saved to {output_dir}")
        print(f"   - Claim submissions: {len(dataset['claim_submissions'])}")
        print(f"   - Recovery evidence: {len(dataset['recovery_evidence'])}")
        print(f"   - Expected matches: {len(dataset['expected_matches'])}")


def main():
    """Generate synthetic insurance subrogation data"""
    print("Generating synthetic subrogation claims data...")

    generator = InsuranceDataGenerator()
    dataset = generator.generate_dataset(n_cases=20)

    # Save to data/synthetic/insurance
    output_dir = Path(__file__).parent / "insurance"
    generator.save_dataset(dataset, output_dir)

    # Print summary
    print("\nDataset Summary:")
    print(f"  Auto-Approve Cases: {dataset['metadata']['auto_approve']} (35%)")
    print(f"  HITL Review Cases: {dataset['metadata']['hitl_review']} (40%)")
    print(f"  Request Info Cases: {dataset['metadata']['request_info']} (15%)")
    print(f"  Reject Cases: {dataset['metadata']['reject']} (10%)")
    print("\nExpected Auto-Resolution Rate: ~35%")
    print("Expected Human Review Rate: ~40%")
    print("Expected Rejection Rate: ~25%")


if __name__ == "__main__":
    main()
