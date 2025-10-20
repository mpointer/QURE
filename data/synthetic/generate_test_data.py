"""
Synthetic Data Generator for QURE Multi-Vertical Demos

Generates realistic test cases for:
1. Finance: GL↔Bank reconciliation
2. Insurance: Subrogation claims
3. Healthcare: Prior authorization

Each case includes:
- Rich context features for ML/clustering
- Ground truth labels for validation
- Complexity variations (easy, medium, hard)
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import numpy as np


class FinanceDataGenerator:
    """Generate GL-Bank reconciliation test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.banks = ["Chase", "BofA", "Wells Fargo", "Citi"]
        self.gl_accounts = ["1000-Cash", "1010-Operating", "1020-Payroll", "2000-AR"]
        self.vendors = ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Partners", "Epsilon Ltd"]

    def generate_case(self, case_id: int, complexity: str = "medium") -> Dict:
        """Generate a single finance reconciliation case"""

        # Base transaction
        amount = self._get_amount_by_complexity(complexity)
        date = datetime.now() - timedelta(days=random.randint(1, 90))

        # GL entry
        gl_entry = {
            'id': f'GL-{date.strftime("%Y%m%d")}-{case_id:04d}',
            'date': date.strftime('%Y-%m-%d'),
            'account': random.choice(self.gl_accounts),
            'amount': amount,
            'description': f"Payment to {random.choice(self.vendors)}",
            'reference': f'INV-{random.randint(10000, 99999)}',
            'posted_by': random.choice(['jsmith', 'mjones', 'alee'])
        }

        # Bank statement entry (may have variations)
        bank_amount = amount
        bank_date = date
        bank_description = gl_entry['description']

        # Introduce complexity variations
        if complexity == "hard":
            # Add intentional mismatches
            if random.random() < 0.3:
                bank_amount += random.uniform(-50, 50)  # Amount variation
            if random.random() < 0.2:
                bank_date += timedelta(days=random.randint(1, 5))  # Date drift
            if random.random() < 0.2:
                bank_description = bank_description[:20] + "..."  # Truncated

        bank_entry = {
            'id': f'BNK-{bank_date.strftime("%Y%m%d")}-{random.randint(1000, 9999)}',
            'date': bank_date.strftime('%Y-%m-%d'),
            'bank': random.choice(self.banks),
            'amount': bank_amount,
            'description': bank_description,
            'check_number': f'{random.randint(1000, 9999)}' if random.random() < 0.5 else None
        }

        # Determine ground truth
        amount_match = abs(gl_entry['amount'] - bank_entry['amount']) < 1.0
        date_match = abs((date - bank_date).days) <= 2
        description_sim = self._fuzzy_match(gl_entry['description'], bank_entry['description'])

        is_match = amount_match and date_match and description_sim > 0.7
        confidence = (amount_match * 0.4 + date_match * 0.3 + description_sim * 0.3)

        # Context features
        context = {
            'transaction_amount': amount,
            'data_quality_score': self._compute_quality_score(gl_entry, bank_entry),
            'urgency': self._determine_urgency(date, amount),
            'has_swift_reference': bool(gl_entry['reference']),
            'sox_controlled': amount > 500_000,
            'vertical': 'finance',
            'complexity': complexity
        }

        return {
            'case_id': gl_entry['id'],
            'gl_entry': gl_entry,
            'bank_entry': bank_entry,
            'ground_truth': {
                'is_match': is_match,
                'confidence': confidence,
                'should_auto_resolve': complexity == "easy" and is_match,
                'requires_review': complexity == "hard" or not is_match
            },
            'context': context
        }

    def _get_amount_by_complexity(self, complexity: str) -> float:
        """Generate transaction amount based on complexity"""
        if complexity == "easy":
            return round(random.uniform(1000, 50000), 2)
        elif complexity == "medium":
            return round(random.uniform(50000, 500000), 2)
        else:  # hard
            return round(random.uniform(500000, 5000000), 2)

    def _fuzzy_match(self, s1: str, s2: str) -> float:
        """Simple fuzzy string matching"""
        s1_words = set(s1.lower().split())
        s2_words = set(s2.lower().split())
        if not s1_words or not s2_words:
            return 0.0
        intersection = len(s1_words & s2_words)
        union = len(s1_words | s2_words)
        return intersection / union if union > 0 else 0.0

    def _compute_quality_score(self, gl: Dict, bank: Dict) -> float:
        """Compute data quality score"""
        score = 1.0

        # Penalize missing data
        if not gl.get('reference'):
            score -= 0.1
        if not gl.get('description'):
            score -= 0.2
        if not bank.get('description'):
            score -= 0.2

        # Penalize short descriptions
        if len(gl.get('description', '')) < 10:
            score -= 0.1
        if len(bank.get('description', '')) < 10:
            score -= 0.1

        return max(score, 0.3)

    def _determine_urgency(self, date: datetime, amount: float) -> str:
        """Determine case urgency"""
        days_old = (datetime.now() - date).days

        if days_old > 60 or amount > 1_000_000:
            return 'critical'
        elif days_old > 30 or amount > 500_000:
            return 'high'
        elif days_old > 14:
            return 'medium'
        else:
            return 'low'

    def generate_dataset(self, n_cases: int = 100) -> List[Dict]:
        """Generate full dataset with complexity distribution"""
        # Distribution: 60% easy, 30% medium, 10% hard
        complexities = (
            ['easy'] * int(n_cases * 0.6) +
            ['medium'] * int(n_cases * 0.3) +
            ['hard'] * int(n_cases * 0.1)
        )
        random.shuffle(complexities)

        cases = []
        for i, complexity in enumerate(complexities):
            cases.append(self.generate_case(i, complexity))

        return cases


class InsuranceDataGenerator:
    """Generate insurance subrogation claim cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.incident_types = ["Auto Accident", "Property Damage", "Slip and Fall", "Medical Malpractice"]
        self.liable_parties = ["State Farm", "Allstate", "GEICO", "Progressive", "Third Party"]

    def generate_case(self, case_id: int, complexity: str = "medium") -> Dict:
        """Generate a single subrogation case"""

        claim_date = datetime.now() - timedelta(days=random.randint(30, 365))
        recovery_amount = self._get_recovery_amount(complexity)

        claim = {
            'claim_id': f'SUB-{claim_date.strftime("%Y%m%d")}-{case_id:04d}',
            'incident_date': claim_date.strftime('%Y-%m-%d'),
            'incident_type': random.choice(self.incident_types),
            'recovery_amount': recovery_amount,
            'liable_party': random.choice(self.liable_parties),
            'policy_number': f'POL-{random.randint(100000, 999999)}',
            'has_police_report': random.random() > 0.2,
            'has_liability_admission': random.random() > 0.6 if complexity == "easy" else random.random() > 0.8,
            'statute_expiry_days': random.randint(30, 730)
        }

        # Documentation quality
        doc_quality = 0.9 if complexity == "easy" else 0.7 if complexity == "medium" else 0.4
        doc_quality += random.uniform(-0.1, 0.1)

        # Ground truth
        success_probability = (
            0.9 if complexity == "easy" else
            0.6 if complexity == "medium" else
            0.3
        )

        context = {
            'transaction_amount': recovery_amount,
            'data_quality_score': doc_quality,
            'urgency': self._determine_subro_urgency(claim['statute_expiry_days']),
            'has_swift_reference': claim['has_police_report'],
            'days_to_statute_expiry': claim['statute_expiry_days'],
            'vertical': 'insurance',
            'complexity': complexity
        }

        return {
            'case_id': claim['claim_id'],
            'claim': claim,
            'ground_truth': {
                'should_pursue': success_probability > 0.5,
                'expected_recovery': recovery_amount * success_probability,
                'estimated_success_rate': success_probability,
                'requires_review': complexity == "hard"
            },
            'context': context
        }

    def _get_recovery_amount(self, complexity: str) -> float:
        """Generate recovery amount"""
        if complexity == "easy":
            return round(random.uniform(5000, 25000), 2)
        elif complexity == "medium":
            return round(random.uniform(25000, 100000), 2)
        else:
            return round(random.uniform(100000, 500000), 2)

    def _determine_subro_urgency(self, days_to_expiry: int) -> str:
        """Determine subrogation urgency based on statute of limitations"""
        if days_to_expiry < 30:
            return 'critical'
        elif days_to_expiry < 90:
            return 'high'
        elif days_to_expiry < 180:
            return 'medium'
        else:
            return 'low'

    def generate_dataset(self, n_cases: int = 100) -> List[Dict]:
        """Generate full dataset"""
        complexities = (
            ['easy'] * int(n_cases * 0.5) +
            ['medium'] * int(n_cases * 0.3) +
            ['hard'] * int(n_cases * 0.2)
        )
        random.shuffle(complexities)

        cases = []
        for i, complexity in enumerate(complexities):
            cases.append(self.generate_case(i, complexity))

        return cases


class HealthcareDataGenerator:
    """Generate healthcare prior authorization cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.procedures = [
            "MRI Scan", "CT Scan", "Surgery - Orthopedic",
            "Physical Therapy", "Specialty Medication", "Inpatient Stay"
        ]
        self.diagnoses = [
            "Chronic Back Pain", "Sports Injury", "Arthritis",
            "Cancer Treatment", "Cardiac Condition", "Diabetes Management"
        ]

    def generate_case(self, case_id: int, complexity: str = "medium") -> Dict:
        """Generate a single prior authorization case"""

        request_date = datetime.now() - timedelta(days=random.randint(1, 30))
        procedure_cost = self._get_procedure_cost(complexity)

        auth_request = {
            'auth_id': f'AUTH-{request_date.strftime("%Y%m%d")}-{case_id:04d}',
            'request_date': request_date.strftime('%Y-%m-%d'),
            'procedure': random.choice(self.procedures),
            'diagnosis': random.choice(self.diagnoses),
            'estimated_cost': procedure_cost,
            'provider_id': f'NPI-{random.randint(1000000, 9999999)}',
            'member_id': f'MEM-{random.randint(100000, 999999)}',
            'clinical_urgency': self._determine_clinical_urgency(complexity),
            'has_prior_treatment': random.random() > 0.3,
            'has_clinical_notes': random.random() > 0.2,
            'medical_necessity_score': self._get_medical_necessity(complexity)
        }

        # Documentation quality
        doc_quality = 0.95 if complexity == "easy" else 0.75 if complexity == "medium" else 0.5
        doc_quality += random.uniform(-0.05, 0.05)

        # Ground truth (approval decision)
        should_approve = (
            auth_request['medical_necessity_score'] > 0.7 and
            auth_request['has_clinical_notes'] and
            (complexity == "easy" or (complexity == "medium" and random.random() > 0.3))
        )

        context = {
            'transaction_amount': procedure_cost,
            'data_quality_score': doc_quality,
            'urgency': auth_request['clinical_urgency'],
            'has_swift_reference': auth_request['has_prior_treatment'],
            'medical_necessity': auth_request['medical_necessity_score'],
            'clinical_urgency': auth_request['clinical_urgency'],
            'vertical': 'healthcare',
            'complexity': complexity
        }

        return {
            'case_id': auth_request['auth_id'],
            'auth_request': auth_request,
            'ground_truth': {
                'should_approve': should_approve,
                'confidence': auth_request['medical_necessity_score'],
                'requires_peer_review': complexity == "hard",
                'requires_additional_info': not auth_request['has_clinical_notes']
            },
            'context': context
        }

    def _get_procedure_cost(self, complexity: str) -> float:
        """Generate procedure cost"""
        if complexity == "easy":
            return round(random.uniform(500, 5000), 2)
        elif complexity == "medium":
            return round(random.uniform(5000, 50000), 2)
        else:
            return round(random.uniform(50000, 200000), 2)

    def _determine_clinical_urgency(self, complexity: str) -> str:
        """Determine clinical urgency"""
        if complexity == "hard":
            return random.choice(['urgent', 'critical'])
        elif complexity == "medium":
            return random.choice(['medium', 'high', 'urgent'])
        else:
            return random.choice(['routine', 'low', 'medium'])

    def _get_medical_necessity(self, complexity: str) -> float:
        """Generate medical necessity score"""
        if complexity == "easy":
            return random.uniform(0.8, 0.95)
        elif complexity == "medium":
            return random.uniform(0.6, 0.8)
        else:
            return random.uniform(0.3, 0.7)

    def generate_dataset(self, n_cases: int = 100) -> List[Dict]:
        """Generate full dataset"""
        complexities = (
            ['easy'] * int(n_cases * 0.4) +
            ['medium'] * int(n_cases * 0.4) +
            ['hard'] * int(n_cases * 0.2)
        )
        random.shuffle(complexities)

        cases = []
        for i, complexity in enumerate(complexities):
            cases.append(self.generate_case(i, complexity))

        return cases


def generate_all_verticals(n_cases_per_vertical: int = 100, output_dir: str = "data/synthetic"):
    """Generate synthetic data for all verticals"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== QURE Synthetic Data Generator ===\n")

    # Finance
    print(f"Generating {n_cases_per_vertical} finance cases...")
    finance_gen = FinanceDataGenerator()
    finance_cases = finance_gen.generate_dataset(n_cases_per_vertical)

    finance_path = output_path / "finance_reconciliation_cases.json"
    with open(finance_path, 'w') as f:
        json.dump(finance_cases, f, indent=2)

    print(f"[OK] Saved to {finance_path}")
    print(f"  - Easy: {sum(1 for c in finance_cases if c['context']['complexity'] == 'easy')}")
    print(f"  - Medium: {sum(1 for c in finance_cases if c['context']['complexity'] == 'medium')}")
    print(f"  - Hard: {sum(1 for c in finance_cases if c['context']['complexity'] == 'hard')}\n")

    # Insurance
    print(f"Generating {n_cases_per_vertical} insurance cases...")
    insurance_gen = InsuranceDataGenerator()
    insurance_cases = insurance_gen.generate_dataset(n_cases_per_vertical)

    insurance_path = output_path / "insurance_subrogation_cases.json"
    with open(insurance_path, 'w') as f:
        json.dump(insurance_cases, f, indent=2)

    print(f"[OK] Saved to {insurance_path}")
    print(f"  - Easy: {sum(1 for c in insurance_cases if c['context']['complexity'] == 'easy')}")
    print(f"  - Medium: {sum(1 for c in insurance_cases if c['context']['complexity'] == 'medium')}")
    print(f"  - Hard: {sum(1 for c in insurance_cases if c['context']['complexity'] == 'hard')}\n")

    # Healthcare
    print(f"Generating {n_cases_per_vertical} healthcare cases...")
    healthcare_gen = HealthcareDataGenerator()
    healthcare_cases = healthcare_gen.generate_dataset(n_cases_per_vertical)

    healthcare_path = output_path / "healthcare_prior_auth_cases.json"
    with open(healthcare_path, 'w') as f:
        json.dump(healthcare_cases, f, indent=2)

    print(f"[OK] Saved to {healthcare_path}")
    print(f"  - Easy: {sum(1 for c in healthcare_cases if c['context']['complexity'] == 'easy')}")
    print(f"  - Medium: {sum(1 for c in healthcare_cases if c['context']['complexity'] == 'medium')}")
    print(f"  - Hard: {sum(1 for c in healthcare_cases if c['context']['complexity'] == 'hard')}\n")

    # Summary statistics
    print("=== Summary ===")
    print(f"Total cases generated: {len(finance_cases) + len(insurance_cases) + len(healthcare_cases)}")
    print(f"Output directory: {output_path.absolute()}")

    return {
        'finance': finance_cases,
        'insurance': insurance_cases,
        'healthcare': healthcare_cases
    }


if __name__ == "__main__":
    generate_all_verticals(n_cases_per_vertical=100)
