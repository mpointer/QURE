"""
Synthetic Finance Data Generator

Generates realistic GL↔Bank reconciliation test cases with various scenarios.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)


class FinanceDataGenerator:
    """Generate synthetic GL and Bank transaction data"""

    def __init__(self):
        self.companies = [
            # Tech & Software
            "CloudTech Solutions Inc", "DataSync Systems LLC", "NextGen Software Corp",
            "Quantum Computing Associates", "CyberSecure Networks Ltd",
            # Manufacturing
            "Precision Manufacturing Group", "American Steel & Alloys Corp",
            "Industrial Components Supply Inc", "Advanced Robotics Solutions",
            # Healthcare
            "MediCare Health Services LLC", "BioPharm Research Corporation",
            "Regional Medical Suppliers Inc", "HealthFirst Diagnostics Group",
            # Retail & Consumer
            "Metropolitan Retail Holdings", "EcoFriendly Products Co",
            "Premium Home Furnishings Inc", "Urban Fashion Distributors LLC",
            # Professional Services
            "Strategic Consulting Partners", "Apex Financial Advisors Inc",
            "Global Legal Associates LLC", "Corporate Training Solutions",
            # Energy & Utilities
            "Renewable Energy Systems Corp", "National Power & Gas LLC",
            "SolarTech Installations Inc", "Energy Management Solutions",
            # Logistics
            "Express Freight Services LLC", "Continental Shipping Corp",
            "Warehousing Solutions International", "Supply Chain Innovations Inc",
            # Food & Beverage
            "Artisan Coffee Roasters Inc", "Organic Foods Distribution LLC",
            "Restaurant Supply Group Corp", "Gourmet Catering Services",
            # Real Estate
            "Metro Property Management LLC", "Commercial Real Estate Partners",
            "Skyline Construction & Development", "Residential Services Group Inc"
        ]

        self.banks = [
            "Wells Fargo Bank N.A.", "JPMorgan Chase Bank", "Bank of America N.A.",
            "Citibank N.A.", "U.S. Bank National Association", "PNC Bank N.A.",
            "Capital One N.A.", "TD Bank N.A.", "Truist Bank", "Fifth Third Bank",
            "BMO Harris Bank N.A.", "KeyBank National Association"
        ]

        self.invoice_prefixes = ["INV", "PO", "SO", "ORD", "REF", "BILL", "ACH", "PMT"]

        self.payment_types = {
            "ACH": ["ACH Credit", "ACH Debit", "Direct Deposit", "Electronic Transfer"],
            "WIRE": ["Wire Transfer", "International Wire", "Domestic Wire", "Same-Day Wire"],
            "CHECK": ["Business Check", "Cashier's Check", "Company Check"],
            "CARD": ["Corporate Card", "Business Debit", "Procurement Card"]
        }

        self.memo_templates = [
            "Invoice payment {ref} - {description}",
            "Payment for invoice {ref}",
            "Wire transfer {ref} via {bank}",
            "ACH payment {ref} - {description}",
            "Electronic payment {ref}",
            "{payment_type} {ref}",
            "Payment ref {ref} for {description}",
        ]

        self.payment_descriptions = [
            "Professional services", "Equipment purchase", "Monthly subscription",
            "Consulting fees", "Software licensing", "Hardware maintenance",
            "Annual contract payment", "Q1 service fees", "Material supplies",
            "Installation services", "Training and support", "Project milestone payment"
        ]

    def generate_dataset(self, n_cases: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate dataset with various scenarios

        Args:
            n_cases: Number of test cases

        Returns:
            Dict with 'gl_transactions', 'bank_transactions', 'expected_matches'
        """
        gl_transactions = []
        bank_transactions = []
        expected_matches = []

        # Case types distribution
        perfect_matches = int(n_cases * 0.40)  # 40% perfect matches
        close_matches = int(n_cases * 0.30)    # 30% close matches (need review)
        mismatches = int(n_cases * 0.20)       # 20% mismatches
        edge_cases = n_cases - perfect_matches - close_matches - mismatches  # 10% edge cases

        case_id = 1

        # Generate perfect matches
        for i in range(perfect_matches):
            gl, bank, match = self._generate_perfect_match(case_id)
            gl_transactions.append(gl)
            bank_transactions.append(bank)
            expected_matches.append(match)
            case_id += 1

        # Generate close matches
        for i in range(close_matches):
            gl, bank, match = self._generate_close_match(case_id)
            gl_transactions.append(gl)
            bank_transactions.append(bank)
            expected_matches.append(match)
            case_id += 1

        # Generate mismatches
        for i in range(mismatches):
            gl, bank, match = self._generate_mismatch(case_id)
            gl_transactions.append(gl)
            bank_transactions.append(bank)
            expected_matches.append(match)
            case_id += 1

        # Generate edge cases
        for i in range(edge_cases):
            gl, bank, match = self._generate_edge_case(case_id)
            gl_transactions.append(gl)
            bank_transactions.append(bank)
            expected_matches.append(match)
            case_id += 1

        return {
            "gl_transactions": gl_transactions,
            "bank_transactions": bank_transactions,
            "expected_matches": expected_matches,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_cases": n_cases,
                "perfect_matches": perfect_matches,
                "close_matches": close_matches,
                "mismatches": mismatches,
                "edge_cases": edge_cases,
            }
        }

    def _generate_perfect_match(self, case_id: int) -> tuple:
        """Generate perfect match scenario"""
        base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        company = random.choice(self.companies)
        amount = round(random.uniform(100, 50000), 2)
        invoice_ref = f"{random.choice(self.invoice_prefixes)}-2024-{case_id:04d}"
        bank_name = random.choice(self.banks)
        description = random.choice(self.payment_descriptions)

        # Determine transaction type
        transaction_type = "WIRE" if amount > 10000 else "ACH"
        payment_type_detail = random.choice(self.payment_types[transaction_type])

        # GL transaction
        gl = {
            "id": f"GL_{base_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": base_date.strftime("%Y-%m-%d"),
            "amount": amount,
            "currency": "USD",
            "payer": company,
            "memo": f"{description} - Invoice {invoice_ref}",
            "reference": invoice_ref,
            "status": "unreconciled",
            "account": "1200 - Accounts Receivable",
            "created_at": base_date.isoformat(),
            "payment_method": transaction_type,
        }

        # Bank transaction (same day or next day)
        bank_date = base_date + timedelta(days=random.choice([0, 1]))
        bank = {
            "id": f"BANK_{bank_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": bank_date.strftime("%Y-%m-%d"),
            "amount": amount,
            "currency": "USD",
            "payer": company.upper(),  # Slight variation in casing
            "memo": f"{payment_type_detail} - Ref {invoice_ref}",
            "swift_ref": f"SWIFT{random.randint(100000, 999999)}" if amount > 10000 else None,
            "transaction_type": transaction_type,
            "bank_name": bank_name,
            "bank_account": f"{random.randint(1000, 9999)}****{random.randint(1000, 9999)}",
            "routing_number": f"0{random.randint(10000000, 99999999)}",
            "trace_number": f"TRC{random.randint(100000000000, 999999999999)}" if transaction_type == "ACH" else None,
        }

        # Expected match
        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "gl_id": gl["id"],
            "bank_id": bank["id"],
            "expected_decision": "auto_resolve",
            "match_score": 1.0,
            "notes": "Perfect match - same amount, dates within 1 day, payer names match",
        }

        return gl, bank, match

    def _generate_close_match(self, case_id: int) -> tuple:
        """Generate close match scenario (requires review)"""
        base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        company = random.choice(self.companies)
        amount = round(random.uniform(100, 50000), 2)
        invoice_ref = f"{random.choice(self.invoice_prefixes)}-2024-{case_id:04d}"
        bank_name = random.choice(self.banks)
        description = random.choice(self.payment_descriptions)

        # Introduce a small variation
        variation_type = random.choice([
            "date_difference",  # 2-3 days apart
            "payer_typo",       # Typo in payer name
            "amount_cents",     # Off by cents
            "memo_different",   # Different memo format
        ])

        transaction_type = "WIRE" if amount > 10000 else "ACH"
        payment_type_detail = random.choice(self.payment_types[transaction_type])

        gl = {
            "id": f"GL_{base_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": base_date.strftime("%Y-%m-%d"),
            "amount": amount,
            "currency": "USD",
            "payer": company,
            "memo": f"{description} - Invoice {invoice_ref}",
            "reference": invoice_ref,
            "status": "unreconciled",
            "account": "1200 - Accounts Receivable",
            "created_at": base_date.isoformat(),
            "payment_method": transaction_type,
        }

        # Apply variation
        if variation_type == "date_difference":
            bank_date = base_date + timedelta(days=random.randint(2, 3))
        else:
            bank_date = base_date + timedelta(days=1)

        if variation_type == "payer_typo":
            bank_payer = company.replace("Corp", "Corporation").replace("Inc", "Incorporated").replace("LLC", "L.L.C.")
        else:
            bank_payer = company.upper()

        if variation_type == "amount_cents":
            bank_amount = amount + random.choice([0.01, -0.01, 0.02])
        else:
            bank_amount = amount

        if variation_type == "memo_different":
            bank_memo = f"{payment_type_detail} - {description}"
        else:
            bank_memo = f"{payment_type_detail} - Ref {invoice_ref}"

        bank = {
            "id": f"BANK_{bank_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": bank_date.strftime("%Y-%m-%d"),
            "amount": bank_amount,
            "currency": "USD",
            "payer": bank_payer,
            "memo": bank_memo,
            "swift_ref": f"SWIFT{random.randint(100000, 999999)}" if amount > 10000 else None,
            "transaction_type": transaction_type,
            "bank_name": bank_name,
            "bank_account": f"{random.randint(1000, 9999)}****{random.randint(1000, 9999)}",
            "routing_number": f"0{random.randint(10000000, 99999999)}",
            "trace_number": f"TRC{random.randint(100000000000, 999999999999)}" if transaction_type == "ACH" else None,
        }

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "gl_id": gl["id"],
            "bank_id": bank["id"],
            "expected_decision": "hitl_review",
            "match_score": 0.75,
            "notes": f"Close match with {variation_type} - requires human review",
        }

        return gl, bank, match

    def _generate_mismatch(self, case_id: int) -> tuple:
        """Generate mismatch scenario"""
        base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        company1 = random.choice(self.companies)
        company2 = random.choice([c for c in self.companies if c != company1])
        amount1 = round(random.uniform(100, 50000), 2)
        amount2 = round(random.uniform(100, 50000), 2)
        invoice_ref1 = f"{random.choice(self.invoice_prefixes)}-2024-{case_id:04d}A"
        invoice_ref2 = f"{random.choice(self.invoice_prefixes)}-2024-{case_id:04d}B"
        bank_name = random.choice(self.banks)
        description1 = random.choice(self.payment_descriptions)
        description2 = random.choice(self.payment_descriptions)

        transaction_type = "ACH"
        payment_type_detail = random.choice(self.payment_types[transaction_type])

        gl = {
            "id": f"GL_{base_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": base_date.strftime("%Y-%m-%d"),
            "amount": amount1,
            "currency": "USD",
            "payer": company1,
            "memo": f"{description1} - Invoice {invoice_ref1}",
            "reference": invoice_ref1,
            "status": "unreconciled",
            "account": "1200 - Accounts Receivable",
            "created_at": base_date.isoformat(),
            "payment_method": transaction_type,
        }

        bank_date = base_date + timedelta(days=random.randint(5, 10))
        bank = {
            "id": f"BANK_{bank_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": bank_date.strftime("%Y-%m-%d"),
            "amount": amount2,
            "currency": "USD",
            "payer": company2.upper(),
            "memo": f"{payment_type_detail} - {description2} - Ref {invoice_ref2}",
            "swift_ref": None,
            "transaction_type": transaction_type,
            "bank_name": bank_name,
            "bank_account": f"{random.randint(1000, 9999)}****{random.randint(1000, 9999)}",
            "routing_number": f"0{random.randint(10000000, 99999999)}",
            "trace_number": f"TRC{random.randint(100000000000, 999999999999)}",
        }

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "gl_id": gl["id"],
            "bank_id": bank["id"],
            "expected_decision": "reject",
            "match_score": 0.2,
            "notes": "Mismatch - different payers, amounts, and dates",
        }

        return gl, bank, match

    def _generate_edge_case(self, case_id: int) -> tuple:
        """Generate edge case scenario"""
        base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        company = random.choice(self.companies)
        amount = round(random.uniform(100, 100000), 2)
        invoice_ref = f"{random.choice(self.invoice_prefixes)}-2024-{case_id:04d}"
        bank_name = random.choice(self.banks)
        description = random.choice(self.payment_descriptions)

        edge_case_type = random.choice([
            "high_value_no_swift",      # > $10k but missing SWIFT
            "currency_mismatch",        # Different currencies
            "duplicate_suspicion",      # Similar to another transaction
            "compliance_flag",          # Needs additional evidence
        ])

        transaction_type = "WIRE" if amount > 10000 else "ACH"
        payment_type_detail = random.choice(self.payment_types[transaction_type])

        gl = {
            "id": f"GL_{base_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": base_date.strftime("%Y-%m-%d"),
            "amount": amount if edge_case_type != "currency_mismatch" else amount,
            "currency": "USD" if edge_case_type != "currency_mismatch" else "USD",
            "payer": company,
            "memo": f"{description} - Invoice {invoice_ref}",
            "reference": invoice_ref,
            "status": "unreconciled",
            "account": "1200 - Accounts Receivable",
            "created_at": base_date.isoformat(),
            "payment_method": transaction_type,
        }

        bank_date = base_date + timedelta(days=1)
        bank = {
            "id": f"BANK_{bank_date.strftime('%Y%m%d')}_{case_id:03d}",
            "date": bank_date.strftime("%Y-%m-%d"),
            "amount": amount,
            "currency": "EUR" if edge_case_type == "currency_mismatch" else "USD",
            "payer": company.upper(),
            "memo": f"{payment_type_detail} - Ref {invoice_ref}",
            "swift_ref": None if edge_case_type == "high_value_no_swift" and amount > 10000 else (f"SWIFT{random.randint(100000, 999999)}" if amount > 10000 else None),
            "transaction_type": transaction_type,
            "bank_name": bank_name,
            "bank_account": f"{random.randint(1000, 9999)}****{random.randint(1000, 9999)}",
            "routing_number": f"0{random.randint(10000000, 99999999)}",
            "trace_number": f"TRC{random.randint(100000000000, 999999999999)}" if transaction_type == "ACH" else None,
        }

        if edge_case_type == "high_value_no_swift":
            gl["amount"] = 15000.00
            bank["amount"] = 15000.00
            expected_decision = "request_evidence"
            match_score = 0.5
            notes = "High-value transaction missing SWIFT reference - SOX compliance issue"
        elif edge_case_type == "currency_mismatch":
            expected_decision = "reject"
            match_score = 0.0
            notes = "Currency mismatch - cannot reconcile USD vs EUR"
        elif edge_case_type == "compliance_flag":
            gl["amount"] = 60000.00
            bank["amount"] = 60000.00
            expected_decision = "escalate"
            match_score = 0.7
            notes = "High-value transaction requires manager approval"
        else:
            expected_decision = "hitl_review"
            match_score = 0.6
            notes = "Potential duplicate - requires investigation"

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "gl_id": gl["id"],
            "bank_id": bank["id"],
            "expected_decision": expected_decision,
            "match_score": match_score,
            "notes": notes,
        }

        return gl, bank, match

    def save_dataset(self, dataset: Dict[str, Any], output_dir: Path):
        """Save dataset to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save GL transactions
        gl_file = output_dir / "gl_transactions.json"
        with open(gl_file, "w") as f:
            json.dump(dataset["gl_transactions"], f, indent=2)

        # Save Bank transactions
        bank_file = output_dir / "bank_transactions.json"
        with open(bank_file, "w") as f:
            json.dump(dataset["bank_transactions"], f, indent=2)

        # Save expected matches
        matches_file = output_dir / "expected_matches.json"
        with open(matches_file, "w") as f:
            json.dump(dataset["expected_matches"], f, indent=2)

        # Save metadata
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

        print(f"[OK] Dataset saved to {output_dir}")
        print(f"   - GL transactions: {len(dataset['gl_transactions'])}")
        print(f"   - Bank transactions: {len(dataset['bank_transactions'])}")
        print(f"   - Expected matches: {len(dataset['expected_matches'])}")


def main():
    """Generate synthetic finance data"""
    print("Generating synthetic GL<->Bank reconciliation data...")

    generator = FinanceDataGenerator()
    dataset = generator.generate_dataset(n_cases=20)

    # Save to data/synthetic/finance
    output_dir = Path(__file__).parent / "finance"
    generator.save_dataset(dataset, output_dir)

    # Print summary
    print("\nDataset Summary:")
    print(f"  Perfect Matches: {dataset['metadata']['perfect_matches']} (40%)")
    print(f"  Close Matches: {dataset['metadata']['close_matches']} (30%)")
    print(f"  Mismatches: {dataset['metadata']['mismatches']} (20%)")
    print(f"  Edge Cases: {dataset['metadata']['edge_cases']} (10%)")
    print("\nExpected Auto-Resolution Rate: ~40%")
    print("Expected Human Review Rate: ~40%")
    print("Expected Rejection Rate: ~20%")


if __name__ == "__main__":
    main()
