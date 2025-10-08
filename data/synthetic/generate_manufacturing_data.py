"""
Synthetic Manufacturing Data Generator

Generates realistic supply chain reconciliation test cases with various scenarios.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)


class ManufacturingDataGenerator:
    """Generate synthetic manufacturing supply chain reconciliation data"""

    def __init__(self):
        self.suppliers = [
            "Pacific Components Ltd", "Global Electronics Inc", "Premium Metals Corp",
            "Reliable Plastics LLC", "Quality Fasteners Co", "Advanced Materials Group",
            "Standard Parts Supply", "Premier Industrial Solutions"
        ]

        self.part_categories = [
            {"category": "Electronic Components", "prefix": "EC", "typical_qty": 5000},
            {"category": "Metal Parts", "prefix": "MP", "typical_qty": 2000},
            {"category": "Plastic Components", "prefix": "PC", "typical_qty": 10000},
            {"category": "Fasteners", "prefix": "FT", "typical_qty": 50000},
            {"category": "Raw Materials", "prefix": "RM", "typical_qty": 1000},
        ]

        self.discrepancy_reasons = [
            "Partial shipment",
            "Damaged in transit",
            "Receiving dock error",
            "Invoice quantity mismatch",
            "Quality rejection",
            "Supplier short ship",
            "Data entry error",
        ]

    def generate_dataset(self, n_cases: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate dataset with various scenarios

        Args:
            n_cases: Number of test cases

        Returns:
            Dict with 'purchase_orders', 'shipment_receipts', 'expected_matches'
        """
        purchase_orders = []
        shipment_receipts = []
        expected_matches = []

        # Case types distribution
        exact_matches = int(n_cases * 0.35)      # 35% exact matches
        partial_receipt = int(n_cases * 0.30)    # 30% partial receipts (common in manufacturing)
        variance = int(n_cases * 0.25)           # 25% variance (requires investigation)
        critical = n_cases - exact_matches - partial_receipt - variance  # 10% critical

        case_id = 1

        # Generate exact matches
        for i in range(exact_matches):
            po, receipt, match = self._generate_exact_match(case_id)
            purchase_orders.append(po)
            shipment_receipts.append(receipt)
            expected_matches.append(match)
            case_id += 1

        # Generate partial receipts
        for i in range(partial_receipt):
            po, receipt, match = self._generate_partial_receipt(case_id)
            purchase_orders.append(po)
            shipment_receipts.append(receipt)
            expected_matches.append(match)
            case_id += 1

        # Generate variance cases
        for i in range(variance):
            po, receipt, match = self._generate_variance(case_id)
            purchase_orders.append(po)
            shipment_receipts.append(receipt)
            expected_matches.append(match)
            case_id += 1

        # Generate critical cases
        for i in range(critical):
            po, receipt, match = self._generate_critical(case_id)
            purchase_orders.append(po)
            shipment_receipts.append(receipt)
            expected_matches.append(match)
            case_id += 1

        return {
            "purchase_orders": purchase_orders,
            "shipment_receipts": shipment_receipts,
            "expected_matches": expected_matches,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_cases": n_cases,
                "exact_matches": exact_matches,
                "partial_receipt": partial_receipt,
                "variance": variance,
                "critical": critical,
            }
        }

    def _generate_exact_match(self, case_id: int) -> tuple:
        """Generate exact match scenario - PO and receipt match perfectly"""
        order_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.part_categories)
        supplier = random.choice(self.suppliers)

        part_number = f"{category['prefix']}-{2024000 + case_id}"
        part_description = f"{category['category']} Part {case_id}"
        quantity = random.randint(100, category['typical_qty'])
        unit_price = round(random.uniform(0.50, 50.0), 2)

        # Purchase Order
        po = {
            "id": f"PO_{order_date.strftime('%Y%m%d')}_{case_id:03d}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "po_number": f"PO-2024-{case_id:04d}",
            "supplier": supplier,
            "part_number": part_number,
            "part_description": part_description,
            "category": category['category'],
            "ordered_quantity": quantity,
            "unit_price": unit_price,
            "total_value": round(quantity * unit_price, 2),
            "expected_delivery": (order_date + timedelta(days=random.randint(7, 14))).strftime("%Y-%m-%d"),
            "status": "open",
        }

        # Shipment Receipt (delivered on expected date or slightly before/after)
        receipt_date = datetime.strptime(po["expected_delivery"], "%Y-%m-%d") + timedelta(days=random.choice([-1, 0, 1]))
        receipt = {
            "id": f"REC_{receipt_date.strftime('%Y%m%d')}_{case_id:03d}",
            "receipt_date": receipt_date.strftime("%Y-%m-%d"),
            "po_number": po["po_number"],
            "supplier": supplier.upper(),  # Slight variation in casing
            "part_number": part_number,
            "part_description": part_description.upper(),
            "received_quantity": quantity,  # Exact match
            "receiver_name": f"Dock_{random.randint(1, 5)}",
            "condition": "Good",
            "notes": "Complete shipment received in good condition",
        }

        # Expected match
        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "po_id": po["id"],
            "receipt_id": receipt["id"],
            "expected_decision": "auto_resolve",
            "match_score": 1.0,
            "quantity_variance": 0,
            "variance_pct": 0.0,
            "notes": "Exact match - complete shipment received",
        }

        return po, receipt, match

    def _generate_partial_receipt(self, case_id: int) -> tuple:
        """Generate partial receipt scenario - common in manufacturing"""
        order_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.part_categories)
        supplier = random.choice(self.suppliers)

        part_number = f"{category['prefix']}-{2024000 + case_id}"
        part_description = f"{category['category']} Part {case_id}"
        ordered_quantity = random.randint(100, category['typical_qty'])
        unit_price = round(random.uniform(0.50, 50.0), 2)

        # Partial receipt (50-90% of ordered quantity)
        receipt_pct = random.uniform(0.50, 0.90)
        received_quantity = int(ordered_quantity * receipt_pct)

        po = {
            "id": f"PO_{order_date.strftime('%Y%m%d')}_{case_id:03d}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "po_number": f"PO-2024-{case_id:04d}",
            "supplier": supplier,
            "part_number": part_number,
            "part_description": part_description,
            "category": category['category'],
            "ordered_quantity": ordered_quantity,
            "unit_price": unit_price,
            "total_value": round(ordered_quantity * unit_price, 2),
            "expected_delivery": (order_date + timedelta(days=random.randint(7, 14))).strftime("%Y-%m-%d"),
            "status": "open",
        }

        receipt_date = datetime.strptime(po["expected_delivery"], "%Y-%m-%d") + timedelta(days=random.choice([0, 1]))
        receipt = {
            "id": f"REC_{receipt_date.strftime('%Y%m%d')}_{case_id:03d}",
            "receipt_date": receipt_date.strftime("%Y-%m-%d"),
            "po_number": po["po_number"],
            "supplier": supplier.upper(),
            "part_number": part_number,
            "part_description": part_description.upper(),
            "received_quantity": received_quantity,
            "receiver_name": f"Dock_{random.randint(1, 5)}",
            "condition": "Good",
            "notes": f"Partial shipment - supplier indicates remaining {ordered_quantity - received_quantity} units on backorder",
        }

        actual_variance_pct = (received_quantity - ordered_quantity) / ordered_quantity * 100

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "po_id": po["id"],
            "receipt_id": receipt["id"],
            "expected_decision": "hitl_review",
            "match_score": 0.75,
            "quantity_variance": received_quantity - ordered_quantity,
            "variance_pct": round(actual_variance_pct, 2),
            "notes": f"Partial shipment - {abs(actual_variance_pct):.1f}% short. Remaining quantity expected on backorder.",
        }

        return po, receipt, match

    def _generate_variance(self, case_id: int) -> tuple:
        """Generate variance scenario - discrepancy requires investigation"""
        order_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.part_categories)
        supplier = random.choice(self.suppliers)

        part_number = f"{category['prefix']}-{2024000 + case_id}"
        part_description = f"{category['category']} Part {case_id}"
        ordered_quantity = random.randint(100, category['typical_qty'])
        unit_price = round(random.uniform(0.50, 50.0), 2)

        # Introduce variance (could be over or under)
        variance_type = random.choice(["short", "over", "damaged"])

        if variance_type == "short":
            # 5-15% short
            variance_pct = random.uniform(0.05, 0.15)
            received_quantity = int(ordered_quantity * (1 - variance_pct))
            condition = "Good"
            reason = random.choice(["Supplier short ship", "Receiving dock error"])
        elif variance_type == "over":
            # 5-10% over (less common)
            variance_pct = random.uniform(0.05, 0.10)
            received_quantity = int(ordered_quantity * (1 + variance_pct))
            condition = "Good"
            reason = "Invoice quantity mismatch - received more than ordered"
        else:  # damaged
            # Some units damaged
            damage_pct = random.uniform(0.05, 0.10)
            damaged_qty = int(ordered_quantity * damage_pct)
            received_quantity = ordered_quantity - damaged_qty
            condition = "Partial Damage"
            reason = f"Damaged in transit - {damaged_qty} units rejected"

        po = {
            "id": f"PO_{order_date.strftime('%Y%m%d')}_{case_id:03d}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "po_number": f"PO-2024-{case_id:04d}",
            "supplier": supplier,
            "part_number": part_number,
            "part_description": part_description,
            "category": category['category'],
            "ordered_quantity": ordered_quantity,
            "unit_price": unit_price,
            "total_value": round(ordered_quantity * unit_price, 2),
            "expected_delivery": (order_date + timedelta(days=random.randint(7, 14))).strftime("%Y-%m-%d"),
            "status": "open",
        }

        receipt_date = datetime.strptime(po["expected_delivery"], "%Y-%m-%d") + timedelta(days=random.choice([0, 1]))
        receipt = {
            "id": f"REC_{receipt_date.strftime('%Y%m%d')}_{case_id:03d}",
            "receipt_date": receipt_date.strftime("%Y-%m-%d"),
            "po_number": po["po_number"],
            "supplier": supplier.upper(),
            "part_number": part_number,
            "part_description": part_description.upper(),
            "received_quantity": received_quantity,
            "receiver_name": f"Dock_{random.randint(1, 5)}",
            "condition": condition,
            "notes": f"Variance detected - {reason}",
        }

        actual_variance_pct = (received_quantity - ordered_quantity) / ordered_quantity * 100

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "po_id": po["id"],
            "receipt_id": receipt["id"],
            "expected_decision": "hitl_review",
            "match_score": 0.60,
            "quantity_variance": received_quantity - ordered_quantity,
            "variance_pct": round(actual_variance_pct, 2),
            "notes": f"Variance ({abs(actual_variance_pct):.1f}%) - {reason}. Requires investigation and supplier follow-up.",
        }

        return po, receipt, match

    def _generate_critical(self, case_id: int) -> tuple:
        """Generate critical scenario - severe issues requiring escalation"""
        order_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.part_categories)
        supplier = random.choice(self.suppliers)

        part_number = f"{category['prefix']}-{2024000 + case_id}"
        part_description = f"{category['category']} Part {case_id}"
        ordered_quantity = random.randint(100, category['typical_qty'])
        unit_price = round(random.uniform(0.50, 50.0), 2)

        critical_type = random.choice([
            "non_delivery",
            "wrong_parts",
            "quality_rejection",
            "severe_shortage",
        ])

        if critical_type == "non_delivery":
            # Nothing received
            received_quantity = 0
            condition = "N/A"
            reason = "Shipment not received - supplier claims shipped"
            decision = "escalate"

        elif critical_type == "wrong_parts":
            # Wrong part number received
            received_quantity = ordered_quantity
            condition = "Wrong Part"
            reason = "Wrong part number received - supplier shipping error"
            decision = "reject"

        elif critical_type == "quality_rejection":
            # All units failed QC
            received_quantity = 0
            condition = "Rejected"
            reason = "All units failed quality inspection - 100% rejection"
            decision = "reject"

        else:  # severe_shortage
            # >30% short
            shortage_pct = random.uniform(0.30, 0.50)
            received_quantity = int(ordered_quantity * (1 - shortage_pct))
            condition = "Good"
            reason = f"Severe shortage - {int(shortage_pct * 100)}% short with no explanation from supplier"
            decision = "escalate"

        po = {
            "id": f"PO_{order_date.strftime('%Y%m%d')}_{case_id:03d}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "po_number": f"PO-2024-{case_id:04d}",
            "supplier": supplier,
            "part_number": part_number,
            "part_description": part_description,
            "category": category['category'],
            "ordered_quantity": ordered_quantity,
            "unit_price": unit_price,
            "total_value": round(ordered_quantity * unit_price, 2),
            "expected_delivery": (order_date + timedelta(days=random.randint(7, 14))).strftime("%Y-%m-%d"),
            "status": "open",
        }

        receipt_date = datetime.strptime(po["expected_delivery"], "%Y-%m-%d") + timedelta(days=random.randint(0, 5))
        receipt = {
            "id": f"REC_{receipt_date.strftime('%Y%m%d')}_{case_id:03d}",
            "receipt_date": receipt_date.strftime("%Y-%m-%d"),
            "po_number": po["po_number"],
            "supplier": supplier.upper(),
            "part_number": part_number if critical_type != "wrong_parts" else f"{category['prefix']}-9999{case_id}",
            "part_description": part_description.upper(),
            "received_quantity": received_quantity,
            "receiver_name": f"Dock_{random.randint(1, 5)}",
            "condition": condition,
            "notes": f"CRITICAL: {reason}",
        }

        if ordered_quantity > 0:
            actual_variance_pct = (received_quantity - ordered_quantity) / ordered_quantity * 100
        else:
            actual_variance_pct = -100.0

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "po_id": po["id"],
            "receipt_id": receipt["id"],
            "expected_decision": decision,
            "match_score": 0.15,
            "quantity_variance": received_quantity - ordered_quantity,
            "variance_pct": round(actual_variance_pct, 2),
            "notes": f"CRITICAL: {reason} ({abs(actual_variance_pct):.1f}% variance). Immediate procurement and supplier management escalation required.",
        }

        return po, receipt, match

    def save_dataset(self, dataset: Dict[str, Any], output_dir: Path):
        """Save dataset to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save purchase orders
        po_file = output_dir / "purchase_orders.json"
        with open(po_file, "w") as f:
            json.dump(dataset["purchase_orders"], f, indent=2)

        # Save shipment receipts
        receipt_file = output_dir / "shipment_receipts.json"
        with open(receipt_file, "w") as f:
            json.dump(dataset["shipment_receipts"], f, indent=2)

        # Save expected matches
        matches_file = output_dir / "expected_matches.json"
        with open(matches_file, "w") as f:
            json.dump(dataset["expected_matches"], f, indent=2)

        # Save metadata
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

        print(f"[OK] Dataset saved to {output_dir}")
        print(f"   - Purchase orders: {len(dataset['purchase_orders'])}")
        print(f"   - Shipment receipts: {len(dataset['shipment_receipts'])}")
        print(f"   - Expected matches: {len(dataset['expected_matches'])}")


def main():
    """Generate synthetic manufacturing supply chain data"""
    print("Generating synthetic manufacturing supply chain reconciliation data...")

    generator = ManufacturingDataGenerator()
    dataset = generator.generate_dataset(n_cases=20)

    # Save to data/synthetic/manufacturing
    output_dir = Path(__file__).parent / "manufacturing"
    generator.save_dataset(dataset, output_dir)

    # Print summary
    print("\nDataset Summary:")
    print(f"  Exact Matches: {dataset['metadata']['exact_matches']} (35%)")
    print(f"  Partial Receipts: {dataset['metadata']['partial_receipt']} (30%)")
    print(f"  Variance Cases: {dataset['metadata']['variance']} (25%)")
    print(f"  Critical Cases: {dataset['metadata']['critical']} (10%)")
    print("\nExpected Auto-Resolution Rate: ~35%")
    print("Expected Human Review Rate: ~55%")
    print("Expected Escalation/Rejection Rate: ~10%")


if __name__ == "__main__":
    main()
