"""
Synthetic Retail Data Generator

Generates realistic inventory reconciliation test cases with various scenarios.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)


class RetailDataGenerator:
    """Generate synthetic retail inventory reconciliation data"""

    def __init__(self):
        self.product_categories = [
            {"category": "Electronics", "prefix": "ELEC", "typical_count": 50},
            {"category": "Clothing", "prefix": "CLTH", "typical_count": 200},
            {"category": "Home Goods", "prefix": "HOME", "typical_count": 100},
            {"category": "Toys", "prefix": "TOYS", "typical_count": 150},
            {"category": "Books", "prefix": "BOOK", "typical_count": 300},
            {"category": "Sports Equipment", "prefix": "SPRT", "typical_count": 75},
            {"category": "Groceries", "prefix": "GROC", "typical_count": 500},
        ]

        self.store_locations = [
            "Store-001-NYC", "Store-002-LA", "Store-003-CHI", "Store-004-HOU",
            "Store-005-PHX", "Store-006-SF", "Store-007-SEA", "Store-008-MIA",
        ]

        self.discrepancy_reasons = [
            "Theft/Shrinkage",
            "Receiving error",
            "Damaged goods",
            "Miscount during physical inventory",
            "System entry error",
            "Returns not logged",
            "Transfer not recorded",
        ]

    def generate_dataset(self, n_cases: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate dataset with various scenarios

        Args:
            n_cases: Number of test cases

        Returns:
            Dict with 'pos_inventory', 'physical_count', 'expected_matches'
        """
        pos_inventory = []
        physical_count = []
        expected_matches = []

        # Case types distribution
        exact_matches = int(n_cases * 0.40)      # 40% exact matches
        minor_variance = int(n_cases * 0.30)     # 30% minor variance (acceptable)
        major_variance = int(n_cases * 0.20)     # 20% major variance (needs investigation)
        critical = n_cases - exact_matches - minor_variance - major_variance  # 10% critical

        case_id = 1

        # Generate exact matches
        for i in range(exact_matches):
            pos, physical, match = self._generate_exact_match(case_id)
            pos_inventory.append(pos)
            physical_count.append(physical)
            expected_matches.append(match)
            case_id += 1

        # Generate minor variance cases
        for i in range(minor_variance):
            pos, physical, match = self._generate_minor_variance(case_id)
            pos_inventory.append(pos)
            physical_count.append(physical)
            expected_matches.append(match)
            case_id += 1

        # Generate major variance cases
        for i in range(major_variance):
            pos, physical, match = self._generate_major_variance(case_id)
            pos_inventory.append(pos)
            physical_count.append(physical)
            expected_matches.append(match)
            case_id += 1

        # Generate critical cases
        for i in range(critical):
            pos, physical, match = self._generate_critical(case_id)
            pos_inventory.append(pos)
            physical_count.append(physical)
            expected_matches.append(match)
            case_id += 1

        return {
            "pos_inventory": pos_inventory,
            "physical_count": physical_count,
            "expected_matches": expected_matches,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_cases": n_cases,
                "exact_matches": exact_matches,
                "minor_variance": minor_variance,
                "major_variance": major_variance,
                "critical": critical,
            }
        }

    def _generate_exact_match(self, case_id: int) -> tuple:
        """Generate exact match scenario - POS and physical counts match"""
        count_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.product_categories)
        store = random.choice(self.store_locations)

        product_name = f"{category['category']} Item {case_id}"
        sku = f"{category['prefix']}-{2024000 + case_id}"
        quantity = random.randint(10, category['typical_count'])
        unit_cost = round(random.uniform(5.0, 200.0), 2)

        # POS system record
        pos = {
            "id": f"POS_{count_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": count_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name,
            "category": category['category'],
            "quantity_on_hand": quantity,
            "unit_cost": unit_cost,
            "total_value": round(quantity * unit_cost, 2),
            "store_location": store,
            "last_updated": count_date.isoformat(),
            "status": "active",
        }

        # Physical count (same day or next day)
        physical_date = count_date + timedelta(days=random.choice([0, 1]))
        physical = {
            "id": f"PHY_{physical_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": physical_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name.upper(),  # Slight variation in casing
            "category": category['category'],
            "counted_quantity": quantity,  # Exact match
            "counter_name": f"Counter_{random.randint(1, 5)}",
            "store_location": store,
            "verified": True,
            "notes": "Standard physical count",
        }

        # Expected match
        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "pos_id": pos["id"],
            "physical_id": physical["id"],
            "expected_decision": "auto_resolve",
            "match_score": 1.0,
            "variance": 0,
            "variance_pct": 0.0,
            "notes": "Exact match - no discrepancy",
        }

        return pos, physical, match

    def _generate_minor_variance(self, case_id: int) -> tuple:
        """Generate minor variance scenario - small acceptable difference"""
        count_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.product_categories)
        store = random.choice(self.store_locations)

        product_name = f"{category['category']} Item {case_id}"
        sku = f"{category['prefix']}-{2024000 + case_id}"
        pos_quantity = random.randint(50, category['typical_count'])
        unit_cost = round(random.uniform(5.0, 200.0), 2)

        # Introduce minor variance (1-3% difference)
        variance_type = random.choice(["under", "over"])
        variance_pct = random.uniform(0.01, 0.03)
        variance = int(pos_quantity * variance_pct)

        if variance == 0:
            variance = random.choice([1, 2])

        physical_quantity = pos_quantity - variance if variance_type == "under" else pos_quantity + variance

        pos = {
            "id": f"POS_{count_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": count_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name,
            "category": category['category'],
            "quantity_on_hand": pos_quantity,
            "unit_cost": unit_cost,
            "total_value": round(pos_quantity * unit_cost, 2),
            "store_location": store,
            "last_updated": count_date.isoformat(),
            "status": "active",
        }

        physical_date = count_date + timedelta(days=1)
        physical = {
            "id": f"PHY_{physical_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": physical_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name.upper(),
            "category": category['category'],
            "counted_quantity": physical_quantity,
            "counter_name": f"Counter_{random.randint(1, 5)}",
            "store_location": store,
            "verified": True,
            "notes": f"Minor variance detected - {variance_type}count",
        }

        actual_variance_pct = abs(physical_quantity - pos_quantity) / pos_quantity * 100

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "pos_id": pos["id"],
            "physical_id": physical["id"],
            "expected_decision": "auto_resolve",
            "match_score": 0.85,
            "variance": physical_quantity - pos_quantity,
            "variance_pct": round(actual_variance_pct, 2),
            "notes": f"Minor variance ({actual_variance_pct:.1f}%) - within acceptable tolerance",
        }

        return pos, physical, match

    def _generate_major_variance(self, case_id: int) -> tuple:
        """Generate major variance scenario - requires investigation"""
        count_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.product_categories)
        store = random.choice(self.store_locations)

        product_name = f"{category['category']} Item {case_id}"
        sku = f"{category['prefix']}-{2024000 + case_id}"
        pos_quantity = random.randint(50, category['typical_count'])
        unit_cost = round(random.uniform(5.0, 200.0), 2)

        # Introduce major variance (5-15% difference)
        variance_type = random.choice(["under", "over"])
        variance_pct = random.uniform(0.05, 0.15)
        variance = int(pos_quantity * variance_pct)

        if variance < 5:
            variance = random.randint(5, 10)

        physical_quantity = pos_quantity - variance if variance_type == "under" else pos_quantity + variance

        reason = random.choice(self.discrepancy_reasons)

        pos = {
            "id": f"POS_{count_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": count_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name,
            "category": category['category'],
            "quantity_on_hand": pos_quantity,
            "unit_cost": unit_cost,
            "total_value": round(pos_quantity * unit_cost, 2),
            "store_location": store,
            "last_updated": count_date.isoformat(),
            "status": "active",
        }

        physical_date = count_date + timedelta(days=1)
        physical = {
            "id": f"PHY_{physical_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": physical_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name.upper(),
            "category": category['category'],
            "counted_quantity": physical_quantity,
            "counter_name": f"Counter_{random.randint(1, 5)}",
            "store_location": store,
            "verified": True,
            "notes": f"Major variance - possible {reason.lower()}",
        }

        actual_variance_pct = abs(physical_quantity - pos_quantity) / pos_quantity * 100

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "pos_id": pos["id"],
            "physical_id": physical["id"],
            "expected_decision": "hitl_review",
            "match_score": 0.65,
            "variance": physical_quantity - pos_quantity,
            "variance_pct": round(actual_variance_pct, 2),
            "notes": f"Major variance ({actual_variance_pct:.1f}%) - requires investigation. Suspected: {reason}",
        }

        return pos, physical, match

    def _generate_critical(self, case_id: int) -> tuple:
        """Generate critical scenario - severe discrepancy or missing items"""
        count_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        category = random.choice(self.product_categories)
        store = random.choice(self.store_locations)

        product_name = f"{category['category']} Item {case_id}"
        sku = f"{category['prefix']}-{2024000 + case_id}"
        pos_quantity = random.randint(50, category['typical_count'])
        unit_cost = round(random.uniform(5.0, 200.0), 2)

        critical_type = random.choice([
            "severe_shortage",
            "missing_item",
            "excess_inventory",
            "sku_mismatch",
        ])

        if critical_type == "severe_shortage":
            # >20% shortage
            variance_pct = random.uniform(0.20, 0.40)
            variance = int(pos_quantity * variance_pct)
            physical_quantity = pos_quantity - variance
            reason = "Severe shortage - possible theft or unrecorded sale"

        elif critical_type == "missing_item":
            # Item not found
            physical_quantity = 0
            variance = pos_quantity
            reason = "Item not found during physical count - requires immediate investigation"

        elif critical_type == "excess_inventory":
            # >20% excess
            variance_pct = random.uniform(0.20, 0.40)
            variance = int(pos_quantity * variance_pct)
            physical_quantity = pos_quantity + variance
            reason = "Excess inventory - possible receiving error or system glitch"

        else:  # sku_mismatch
            # Wrong SKU scanned
            physical_quantity = random.randint(10, 100)
            variance = physical_quantity - pos_quantity
            reason = "SKU mismatch - physical count may be for different product"

        pos = {
            "id": f"POS_{count_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": count_date.strftime("%Y-%m-%d"),
            "sku": sku,
            "product_name": product_name,
            "category": category['category'],
            "quantity_on_hand": pos_quantity,
            "unit_cost": unit_cost,
            "total_value": round(pos_quantity * unit_cost, 2),
            "store_location": store,
            "last_updated": count_date.isoformat(),
            "status": "active",
        }

        physical_date = count_date + timedelta(days=1)
        physical = {
            "id": f"PHY_{physical_date.strftime('%Y%m%d')}_{case_id:03d}",
            "count_date": physical_date.strftime("%Y-%m-%d"),
            "sku": sku if critical_type != "sku_mismatch" else f"{category['prefix']}-9999{case_id}",
            "product_name": product_name.upper(),
            "category": category['category'],
            "counted_quantity": physical_quantity,
            "counter_name": f"Counter_{random.randint(1, 5)}",
            "store_location": store,
            "verified": critical_type != "sku_mismatch",
            "notes": f"CRITICAL: {reason}",
        }

        if pos_quantity > 0:
            actual_variance_pct = abs(physical_quantity - pos_quantity) / pos_quantity * 100
        else:
            actual_variance_pct = 100.0

        match = {
            "case_id": f"RECON_2024_{case_id:03d}",
            "pos_id": pos["id"],
            "physical_id": physical["id"],
            "expected_decision": "escalate" if critical_type in ["severe_shortage", "missing_item"] else "reject",
            "match_score": 0.25,
            "variance": physical_quantity - pos_quantity,
            "variance_pct": round(actual_variance_pct, 2),
            "notes": f"CRITICAL: {reason} ({actual_variance_pct:.1f}% variance)",
        }

        return pos, physical, match

    def save_dataset(self, dataset: Dict[str, Any], output_dir: Path):
        """Save dataset to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save POS inventory
        pos_file = output_dir / "pos_inventory.json"
        with open(pos_file, "w") as f:
            json.dump(dataset["pos_inventory"], f, indent=2)

        # Save physical count
        physical_file = output_dir / "physical_count.json"
        with open(physical_file, "w") as f:
            json.dump(dataset["physical_count"], f, indent=2)

        # Save expected matches
        matches_file = output_dir / "expected_matches.json"
        with open(matches_file, "w") as f:
            json.dump(dataset["expected_matches"], f, indent=2)

        # Save metadata
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

        print(f"[OK] Dataset saved to {output_dir}")
        print(f"   - POS inventory: {len(dataset['pos_inventory'])}")
        print(f"   - Physical count: {len(dataset['physical_count'])}")
        print(f"   - Expected matches: {len(dataset['expected_matches'])}")


def main():
    """Generate synthetic retail inventory reconciliation data"""
    print("Generating synthetic retail inventory reconciliation data...")

    generator = RetailDataGenerator()
    dataset = generator.generate_dataset(n_cases=20)

    # Save to data/synthetic/retail
    output_dir = Path(__file__).parent / "retail"
    generator.save_dataset(dataset, output_dir)

    # Print summary
    print("\nDataset Summary:")
    print(f"  Exact Matches: {dataset['metadata']['exact_matches']} (40%)")
    print(f"  Minor Variance: {dataset['metadata']['minor_variance']} (30%)")
    print(f"  Major Variance: {dataset['metadata']['major_variance']} (20%)")
    print(f"  Critical Cases: {dataset['metadata']['critical']} (10%)")
    print("\nExpected Auto-Resolution Rate: ~70%")
    print("Expected Human Review Rate: ~20%")
    print("Expected Escalation Rate: ~10%")


if __name__ == "__main__":
    main()
