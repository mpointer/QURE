"""
Synthetic Healthcare Data Generator

Generates realistic Prior Authorization test cases with various scenarios.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Seed for reproducibility
random.seed(42)


class HealthcareDataGenerator:
    """Generate synthetic Prior Authorization request and clinical data"""

    def __init__(self):
        self.procedures = [
            {"code": "99285", "name": "Emergency Department Visit - High Complexity", "typical_cost": 850},
            {"code": "27447", "name": "Total Knee Arthroplasty", "typical_cost": 35000},
            {"code": "70553", "name": "MRI Brain with Contrast", "typical_cost": 2500},
            {"code": "93458", "name": "Cardiac Catheterization", "typical_cost": 8500},
            {"code": "43235", "name": "Upper GI Endoscopy", "typical_cost": 1500},
            {"code": "47562", "name": "Laparoscopic Cholecystectomy", "typical_cost": 15000},
            {"code": "62223", "name": "Spinal Fusion Surgery", "typical_cost": 45000},
            {"code": "29827", "name": "Arthroscopy Shoulder Repair", "typical_cost": 12000},
            {"code": "99284", "name": "Emergency Department Visit - Moderate", "typical_cost": 450},
            {"code": "99213", "name": "Office Visit - Established Patient", "typical_cost": 150},
        ]

        self.diagnoses = [
            {"icd10": "M25.561", "name": "Pain in right knee", "severity": "moderate"},
            {"icd10": "I25.10", "name": "Atherosclerotic heart disease", "severity": "high"},
            {"icd10": "K21.9", "name": "Gastroesophageal reflux disease", "severity": "low"},
            {"icd10": "M54.5", "name": "Low back pain", "severity": "moderate"},
            {"icd10": "G43.909", "name": "Migraine headache", "severity": "moderate"},
            {"icd10": "K80.20", "name": "Calculus of gallbladder", "severity": "high"},
            {"icd10": "M75.100", "name": "Rotator cuff tear", "severity": "moderate"},
            {"icd10": "I63.9", "name": "Cerebral infarction", "severity": "critical"},
            {"icd10": "E11.9", "name": "Type 2 diabetes mellitus", "severity": "moderate"},
            {"icd10": "J44.1", "name": "Chronic obstructive pulmonary disease", "severity": "high"},
        ]

        self.providers = [
            "Dr. Sarah Johnson, MD - Orthopedic Surgery",
            "Dr. Michael Chen, MD - Cardiology",
            "Dr. Emily Rodriguez, MD - Gastroenterology",
            "Dr. James Williams, MD - Neurosurgery",
            "Dr. Lisa Anderson, MD - General Surgery",
            "Dr. Robert Martinez, DO - Pain Management",
            "Dr. Jennifer Taylor, MD - Emergency Medicine",
            "Dr. David Brown, MD - Internal Medicine",
        ]

        self.facilities = [
            "Memorial Hospital - Main Campus",
            "St. Mary's Medical Center",
            "University Health System",
            "Regional Orthopedic Center",
            "Downtown Surgical Hospital",
            "Lakeside Medical Plaza",
        ]

    def generate_dataset(self, n_cases: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate dataset with various scenarios

        Args:
            n_cases: Number of test cases

        Returns:
            Dict with 'prior_auth_requests', 'clinical_documentation', 'expected_matches'
        """
        prior_auth_requests = []
        clinical_documentation = []
        expected_matches = []

        # Case types distribution
        auto_approve = int(n_cases * 0.35)      # 35% auto-approve (clear medical necessity)
        hitl_review = int(n_cases * 0.40)       # 40% requires review (borderline cases)
        request_info = int(n_cases * 0.15)      # 15% missing documentation
        deny = n_cases - auto_approve - hitl_review - request_info  # 10% deny

        case_id = 1

        # Generate auto-approve cases
        for i in range(auto_approve):
            pa, clinical, match = self._generate_auto_approve(case_id)
            prior_auth_requests.append(pa)
            clinical_documentation.append(clinical)
            expected_matches.append(match)
            case_id += 1

        # Generate HITL review cases
        for i in range(hitl_review):
            pa, clinical, match = self._generate_hitl_review(case_id)
            prior_auth_requests.append(pa)
            clinical_documentation.append(clinical)
            expected_matches.append(match)
            case_id += 1

        # Generate request info cases
        for i in range(request_info):
            pa, clinical, match = self._generate_request_info(case_id)
            prior_auth_requests.append(pa)
            clinical_documentation.append(clinical)
            expected_matches.append(match)
            case_id += 1

        # Generate deny cases
        for i in range(deny):
            pa, clinical, match = self._generate_deny(case_id)
            prior_auth_requests.append(pa)
            clinical_documentation.append(clinical)
            expected_matches.append(match)
            case_id += 1

        return {
            "prior_auth_requests": prior_auth_requests,
            "clinical_documentation": clinical_documentation,
            "expected_matches": expected_matches,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_cases": n_cases,
                "auto_approve": auto_approve,
                "hitl_review": hitl_review,
                "request_info": request_info,
                "deny": deny,
            }
        }

    def _generate_auto_approve(self, case_id: int) -> tuple:
        """Generate auto-approve scenario (clear medical necessity)"""
        request_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        procedure = random.choice(self.procedures)
        diagnosis = random.choice([d for d in self.diagnoses if d["severity"] in ["high", "critical"]])
        provider = random.choice(self.providers)
        facility = random.choice(self.facilities)

        # Prior auth request
        pa = {
            "id": f"PA_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "request_date": request_date.strftime("%Y-%m-%d"),
            "patient_id": f"MRN{random.randint(100000, 999999)}",
            "patient_name": f"Patient {case_id}",
            "age": random.randint(45, 75),
            "procedure_code": procedure["code"],
            "procedure_name": procedure["name"],
            "diagnosis_code": diagnosis["icd10"],
            "diagnosis_name": diagnosis["name"],
            "requesting_provider": provider,
            "facility": facility,
            "urgency": "urgent" if diagnosis["severity"] == "critical" else "routine",
            "estimated_cost": procedure["typical_cost"],
            "status": "pending",
        }

        # Clinical documentation
        clinical = {
            "id": f"CLINICAL_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "pa_request_id": pa["id"],
            "patient_id": pa["patient_id"],
            "documentation_date": request_date.strftime("%Y-%m-%d"),
            "clinical_notes": f"Patient presents with {diagnosis['name']}. Conservative treatment attempted for 6 weeks without improvement. Imaging studies confirm diagnosis. Procedure medically necessary per clinical guidelines.",
            "prior_treatments": ["Physical therapy - 6 weeks", "NSAIDs - 8 weeks", "Imaging studies completed"],
            "contraindications": [],
            "comorbidities": random.sample(["hypertension", "diabetes", "hyperlipidemia", "asthma"], k=random.randint(0, 2)),
            "lab_results": "Within normal limits",
            "imaging_results": f"Confirms {diagnosis['name']}",
            "medical_necessity_score": random.uniform(0.85, 0.98),
        }

        # Expected match
        match = {
            "case_id": f"HEALTHCARE_2024_{case_id:03d}",
            "pa_id": pa["id"],
            "clinical_id": clinical["id"],
            "expected_decision": "auto_resolve",
            "match_score": random.uniform(0.90, 0.99),
            "notes": f"Clear medical necessity - {diagnosis['severity']} severity diagnosis with documented conservative treatment failure",
        }

        return pa, clinical, match

    def _generate_hitl_review(self, case_id: int) -> tuple:
        """Generate HITL review scenario (borderline medical necessity)"""
        request_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        procedure = random.choice(self.procedures)
        diagnosis = random.choice([d for d in self.diagnoses if d["severity"] == "moderate"])
        provider = random.choice(self.providers)
        facility = random.choice(self.facilities)

        # Variation type for borderline cases
        variation = random.choice([
            "short_conservative_treatment",  # Less than recommended treatment duration
            "unclear_medical_necessity",      # Vague clinical notes
            "alternative_available",          # Less expensive alternative exists
            "policy_exception_needed",        # Outside standard policy but reasonable
        ])

        pa = {
            "id": f"PA_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "request_date": request_date.strftime("%Y-%m-%d"),
            "patient_id": f"MRN{random.randint(100000, 999999)}",
            "patient_name": f"Patient {case_id}",
            "age": random.randint(30, 70),
            "procedure_code": procedure["code"],
            "procedure_name": procedure["name"],
            "diagnosis_code": diagnosis["icd10"],
            "diagnosis_name": diagnosis["name"],
            "requesting_provider": provider,
            "facility": facility,
            "urgency": "routine",
            "estimated_cost": procedure["typical_cost"],
            "status": "pending",
        }

        # Clinical documentation varies by scenario
        if variation == "short_conservative_treatment":
            clinical_notes = f"Patient with {diagnosis['name']}. Conservative treatment attempted for 3 weeks with partial improvement. Patient requesting procedure."
            prior_treatments = ["Physical therapy - 3 weeks", "OTC pain medication"]
            med_necessity_score = random.uniform(0.65, 0.75)
        elif variation == "unclear_medical_necessity":
            clinical_notes = f"Patient complains of {diagnosis['name']}. Procedure recommended."
            prior_treatments = ["Conservative management attempted"]
            med_necessity_score = random.uniform(0.60, 0.70)
        elif variation == "alternative_available":
            clinical_notes = f"Patient with {diagnosis['name']}. Less invasive options available but patient preference for surgical intervention."
            prior_treatments = ["Physical therapy - 6 weeks", "Medications"]
            med_necessity_score = random.uniform(0.70, 0.80)
        else:  # policy_exception_needed
            clinical_notes = f"Patient with {diagnosis['name']}. Age/condition outside typical policy guidelines but clinical team recommends exception."
            prior_treatments = ["Multiple conservative treatments attempted"]
            med_necessity_score = random.uniform(0.75, 0.82)

        clinical = {
            "id": f"CLINICAL_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "pa_request_id": pa["id"],
            "patient_id": pa["patient_id"],
            "documentation_date": request_date.strftime("%Y-%m-%d"),
            "clinical_notes": clinical_notes,
            "prior_treatments": prior_treatments,
            "contraindications": [],
            "comorbidities": random.sample(["hypertension", "diabetes", "obesity", "smoking"], k=random.randint(1, 3)),
            "lab_results": "Available",
            "imaging_results": f"Shows {diagnosis['name']}",
            "medical_necessity_score": med_necessity_score,
        }

        match = {
            "case_id": f"HEALTHCARE_2024_{case_id:03d}",
            "pa_id": pa["id"],
            "clinical_id": clinical["id"],
            "expected_decision": "hitl_review",
            "match_score": random.uniform(0.65, 0.82),
            "notes": f"Borderline case - {variation.replace('_', ' ')} - requires clinical review",
        }

        return pa, clinical, match

    def _generate_request_info(self, case_id: int) -> tuple:
        """Generate request info scenario (missing documentation)"""
        request_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        procedure = random.choice(self.procedures)
        diagnosis = random.choice(self.diagnoses)
        provider = random.choice(self.providers)
        facility = random.choice(self.facilities)

        pa = {
            "id": f"PA_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "request_date": request_date.strftime("%Y-%m-%d"),
            "patient_id": f"MRN{random.randint(100000, 999999)}",
            "patient_name": f"Patient {case_id}",
            "age": random.randint(25, 80),
            "procedure_code": procedure["code"],
            "procedure_name": procedure["name"],
            "diagnosis_code": diagnosis["icd10"],
            "diagnosis_name": diagnosis["name"],
            "requesting_provider": provider,
            "facility": facility,
            "urgency": "routine",
            "estimated_cost": procedure["typical_cost"],
            "status": "pending",
        }

        # Missing key documentation
        missing_item = random.choice([
            "prior_treatments",
            "imaging_results",
            "lab_results",
            "clinical_notes",
        ])

        clinical = {
            "id": f"CLINICAL_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "pa_request_id": pa["id"],
            "patient_id": pa["patient_id"],
            "documentation_date": request_date.strftime("%Y-%m-%d"),
            "clinical_notes": "See attached" if missing_item == "clinical_notes" else f"Patient with {diagnosis['name']}",
            "prior_treatments": [] if missing_item == "prior_treatments" else ["Conservative treatment"],
            "contraindications": [],
            "comorbidities": [],
            "lab_results": "Pending" if missing_item == "lab_results" else "Available",
            "imaging_results": "Not submitted" if missing_item == "imaging_results" else "Available",
            "medical_necessity_score": random.uniform(0.45, 0.60),
        }

        match = {
            "case_id": f"HEALTHCARE_2024_{case_id:03d}",
            "pa_id": pa["id"],
            "clinical_id": clinical["id"],
            "expected_decision": "request_info",
            "match_score": random.uniform(0.40, 0.60),
            "notes": f"Incomplete documentation - missing {missing_item.replace('_', ' ')}",
        }

        return pa, clinical, match

    def _generate_deny(self, case_id: int) -> tuple:
        """Generate deny scenario (not medically necessary)"""
        request_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        procedure = random.choice(self.procedures)
        diagnosis = random.choice([d for d in self.diagnoses if d["severity"] == "low"])
        provider = random.choice(self.providers)
        facility = random.choice(self.facilities)

        # Denial reason
        denial_reason = random.choice([
            "experimental_treatment",
            "no_conservative_treatment",
            "diagnosis_mismatch",
            "not_medically_necessary",
        ])

        pa = {
            "id": f"PA_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "request_date": request_date.strftime("%Y-%m-%d"),
            "patient_id": f"MRN{random.randint(100000, 999999)}",
            "patient_name": f"Patient {case_id}",
            "age": random.randint(20, 65),
            "procedure_code": procedure["code"],
            "procedure_name": procedure["name"],
            "diagnosis_code": diagnosis["icd10"],
            "diagnosis_name": diagnosis["name"],
            "requesting_provider": provider,
            "facility": facility,
            "urgency": "routine",
            "estimated_cost": procedure["typical_cost"],
            "status": "pending",
        }

        if denial_reason == "experimental_treatment":
            clinical_notes = f"Requesting experimental treatment for {diagnosis['name']}. Not FDA approved for this indication."
        elif denial_reason == "no_conservative_treatment":
            clinical_notes = f"Patient requesting {procedure['name']} without attempting conservative treatment."
        elif denial_reason == "diagnosis_mismatch":
            clinical_notes = f"Procedure does not match diagnosis. {procedure['name']} not indicated for {diagnosis['name']}."
        else:
            clinical_notes = f"Procedure not medically necessary for {diagnosis['name']} with low severity."

        clinical = {
            "id": f"CLINICAL_{request_date.strftime('%Y%m%d')}_{case_id:03d}",
            "pa_request_id": pa["id"],
            "patient_id": pa["patient_id"],
            "documentation_date": request_date.strftime("%Y-%m-%d"),
            "clinical_notes": clinical_notes,
            "prior_treatments": [] if denial_reason == "no_conservative_treatment" else ["Minimal conservative treatment"],
            "contraindications": ["Procedure not indicated"],
            "comorbidities": [],
            "lab_results": "Normal",
            "imaging_results": "No significant findings",
            "medical_necessity_score": random.uniform(0.15, 0.35),
        }

        match = {
            "case_id": f"HEALTHCARE_2024_{case_id:03d}",
            "pa_id": pa["id"],
            "clinical_id": clinical["id"],
            "expected_decision": "reject",
            "match_score": random.uniform(0.15, 0.35),
            "notes": f"Denial recommended - {denial_reason.replace('_', ' ')}",
        }

        return pa, clinical, match

    def save_dataset(self, dataset: Dict[str, Any], output_dir: Path):
        """Save dataset to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save prior auth requests
        pa_file = output_dir / "prior_auth_requests.json"
        with open(pa_file, "w") as f:
            json.dump(dataset["prior_auth_requests"], f, indent=2)

        # Save clinical documentation
        clinical_file = output_dir / "clinical_documentation.json"
        with open(clinical_file, "w") as f:
            json.dump(dataset["clinical_documentation"], f, indent=2)

        # Save expected matches
        matches_file = output_dir / "expected_matches.json"
        with open(matches_file, "w") as f:
            json.dump(dataset["expected_matches"], f, indent=2)

        # Save metadata
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

        print(f"[OK] Dataset saved to {output_dir}")
        print(f"   - Prior auth requests: {len(dataset['prior_auth_requests'])}")
        print(f"   - Clinical documentation: {len(dataset['clinical_documentation'])}")
        print(f"   - Expected matches: {len(dataset['expected_matches'])}")


def main():
    """Generate synthetic healthcare data"""
    print("Generating synthetic Healthcare Prior Authorization data...")

    generator = HealthcareDataGenerator()
    dataset = generator.generate_dataset(n_cases=20)

    # Save to data/synthetic/healthcare
    output_dir = Path(__file__).parent / "healthcare"
    generator.save_dataset(dataset, output_dir)

    # Print summary
    print("\nDataset Summary:")
    print(f"  Auto-Approve: {dataset['metadata']['auto_approve']} (35%)")
    print(f"  HITL Review: {dataset['metadata']['hitl_review']} (40%)")
    print(f"  Request Info: {dataset['metadata']['request_info']} (15%)")
    print(f"  Deny: {dataset['metadata']['deny']} (10%)")
    print("\nExpected Auto-Resolution Rate: ~35%")
    print("Expected Human Review Rate: ~40%")
    print("Expected Request Info Rate: ~15%")
    print("Expected Rejection Rate: ~10%")


if __name__ == "__main__":
    main()
