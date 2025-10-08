"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_gl_transaction():
    """Sample GL transaction for testing"""
    return {
        "id": "GL_20240115_001",
        "date": "2024-01-15",
        "amount": 1250.00,
        "currency": "USD",
        "payer": "Acme Corp",
        "memo": "Invoice payment INV-2024-001",
        "reference": "INV-2024-001",
        "status": "unreconciled",
    }


@pytest.fixture
def sample_bank_transaction():
    """Sample Bank transaction for testing"""
    return {
        "id": "BANK_20240116_045",
        "date": "2024-01-16",
        "amount": 1250.00,
        "currency": "USD",
        "payer": "ACME CORPORATION",
        "memo": "Payment for invoice INV-2024-001",
        "swift_ref": "SWIFT123456",
        "transaction_type": "WIRE",
    }


@pytest.fixture
def sample_case_data():
    """Sample case data for rules evaluation"""
    return {
        "gl_amount": 1250.00,
        "bank_amount": 1250.00,
        "date_diff_days": 1,
        "gl_payer": "acme corp",
        "bank_payer": "acme corp",
        "swift_ref": "SWIFT123456",
        "gl_currency": "USD",
        "bank_currency": "USD",
        "gl_status": "unreconciled",
        "memo_similarity": 0.85,
        "is_duplicate": False,
        "business_days_elapsed": 1,
    }
