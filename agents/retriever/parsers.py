"""
Document parsers for various file formats
"""

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def parse_csv(content: str) -> Dict[str, Any]:
    """
    Parse CSV content

    Args:
        content: CSV text content

    Returns:
        Dict with rows and metadata
    """
    try:
        reader = csv.DictReader(StringIO(content))
        rows = list(reader)

        return {
            "rows": rows,
            "row_count": len(rows),
            "columns": reader.fieldnames or [],
            "format": "csv",
        }

    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        return {"error": str(e), "format": "csv"}


def parse_json(content: str) -> Dict[str, Any]:
    """
    Parse JSON content

    Args:
        content: JSON text content

    Returns:
        Parsed JSON data
    """
    try:
        data = json.loads(content)

        return {
            "data": data,
            "format": "json",
            "type": type(data).__name__,
        }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return {"error": str(e), "format": "json"}


def parse_table_text(content: str) -> List[Dict[str, str]]:
    """
    Parse simple text tables (whitespace-separated)

    Args:
        content: Text content with tables

    Returns:
        List of row dicts
    """
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    if not lines:
        return []

    # Assume first line is headers
    headers = lines[0].split()

    rows = []
    for line in lines[1:]:
        values = line.split()
        if len(values) == len(headers):
            row = dict(zip(headers, values))
            rows.append(row)

    return rows


def extract_amounts(content: str) -> List[float]:
    """
    Extract monetary amounts from text

    Args:
        content: Text content

    Returns:
        List of extracted amounts
    """
    import re

    # Pattern for dollar amounts: $1,234.56 or 1234.56
    pattern = r'\$?[\d,]+\.?\d*'

    matches = re.findall(pattern, content)

    amounts = []
    for match in matches:
        try:
            # Remove $ and commas, convert to float
            clean = match.replace('$', '').replace(',', '')
            amounts.append(float(clean))
        except ValueError:
            continue

    return amounts


def extract_dates(content: str) -> List[str]:
    """
    Extract dates from text

    Args:
        content: Text content

    Returns:
        List of date strings
    """
    import re

    # Common date patterns
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or M/D/YY
        r'\d{4}-\d{2}-\d{2}',         # YYYY-MM-DD
        r'\d{2}-\d{2}-\d{4}',         # DD-MM-YYYY
        r'[A-Z][a-z]+ \d{1,2},? \d{4}',  # Month DD, YYYY
    ]

    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, content)
        dates.extend(matches)

    return dates


def extract_email_parts(content: str) -> Dict[str, Any]:
    """
    Extract structured parts from email content

    Args:
        content: Email text

    Returns:
        Dict with from, to, subject, body
    """
    parts = {
        "from": None,
        "to": None,
        "subject": None,
        "date": None,
        "body": content,
    }

    lines = content.split("\n")

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if line_lower.startswith("from:"):
            parts["from"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("to:"):
            parts["to"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("subject:"):
            parts["subject"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("date:"):
            parts["date"] = line.split(":", 1)[1].strip()

        # Body starts after headers
        if not line.strip() and i < len(lines) - 1:
            parts["body"] = "\n".join(lines[i+1:])
            break

    return parts


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        chunk_size: Max chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of chunk dicts with text and offsets
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start:
                end = sentence_end + 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "chunk_index": len(chunks),
            })

        # Move start position with overlap
        start = end - overlap if end > overlap else end

    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks


def extract_key_value_pairs(content: str) -> Dict[str, str]:
    """
    Extract key-value pairs from text (e.g., "Name: John Doe")

    Args:
        content: Text content

    Returns:
        Dict of extracted key-value pairs
    """
    import re

    # Pattern: "Key: Value" or "Key = Value"
    pattern = r'([A-Za-z\s]+)[:=]\s*([^\n]+)'

    matches = re.findall(pattern, content)

    pairs = {}
    for key, value in matches:
        key_clean = key.strip().lower().replace(" ", "_")
        pairs[key_clean] = value.strip()

    return pairs
