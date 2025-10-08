"""
Retriever Agent

Fetches data from heterogeneous sources (local files, S3, APIs, databases).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.schemas import Document, RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Retriever Agent for data ingestion

    Responsibilities:
    - Connect to data sources via adapters
    - Handle auth, retries, rate limiting
    - Stream large documents
    - Emit raw documents to Data Agent

    Supported sources:
    - Local files (CSV, JSON, PDF, TXT)
    - S3 buckets (via boto3)
    - HTTP APIs
    """

    def __init__(self):
        """Initialize retriever agent"""
        self.connectors = {}
        logger.info("✅ Retriever Agent initialized")

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """
        Retrieve documents from sources

        Args:
            request: RetrievalRequest with source IDs and types

        Returns:
            RetrievalResponse with retrieved documents
        """
        documents = []
        stats = {
            "total_requested": len(request.source_ids),
            "successful": 0,
            "failed": 0,
            "errors": [],
        }

        for source_id, source_type in zip(request.source_ids, request.source_types):
            try:
                doc = self._retrieve_single(source_id, source_type)
                if doc:
                    documents.append(doc)
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(f"Failed to retrieve {source_id}")

            except Exception as e:
                logger.error(f"Error retrieving {source_id}: {e}")
                stats["failed"] += 1
                stats["errors"].append(f"{source_id}: {str(e)}")

        logger.info(
            f"Retrieved {stats['successful']}/{stats['total_requested']} documents"
        )

        return RetrievalResponse(
            case_id=request.case_id,
            from_agent=request.to_agent or request.from_agent,
            to_agent=None,
            documents=documents,
            retrieval_stats=stats,
        )

    def _retrieve_single(
        self,
        source_id: str,
        source_type: str,
    ) -> Optional[Document]:
        """
        Retrieve a single document

        Args:
            source_id: Source identifier (file path, S3 URI, URL, etc.)
            source_type: Source type (local, s3, api, db)

        Returns:
            Document or None if failed
        """
        if source_type == "local":
            return self._retrieve_local_file(source_id)
        elif source_type == "s3":
            return self._retrieve_s3_file(source_id)
        elif source_type == "api":
            return self._retrieve_api(source_id)
        elif source_type == "db":
            return self._retrieve_database(source_id)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return None

    def _retrieve_local_file(self, file_path: str) -> Optional[Document]:
        """
        Retrieve a local file

        Args:
            file_path: Path to local file

        Returns:
            Document object
        """
        try:
            path = Path(file_path)

            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Determine file type
            suffix = path.suffix.lower()
            doc_type = self._get_doc_type(suffix)

            # Read content based on type
            if suffix in [".txt", ".csv", ".json", ".md"]:
                content = path.read_text(encoding="utf-8")
            elif suffix == ".pdf":
                content = self._read_pdf(path)
            else:
                # Binary files - read as bytes and encode
                content = path.read_bytes().decode("utf-8", errors="ignore")

            logger.debug(f"Retrieved local file: {file_path} ({len(content)} chars)")

            return Document(
                id=str(path),
                content=content,
                doc_type=doc_type,
                source=f"local:{file_path}",
                metadata={
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "file_type": suffix,
                },
            )

        except Exception as e:
            logger.error(f"Failed to read local file {file_path}: {e}")
            return None

    def _retrieve_s3_file(self, s3_uri: str) -> Optional[Document]:
        """
        Retrieve a file from S3

        Args:
            s3_uri: S3 URI (s3://bucket/key)

        Returns:
            Document object
        """
        try:
            # Parse S3 URI
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"Invalid S3 URI: {s3_uri}")

            parts = s3_uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            # Import boto3 only if needed
            import boto3

            s3_client = boto3.client("s3")

            # Get object
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8", errors="ignore")

            # Determine doc type from key extension
            suffix = Path(key).suffix.lower()
            doc_type = self._get_doc_type(suffix)

            logger.debug(f"Retrieved S3 file: {s3_uri} ({len(content)} chars)")

            return Document(
                id=s3_uri,
                content=content,
                doc_type=doc_type,
                source=f"s3:{s3_uri}",
                metadata={
                    "bucket": bucket,
                    "key": key,
                    "content_type": response.get("ContentType"),
                    "last_modified": str(response.get("LastModified")),
                },
            )

        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve S3 file {s3_uri}: {e}")
            return None

    def _retrieve_api(self, url: str) -> Optional[Document]:
        """
        Retrieve data from an HTTP API

        Args:
            url: API endpoint URL

        Returns:
            Document object
        """
        try:
            import aiohttp
            import asyncio

            async def fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        content = await response.text()
                        return content, response.headers.get("content-type")

            # Run async fetch
            content, content_type = asyncio.run(fetch())

            doc_type = "api_json" if "json" in content_type else "api_text"

            logger.debug(f"Retrieved API data: {url} ({len(content)} chars)")

            return Document(
                id=url,
                content=content,
                doc_type=doc_type,
                source=f"api:{url}",
                metadata={
                    "url": url,
                    "content_type": content_type,
                },
            )

        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve API data from {url}: {e}")
            return None

    def _retrieve_database(self, query_spec: str) -> Optional[Document]:
        """
        Retrieve data from a database

        Args:
            query_spec: Query specification (format: "db_type:connection:query")

        Returns:
            Document object
        """
        # Placeholder for database retrieval
        # Would implement specific connectors for Postgres, MySQL, etc.
        logger.warning(f"Database retrieval not yet implemented: {query_spec}")
        return None

    def _read_pdf(self, path: Path) -> str:
        """
        Read PDF file content

        Args:
            path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            text_parts = []

            for page in reader.pages:
                text_parts.append(page.extract_text())

            content = "\n\n".join(text_parts)
            logger.debug(f"Extracted {len(content)} chars from PDF with {len(reader.pages)} pages")

            return content

        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return f"[PDF file: {path.name} - PyPDF2 not available]"
        except Exception as e:
            logger.error(f"Failed to read PDF {path}: {e}")
            return f"[PDF file: {path.name} - extraction failed]"

    def _get_doc_type(self, suffix: str) -> str:
        """
        Map file suffix to document type

        Args:
            suffix: File extension (e.g., '.pdf')

        Returns:
            Document type string
        """
        type_map = {
            ".pdf": "pdf",
            ".txt": "text",
            ".csv": "csv",
            ".json": "json",
            ".md": "markdown",
            ".html": "html",
            ".xml": "xml",
            ".docx": "docx",
            ".xlsx": "xlsx",
        }
        return type_map.get(suffix, "unknown")

    def retrieve_batch(
        self,
        requests: List[RetrievalRequest],
    ) -> List[RetrievalResponse]:
        """
        Retrieve multiple document sets in batch

        Args:
            requests: List of RetrievalRequest objects

        Returns:
            List of RetrievalResponse objects
        """
        responses = []

        for request in requests:
            response = self.retrieve(request)
            responses.append(response)

        logger.info(f"Completed batch retrieval: {len(responses)} requests processed")
        return responses


# Singleton instance
_retriever_agent: Optional[RetrieverAgent] = None


def get_retriever_agent() -> RetrieverAgent:
    """
    Get or create singleton RetrieverAgent instance

    Returns:
        RetrieverAgent instance
    """
    global _retriever_agent

    if _retriever_agent is None:
        _retriever_agent = RetrieverAgent()

    return _retriever_agent
