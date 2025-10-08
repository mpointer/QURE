"""
Vector Store implementation using ChromaDB

Handles embedding storage and semantic search over document chunks.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from common.schemas import Document, TextSpan

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for semantic search

    Features:
    - Store document embeddings with metadata
    - Semantic similarity search
    - Filter by metadata
    - Evidence linking with source spans
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "qure_documents",
    ):
        """
        Initialize ChromaDB client

        Args:
            host: ChromaDB server host
            port: ChromaDB server port
            collection_name: Collection name for documents
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name

        try:
            # Connect to ChromaDB server
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "QURE document embeddings"}
            )

            logger.info(f"✅ Connected to ChromaDB at {host}:{port}")
            logger.info(f"✅ Collection '{collection_name}' ready with {self.collection.count()} documents")

        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise

    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a document with its embedding

        Args:
            doc_id: Unique document ID
            content: Document text content
            embedding: Embedding vector
            metadata: Optional metadata dict
        """
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}]
            )
            logger.debug(f"Added document {doc_id} to vector store")

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise

    def add_documents(
        self,
        doc_ids: List[str],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add multiple documents in batch

        Args:
            doc_ids: List of document IDs
            contents: List of document contents
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
        """
        if metadatas is None:
            metadatas = [{} for _ in doc_ids]

        try:
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(doc_ids)} documents to vector store")

        except Exception as e:
            logger.error(f"Failed to add batch documents: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"vertical": "finance"})
            where_document: Document content filter

        Returns:
            Dict with 'ids', 'distances', 'documents', 'metadatas'
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document,
            )

            # Flatten results (ChromaDB returns nested lists)
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_text(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search using text query (ChromaDB will embed it)

        Args:
            query_text: Text query
            n_results: Number of results
            where: Metadata filter

        Returns:
            Dict with search results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
            )

            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            }

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0],
                    "embedding": result["embeddings"][0] if result.get("embeddings") else None,
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update document metadata

        Args:
            doc_id: Document ID
            metadata: New metadata dict
        """
        try:
            self.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            logger.debug(f"Updated metadata for document {doc_id}")

        except Exception as e:
            logger.error(f"Failed to update metadata for {doc_id}: {e}")
            raise

    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document

        Args:
            doc_id: Document ID to delete
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id}")

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection (use with caution!)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Deleted collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()

    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at first N documents

        Args:
            limit: Number of documents to peek

        Returns:
            Sample documents
        """
        try:
            return self.collection.peek(limit=limit)
        except Exception as e:
            logger.error(f"Failed to peek: {e}")
            return {}

    def extract_text_spans(
        self,
        query: str,
        source_doc_id: str,
        n_results: int = 3,
    ) -> List[TextSpan]:
        """
        Extract relevant text spans from a document for evidence

        Args:
            query: What we're looking for
            source_doc_id: Source document ID
            n_results: Number of spans to extract

        Returns:
            List of TextSpan objects with citations
        """
        # Search for relevant chunks from the specific document
        results = self.search_text(
            query_text=query,
            n_results=n_results,
            where={"doc_id": source_doc_id}
        )

        spans = []
        for doc_text, metadata, distance in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            # Convert distance to confidence (closer = higher confidence)
            confidence = max(0.0, 1.0 - distance)

            span = TextSpan(
                text=doc_text,
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", len(doc_text)),
                source_id=source_doc_id,
                confidence=confidence
            )
            spans.append(span)

        return spans


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(
    host: str = "localhost",
    port: int = 8000,
    collection_name: str = "qure_documents",
) -> VectorStore:
    """
    Get or create singleton VectorStore instance

    Args:
        host: ChromaDB host
        port: ChromaDB port
        collection_name: Collection name

    Returns:
        VectorStore instance
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore(host=host, port=port, collection_name=collection_name)

    return _vector_store
