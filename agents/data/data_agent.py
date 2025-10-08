"""
Data Agent - Universal Data Integrator (UDI)

Parses, normalizes, links, and enriches raw data into the Knowledge Substrate.
"""

import logging
from typing import Any, Dict, List, Optional

from common.schemas import (
    DataProcessingRequest,
    DataProcessingResponse,
    Document,
    Entity,
    Feature,
    FeatureVector,
    GraphEdge,
    GraphNode,
)
from substrate import (
    get_evidence_tracker,
    get_feature_store,
    get_graph_store,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class DataAgent:
    """
    Universal Data Integrator Agent

    Responsibilities:
    - Extract entities (NER: parties, dates, amounts, codes)
    - Normalize formats (dates, currency, addresses)
    - Resolve references (link claim ID to party, payment to claim)
    - Compute embeddings for text chunks
    - Build property graph edges
    - Store in Knowledge Substrate
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        use_graph: bool = True,
        use_features: bool = True,
    ):
        """
        Initialize Data Agent

        Args:
            use_embeddings: Enable embedding generation
            use_graph: Enable graph construction
            use_features: Enable feature engineering
        """
        self.use_embeddings = use_embeddings
        self.use_graph = use_graph
        self.use_features = use_features

        # Initialize substrate components
        if self.use_embeddings:
            self.vector_store = get_vector_store()

        if self.use_graph:
            self.graph_store = get_graph_store()

        if self.use_features:
            self.feature_store = get_feature_store()

        self.evidence_tracker = get_evidence_tracker()

        # Lazy load NLP models
        self.nlp = None
        self.embedding_model = None

        logger.info("✅ Data Agent initialized")

    def process(
        self,
        request: DataProcessingRequest,
    ) -> DataProcessingResponse:
        """
        Process documents through UDI pipeline

        Args:
            request: DataProcessingRequest with documents

        Returns:
            DataProcessingResponse with enriched documents
        """
        enriched_docs = []
        graph_updates = []
        feature_vectors = []
        embeddings_stored = 0

        for doc in request.documents:
            try:
                # Extract entities
                entities = self._extract_entities(doc)
                doc.entities = entities

                # Normalize entity values
                normalized_entities = self._normalize_entities(entities)
                doc.entities = normalized_entities

                # Build graph nodes and edges
                if self.use_graph:
                    nodes, edges = self._build_graph(doc, normalized_entities)
                    graph_updates.extend([
                        {"type": "node", "data": node} for node in nodes
                    ] + [
                        {"type": "edge", "data": edge} for edge in edges
                    ])

                # Compute embeddings
                if self.use_embeddings:
                    embedding = self._compute_embedding(doc.content)
                    doc.embedding = embedding

                    # Store in vector store
                    self.vector_store.add_document(
                        doc_id=doc.id,
                        content=doc.content,
                        embedding=embedding,
                        metadata={
                            "doc_type": doc.doc_type,
                            "source": doc.source,
                            "entity_count": len(doc.entities),
                        },
                    )
                    embeddings_stored += 1

                # Engineer features
                if self.use_features:
                    fv = self._engineer_features(doc, normalized_entities)
                    if fv:
                        feature_vectors.append(fv)
                        self.feature_store.store_feature_vector(fv)

                enriched_docs.append(doc)

                logger.debug(f"Processed document {doc.id}: {len(entities)} entities")

            except Exception as e:
                logger.error(f"Failed to process document {doc.id}: {e}")
                enriched_docs.append(doc)  # Include even if processing failed

        logger.info(
            f"Processed {len(enriched_docs)} documents: "
            f"{embeddings_stored} embeddings, "
            f"{len(graph_updates)} graph updates, "
            f"{len(feature_vectors)} feature vectors"
        )

        return DataProcessingResponse(
            case_id=request.case_id,
            from_agent=request.to_agent or request.from_agent,
            to_agent=None,
            documents=enriched_docs,
            graph_updates=graph_updates,
            feature_vectors=feature_vectors,
            embeddings_stored=embeddings_stored,
        )

    def _extract_entities(self, doc: Document) -> List[Entity]:
        """
        Extract named entities from document

        Args:
            doc: Document to process

        Returns:
            List of Entity objects
        """
        if self.nlp is None:
            self._load_nlp_model()

        entities = []

        try:
            # Process with spaCy
            doc_nlp = self.nlp(doc.content[:100000])  # Limit to 100k chars

            for ent in doc_nlp.ents:
                entity = Entity(
                    entity_type=ent.label_,
                    value=ent.text,
                    normalized_value=None,  # Will normalize in next step
                    confidence=0.8,  # spaCy doesn't provide confidence scores by default
                )
                entities.append(entity)

            logger.debug(f"Extracted {len(entities)} entities from {doc.id}")

        except Exception as e:
            logger.error(f"Entity extraction failed for {doc.id}: {e}")

        return entities

    def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Normalize entity values

        Args:
            entities: List of entities

        Returns:
            List of normalized entities
        """
        normalized = []

        for entity in entities:
            try:
                if entity.entity_type == "DATE":
                    entity.normalized_value = self._normalize_date(entity.value)
                elif entity.entity_type == "MONEY":
                    entity.normalized_value = self._normalize_money(entity.value)
                elif entity.entity_type in ["PERSON", "ORG"]:
                    entity.normalized_value = self._normalize_name(entity.value)
                else:
                    entity.normalized_value = entity.value.strip()

                normalized.append(entity)

            except Exception as e:
                logger.warning(f"Failed to normalize entity {entity.value}: {e}")
                normalized.append(entity)

        return normalized

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format"""
        from dateutil import parser
        try:
            dt = parser.parse(date_str, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str

    def _normalize_money(self, money_str: str) -> str:
        """Normalize money to float string"""
        import re
        # Remove currency symbols and commas
        clean = re.sub(r'[^\d.]', '', money_str)
        try:
            amount = float(clean)
            return f"{amount:.2f}"
        except:
            return money_str

    def _normalize_name(self, name: str) -> str:
        """Normalize person/org names"""
        # Title case, strip whitespace
        return name.strip().title()

    def _build_graph(
        self,
        doc: Document,
        entities: List[Entity],
    ) -> tuple[List[GraphNode], List[GraphEdge]]:
        """
        Build graph nodes and edges from document and entities

        Args:
            doc: Document
            entities: Extracted entities

        Returns:
            Tuple of (nodes, edges)
        """
        nodes = []
        edges = []

        # Create document node
        doc_node = GraphNode(
            id=doc.id,
            label="Document",
            properties={
                "doc_type": doc.doc_type,
                "source": doc.source,
                "entity_count": len(entities),
            },
        )
        nodes.append(doc_node)

        # Create entity nodes and edges
        for i, entity in enumerate(entities):
            entity_id = f"{doc.id}_entity_{i}"

            entity_node = GraphNode(
                id=entity_id,
                label=entity.entity_type,
                properties={
                    "value": entity.value,
                    "normalized_value": entity.normalized_value,
                    "confidence": entity.confidence,
                },
            )
            nodes.append(entity_node)

            # Link document to entity
            edge = GraphEdge(
                source_id=doc.id,
                target_id=entity_id,
                relationship="CONTAINS",
                properties={},
            )
            edges.append(edge)

        # Actually create in graph store
        try:
            for node in nodes:
                self.graph_store.create_node(
                    node_id=node.id,
                    label=node.label,
                    properties=node.properties,
                )

            for edge in edges:
                self.graph_store.create_relationship(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relationship_type=edge.relationship,
                    properties=edge.properties,
                )

        except Exception as e:
            logger.error(f"Failed to create graph nodes/edges: {e}")

        return nodes, edges

    def _compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            self._load_embedding_model()

        try:
            # Use sentence-transformers for local embeddings
            embedding = self.embedding_model.encode(text[:8192])  # Limit length
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # sentence-transformers default dim

    def _engineer_features(
        self,
        doc: Document,
        entities: List[Entity],
    ) -> Optional[FeatureVector]:
        """
        Engineer features from document

        Args:
            doc: Document
            entities: Extracted entities

        Returns:
            FeatureVector or None
        """
        features = []

        # Basic document features
        features.append(Feature(
            name="doc_length",
            value=len(doc.content),
            feature_type="numeric",
        ))

        features.append(Feature(
            name="entity_count",
            value=len(entities),
            feature_type="numeric",
        ))

        # Entity type counts
        entity_type_counts = {}
        for entity in entities:
            entity_type_counts[entity.entity_type] = entity_type_counts.get(entity.entity_type, 0) + 1

        for entity_type, count in entity_type_counts.items():
            features.append(Feature(
                name=f"entity_{entity_type.lower()}_count",
                value=count,
                feature_type="numeric",
            ))

        # Document type
        features.append(Feature(
            name="doc_type",
            value=doc.doc_type,
            feature_type="categorical",
        ))

        return FeatureVector(
            entity_id=doc.id,
            features=features,
        )

    def _load_nlp_model(self):
        """Lazy load spaCy NLP model"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            # Create a basic nlp object
            import spacy
            self.nlp = spacy.blank("en")

    def _load_embedding_model(self):
        """Lazy load sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise


# Singleton instance
_data_agent: Optional[DataAgent] = None


def get_data_agent(
    use_embeddings: bool = True,
    use_graph: bool = True,
    use_features: bool = True,
) -> DataAgent:
    """
    Get or create singleton DataAgent instance

    Args:
        use_embeddings: Enable embedding generation
        use_graph: Enable graph construction
        use_features: Enable feature engineering

    Returns:
        DataAgent instance
    """
    global _data_agent

    if _data_agent is None:
        _data_agent = DataAgent(
            use_embeddings=use_embeddings,
            use_graph=use_graph,
            use_features=use_features,
        )

    return _data_agent
