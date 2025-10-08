"""
QURE Knowledge Substrate

Unified storage layer for vectors, graphs, features, and evidence.
"""

# Optional imports - only needed if infrastructure is running
try:
    from .evidence_tracker import EvidenceTracker, get_evidence_tracker
    from .feature_store import FeatureStore, get_feature_store
    from .graph_store import GraphStore, get_graph_store
    from .vector_store import VectorStore, get_vector_store

    __all__ = [
        "VectorStore",
        "get_vector_store",
        "GraphStore",
        "get_graph_store",
        "FeatureStore",
        "get_feature_store",
        "EvidenceTracker",
        "get_evidence_tracker",
    ]
except ImportError as e:
    # Infrastructure components not available - demo will run in simplified mode
    import warnings
    warnings.warn(f"Substrate components not available: {e}. Running in simplified mode.")
    __all__ = []
