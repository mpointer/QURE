"""
Context Clusterer

Groups similar decision contexts into clusters for contextual bandit learning.
Uses K-means clustering on context features to identify context types.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ContextClusterer:
    """
    Clusters decision contexts for contextual bandit.

    Identifies context types (e.g., "high-value", "routine", "low-quality-data")
    to enable context-specific policy learning.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        feature_names: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize context clusterer.

        Args:
            n_clusters: Number of context clusters
            feature_names: List of feature names to extract from context dicts
            model_path: Path to save/load trained clusterer
        """
        self.n_clusters = n_clusters
        self.feature_names = feature_names or self._get_default_features()
        self.model_path = model_path or "data/context_clusterer.pkl"

        # Models
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_fitted = False

        # Cluster labels (human-readable)
        self.cluster_labels = [f"context_{i}" for i in range(n_clusters)]

        # Try to load existing model
        self._load_model()

    def _get_default_features(self) -> List[str]:
        """Default features to extract from context"""
        return [
            'transaction_amount',
            'data_quality_score',
            'urgency_encoded',        # low=0, medium=1, high=2, critical=3
            'has_swift_reference',
            'rule_pass_count',
            'rule_fail_count',
            'ml_confidence',
            'genai_confidence',
            'assurance_score'
        ]

    def extract_features(self, contexts: List[Dict]) -> np.ndarray:
        """
        Extract feature matrix from context dicts.

        Args:
            contexts: List of context dictionaries

        Returns:
            features: [n_samples, n_features] array
        """
        features = []

        for ctx in contexts:
            feature_vec = []

            for feat_name in self.feature_names:
                # Extract feature with default
                if feat_name == 'urgency_encoded':
                    # Encode urgency as ordinal
                    urgency_map = {
                        'low': 0, 'medium': 1, 'high': 2, 'critical': 3
                    }
                    value = urgency_map.get(ctx.get('urgency', 'medium'), 1)

                elif feat_name in ['has_swift_reference', 'sox_controlled']:
                    # Binary features
                    value = 1 if ctx.get(feat_name, False) else 0

                else:
                    # Numeric features
                    value = ctx.get(feat_name, 0.0)

                feature_vec.append(value)

            features.append(feature_vec)

        return np.array(features)

    def fit(self, contexts: List[Dict]) -> 'ContextClusterer':
        """
        Fit clusterer on context data.

        Args:
            contexts: List of context dictionaries

        Returns:
            self
        """
        # Extract features
        X = self.extract_features(contexts)

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Cluster
        self.kmeans.fit(X_scaled)
        self.is_fitted = True

        # Assign interpretable labels
        self._assign_cluster_labels(X)

        logger.info(f"Fitted clusterer on {len(contexts)} contexts")
        logger.info(f"Cluster labels: {self.cluster_labels}")

        # Save model
        self.save_model()

        return self

    def predict(self, context: Dict) -> int:
        """
        Predict cluster ID for a single context.

        Args:
            context: Context dictionary

        Returns:
            cluster_id: Integer cluster ID (0 to n_clusters-1)
        """
        if not self.is_fitted:
            logger.warning("Clusterer not fitted, returning default cluster 0")
            return 0

        # Extract features
        X = self.extract_features([context])
        X_scaled = self.scaler.transform(X)

        # Predict cluster
        cluster_id = int(self.kmeans.predict(X_scaled)[0])

        return cluster_id

    def predict_batch(self, contexts: List[Dict]) -> List[int]:
        """
        Predict cluster IDs for batch of contexts.

        Args:
            contexts: List of context dictionaries

        Returns:
            cluster_ids: List of integer cluster IDs
        """
        if not self.is_fitted:
            logger.warning("Clusterer not fitted, returning default clusters")
            return [0] * len(contexts)

        # Extract features
        X = self.extract_features(contexts)
        X_scaled = self.scaler.transform(X)

        # Predict clusters
        cluster_ids = self.kmeans.predict(X_scaled).tolist()

        return cluster_ids

    def _assign_cluster_labels(self, X: np.ndarray):
        """
        Assign interpretable labels to clusters based on centroids.

        Args:
            X: Feature matrix used for clustering
        """
        # Get cluster centroids (in original feature space)
        centroids_scaled = self.kmeans.cluster_centers_
        centroids = self.scaler.inverse_transform(centroids_scaled)

        # Assign labels based on dominant features
        labels = []
        for i, centroid in enumerate(centroids):
            # Create feature dict for centroid
            feat_dict = dict(zip(self.feature_names, centroid))

            # Determine label based on key characteristics
            amount = feat_dict.get('transaction_amount', 0)
            quality = feat_dict.get('data_quality_score', 0.5)
            urgency = feat_dict.get('urgency_encoded', 1)

            if amount > 500_000:
                label = f"high_value_{i}"
            elif quality < 0.5:
                label = f"low_quality_{i}"
            elif urgency >= 2:
                label = f"urgent_{i}"
            elif feat_dict.get('rule_fail_count', 0) > 0:
                label = f"rule_failures_{i}"
            else:
                label = f"routine_{i}"

            labels.append(label)

        self.cluster_labels = labels

    def get_cluster_stats(self, contexts: List[Dict]) -> Dict:
        """
        Get statistics for each cluster.

        Args:
            contexts: List of context dictionaries

        Returns:
            stats: Dict with cluster statistics
        """
        if not self.is_fitted:
            return {}

        # Predict clusters
        cluster_ids = self.predict_batch(contexts)

        # Compute statistics
        stats = {
            'n_clusters': self.n_clusters,
            'total_samples': len(contexts),
            'clusters': []
        }

        for i in range(self.n_clusters):
            # Get contexts in this cluster
            cluster_contexts = [ctx for ctx, cid in zip(contexts, cluster_ids) if cid == i]

            if not cluster_contexts:
                continue

            # Compute statistics
            cluster_stat = {
                'id': i,
                'label': self.cluster_labels[i],
                'count': len(cluster_contexts),
                'percentage': len(cluster_contexts) / len(contexts) * 100,
                'avg_transaction_amount': np.mean([
                    ctx.get('transaction_amount', 0) for ctx in cluster_contexts
                ]),
                'avg_data_quality': np.mean([
                    ctx.get('data_quality_score', 0.5) for ctx in cluster_contexts
                ])
            }

            stats['clusters'].append(cluster_stat)

        return stats

    def save_model(self, path: Optional[str] = None):
        """Save trained clusterer to disk"""
        path = path or self.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'feature_names': self.feature_names,
            'cluster_labels': self.cluster_labels,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved clusterer to {path}")

    def _load_model(self):
        """Load trained clusterer from disk"""
        if not Path(self.model_path).exists():
            return

        try:
            model_data = joblib.load(self.model_path)

            self.scaler = model_data['scaler']
            self.kmeans = model_data['kmeans']
            self.feature_names = model_data['feature_names']
            self.cluster_labels = model_data['cluster_labels']
            self.n_clusters = model_data['n_clusters']
            self.is_fitted = model_data['is_fitted']

            logger.info(f"Loaded clusterer from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load clusterer: {e}")


if __name__ == "__main__":
    # Test context clustering
    logging.basicConfig(level=logging.INFO)

    print("=== Context Clusterer Test ===\n")

    # Generate synthetic contexts
    np.random.seed(42)
    contexts = []

    # High-value transactions
    for _ in range(30):
        contexts.append({
            'transaction_amount': np.random.uniform(500_000, 2_000_000),
            'data_quality_score': np.random.uniform(0.7, 0.95),
            'urgency': np.random.choice(['medium', 'high']),
            'has_swift_reference': True,
            'rule_pass_count': np.random.randint(8, 12),
            'rule_fail_count': 0,
            'ml_confidence': np.random.uniform(0.7, 0.95),
            'genai_confidence': np.random.uniform(0.7, 0.95),
            'assurance_score': np.random.uniform(0.7, 0.95)
        })

    # Routine transactions
    for _ in range(50):
        contexts.append({
            'transaction_amount': np.random.uniform(5_000, 50_000),
            'data_quality_score': np.random.uniform(0.8, 0.95),
            'urgency': 'low',
            'has_swift_reference': True,
            'rule_pass_count': np.random.randint(6, 10),
            'rule_fail_count': 0,
            'ml_confidence': np.random.uniform(0.8, 0.95),
            'genai_confidence': np.random.uniform(0.8, 0.95),
            'assurance_score': np.random.uniform(0.8, 0.95)
        })

    # Low quality data
    for _ in range(20):
        contexts.append({
            'transaction_amount': np.random.uniform(10_000, 200_000),
            'data_quality_score': np.random.uniform(0.2, 0.5),
            'urgency': np.random.choice(['low', 'medium']),
            'has_swift_reference': False,
            'rule_pass_count': np.random.randint(3, 7),
            'rule_fail_count': np.random.randint(1, 3),
            'ml_confidence': np.random.uniform(0.4, 0.7),
            'genai_confidence': np.random.uniform(0.4, 0.7),
            'assurance_score': np.random.uniform(0.3, 0.6)
        })

    # Fit clusterer
    clusterer = ContextClusterer(n_clusters=3)
    clusterer.fit(contexts)

    # Get statistics
    stats = clusterer.get_cluster_stats(contexts)

    print("\nCluster Statistics:")
    for cluster in stats['clusters']:
        print(f"\nCluster {cluster['id']}: {cluster['label']}")
        print(f"  Count: {cluster['count']} ({cluster['percentage']:.1f}%)")
        print(f"  Avg Amount: ${cluster['avg_transaction_amount']:,.0f}")
        print(f"  Avg Quality: {cluster['avg_data_quality']:.2f}")

    # Test prediction
    print("\n\nTest Prediction:")
    test_context = {
        'transaction_amount': 1_250_000,
        'data_quality_score': 0.88,
        'urgency': 'high',
        'has_swift_reference': True,
        'rule_pass_count': 10,
        'rule_fail_count': 0,
        'ml_confidence': 0.85,
        'genai_confidence': 0.82,
        'assurance_score': 0.87
    }

    cluster_id = clusterer.predict(test_context)
    print(f"Context assigned to cluster: {cluster_id} ({clusterer.cluster_labels[cluster_id]})")
