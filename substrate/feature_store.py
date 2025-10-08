"""
Feature Store implementation using PostgreSQL

Handles versioned, point-in-time features for ML models.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from common.schemas import Feature, FeatureVector

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Postgres-based feature store for ML features

    Features:
    - Store versioned features with timestamps
    - Point-in-time feature retrieval
    - Feature lineage tracking
    - Batch feature retrieval for training
    - JSON support for complex features
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "qure",
        user: str = "qure",
        password: str = "qure_dev_password",
    ):
        """
        Initialize Postgres connection

        Args:
            host: Postgres host
            port: Postgres port
            database: Database name
            user: Database user
            password: Database password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user

        try:
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )
            self.conn.autocommit = False

            # Create tables if they don't exist
            self._create_tables()

            logger.info(f"✅ Connected to Postgres at {host}:{port}/{database}")

        except psycopg2.Error as e:
            logger.error(f"❌ Failed to connect to Postgres: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Postgres connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_tables(self):
        """Create feature store tables"""
        with self.conn.cursor() as cur:
            # Features table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(255) NOT NULL,
                    feature_name VARCHAR(255) NOT NULL,
                    feature_value JSONB NOT NULL,
                    feature_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER NOT NULL DEFAULT 1,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_entity_time (entity_id, timestamp),
                    INDEX idx_feature_name (feature_name),
                    INDEX idx_timestamp (timestamp)
                )
            """)

            # Feature sets table (for grouping features)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feature_sets (
                    id SERIAL PRIMARY KEY,
                    set_name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    feature_names TEXT[] NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.conn.commit()
            logger.debug("Feature store tables created/verified")

    def store_feature(
        self,
        entity_id: str,
        feature_name: str,
        feature_value: Any,
        feature_type: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Store a single feature

        Args:
            entity_id: Entity this feature describes
            feature_name: Feature name
            feature_value: Feature value (will be JSON serialized)
            feature_type: Feature type (numeric, categorical, etc.)
            timestamp: Point-in-time (defaults to now)
            metadata: Optional metadata

        Returns:
            Feature ID
        """
        ts = timestamp or datetime.utcnow()
        meta = metadata or {}

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO features (entity_id, feature_name, feature_value, feature_type, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (entity_id, feature_name, Json(feature_value), feature_type, ts, Json(meta)))

            feature_id = cur.fetchone()[0]
            self.conn.commit()

            logger.debug(f"Stored feature {feature_name} for entity {entity_id}")
            return feature_id

    def store_feature_vector(
        self,
        feature_vector: FeatureVector,
    ) -> List[int]:
        """
        Store a complete feature vector

        Args:
            feature_vector: FeatureVector object with multiple features

        Returns:
            List of feature IDs
        """
        feature_ids = []

        for feature in feature_vector.features:
            fid = self.store_feature(
                entity_id=feature_vector.entity_id,
                feature_name=feature.name,
                feature_value=feature.value,
                feature_type=feature.feature_type,
                timestamp=feature.timestamp,
            )
            feature_ids.append(fid)

        logger.info(f"Stored {len(feature_ids)} features for entity {feature_vector.entity_id}")
        return feature_ids

    def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Feature]:
        """
        Get a feature value at a specific point in time

        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Point-in-time (defaults to latest)

        Returns:
            Feature object or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if timestamp:
                # Get feature as of timestamp
                cur.execute("""
                    SELECT feature_name, feature_value, feature_type, timestamp
                    FROM features
                    WHERE entity_id = %s AND feature_name = %s AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (entity_id, feature_name, timestamp))
            else:
                # Get latest feature
                cur.execute("""
                    SELECT feature_name, feature_value, feature_type, timestamp
                    FROM features
                    WHERE entity_id = %s AND feature_name = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (entity_id, feature_name))

            row = cur.fetchone()

            if row:
                return Feature(
                    name=row["feature_name"],
                    value=row["feature_value"],
                    feature_type=row["feature_type"],
                    timestamp=row["timestamp"],
                )

            return None

    def get_feature_vector(
        self,
        entity_id: str,
        feature_names: List[str],
        timestamp: Optional[datetime] = None,
    ) -> Optional[FeatureVector]:
        """
        Get multiple features as a vector

        Args:
            entity_id: Entity ID
            feature_names: List of feature names to retrieve
            timestamp: Point-in-time (defaults to latest)

        Returns:
            FeatureVector or None if entity not found
        """
        features = []

        for feature_name in feature_names:
            feature = self.get_feature(entity_id, feature_name, timestamp)
            if feature:
                features.append(feature)

        if features:
            return FeatureVector(
                entity_id=entity_id,
                features=features,
                timestamp=timestamp or datetime.utcnow(),
            )

        return None

    def get_features_batch(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, FeatureVector]:
        """
        Get features for multiple entities in batch

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            timestamp: Point-in-time (defaults to latest)

        Returns:
            Dict mapping entity_id to FeatureVector
        """
        result = {}

        for entity_id in entity_ids:
            fv = self.get_feature_vector(entity_id, feature_names, timestamp)
            if fv:
                result[entity_id] = fv

        return result

    def create_feature_set(
        self,
        set_name: str,
        feature_names: List[str],
        description: Optional[str] = None,
    ) -> int:
        """
        Create a named feature set (group of features)

        Args:
            set_name: Feature set name
            feature_names: List of feature names in the set
            description: Optional description

        Returns:
            Feature set ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_sets (set_name, feature_names, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (set_name) DO UPDATE
                SET feature_names = EXCLUDED.feature_names,
                    description = EXCLUDED.description,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (set_name, feature_names, description))

            set_id = cur.fetchone()[0]
            self.conn.commit()

            logger.info(f"Created/updated feature set '{set_name}' with {len(feature_names)} features")
            return set_id

    def get_feature_set(
        self,
        entity_id: str,
        set_name: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[FeatureVector]:
        """
        Get all features from a named feature set

        Args:
            entity_id: Entity ID
            set_name: Feature set name
            timestamp: Point-in-time

        Returns:
            FeatureVector with all features in the set
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT feature_names FROM feature_sets WHERE set_name = %s
            """, (set_name,))

            row = cur.fetchone()

            if row:
                feature_names = row["feature_names"]
                return self.get_feature_vector(entity_id, feature_names, timestamp)

            return None

    def get_feature_history(
        self,
        entity_id: str,
        feature_name: str,
        limit: int = 100,
    ) -> List[Feature]:
        """
        Get feature value history over time

        Args:
            entity_id: Entity ID
            feature_name: Feature name
            limit: Max number of historical values

        Returns:
            List of Feature objects ordered by timestamp DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT feature_name, feature_value, feature_type, timestamp
                FROM features
                WHERE entity_id = %s AND feature_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (entity_id, feature_name, limit))

            rows = cur.fetchall()

            return [
                Feature(
                    name=row["feature_name"],
                    value=row["feature_value"],
                    feature_type=row["feature_type"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    def delete_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> int:
        """
        Delete features for an entity

        Args:
            entity_id: Entity ID
            feature_names: Optional list of specific features to delete (all if None)

        Returns:
            Number of features deleted
        """
        with self.conn.cursor() as cur:
            if feature_names:
                cur.execute("""
                    DELETE FROM features
                    WHERE entity_id = %s AND feature_name = ANY(%s)
                """, (entity_id, feature_names))
            else:
                cur.execute("""
                    DELETE FROM features WHERE entity_id = %s
                """, (entity_id,))

            deleted = cur.rowcount
            self.conn.commit()

            logger.debug(f"Deleted {deleted} features for entity {entity_id}")
            return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get feature store statistics

        Returns:
            Dict with counts and other stats
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total features
            cur.execute("SELECT COUNT(*) AS total FROM features")
            total = cur.fetchone()["total"]

            # Unique entities
            cur.execute("SELECT COUNT(DISTINCT entity_id) AS unique_entities FROM features")
            entities = cur.fetchone()["unique_entities"]

            # Unique feature names
            cur.execute("SELECT COUNT(DISTINCT feature_name) AS unique_features FROM features")
            features = cur.fetchone()["unique_features"]

            # Feature sets
            cur.execute("SELECT COUNT(*) AS feature_sets FROM feature_sets")
            sets = cur.fetchone()["feature_sets"]

            return {
                "total_feature_records": total,
                "unique_entities": entities,
                "unique_feature_names": features,
                "feature_sets": sets,
            }


# Singleton instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store(
    host: str = "localhost",
    port: int = 5432,
    database: str = "qure",
    user: str = "qure",
    password: str = "qure_dev_password",
) -> FeatureStore:
    """
    Get or create singleton FeatureStore instance

    Args:
        host: Postgres host
        port: Postgres port
        database: Database name
        user: Database user
        password: Database password

    Returns:
        FeatureStore instance
    """
    global _feature_store

    if _feature_store is None:
        _feature_store = FeatureStore(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )

    return _feature_store
