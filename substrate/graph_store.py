"""
Graph Store implementation using Neo4j

Handles entity relationships, lineage tracking, and graph traversal.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from common.schemas import Entity, GraphEdge, GraphNode

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Neo4j-based graph store for entity relationships

    Features:
    - Store entities as nodes with properties
    - Create typed relationships between entities
    - Graph traversal and path finding
    - Lineage tracking
    - Cypher query execution
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j URI
            user: Database user
            password: Database password
        """
        self.uri = uri
        self.user = user

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"✅ Connected to Neo4j at {uri}")

        except ServiceUnavailable as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_node(
        self,
        node_id: str,
        label: str,
        properties: Dict[str, Any],
    ) -> GraphNode:
        """
        Create a node in the graph

        Args:
            node_id: Unique node ID
            label: Node label/type
            properties: Node properties

        Returns:
            GraphNode object
        """
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{label} {{id: $node_id}})
            SET n += $properties
            RETURN n
            """
            result = session.run(query, node_id=node_id, properties=properties)
            record = result.single()

            if record:
                node_data = dict(record["n"])
                logger.debug(f"Created node {label}:{node_id}")
                return GraphNode(
                    id=node_id,
                    label=label,
                    properties=node_data
                )

    def create_nodes_batch(
        self,
        nodes: List[Tuple[str, str, Dict[str, Any]]]
    ) -> int:
        """
        Create multiple nodes in batch

        Args:
            nodes: List of (node_id, label, properties) tuples

        Returns:
            Number of nodes created
        """
        with self.driver.session() as session:
            query = """
            UNWIND $nodes AS node
            MERGE (n {id: node.id})
            SET n += node.properties
            SET n :node.label
            """
            # Format for Neo4j
            batch_data = [
                {"id": nid, "label": label, "properties": props}
                for nid, label, props in nodes
            ]
            result = session.run(query, nodes=batch_data)
            logger.info(f"Created {len(nodes)} nodes in batch")
            return len(nodes)

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        """
        Create a relationship between two nodes

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Relationship type (e.g., "LINKS_TO", "PAID_BY")
            properties: Optional relationship properties

        Returns:
            GraphEdge object
        """
        props = properties or {}

        with self.driver.session() as session:
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r += $properties
            RETURN r
            """
            result = session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=props
            )

            if result.single():
                logger.debug(f"Created relationship {source_id}-[{relationship_type}]->{target_id}")
                return GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=relationship_type,
                    properties=props
                )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Get a node by ID

        Args:
            node_id: Node ID

        Returns:
            GraphNode or None if not found
        """
        with self.driver.session() as session:
            query = """
            MATCH (n {id: $node_id})
            RETURN n, labels(n) AS labels
            """
            result = session.run(query, node_id=node_id)
            record = result.single()

            if record:
                node_data = dict(record["n"])
                labels = record["labels"]
                return GraphNode(
                    id=node_id,
                    label=labels[0] if labels else "Node",
                    properties=node_data
                )
            return None

    def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",  # "in", "out", or "both"
    ) -> List[GraphNode]:
        """
        Get neighboring nodes

        Args:
            node_id: Center node ID
            relationship_type: Filter by relationship type (optional)
            direction: Direction ("in", "out", "both")

        Returns:
            List of neighbor nodes
        """
        rel_clause = f"[r:{relationship_type}]" if relationship_type else "[r]"

        if direction == "out":
            pattern = f"(n {{id: $node_id}})-{rel_clause}->(m)"
        elif direction == "in":
            pattern = f"(n {{id: $node_id}})<-{rel_clause}-(m)"
        else:  # both
            pattern = f"(n {{id: $node_id}})-{rel_clause}-(m)"

        with self.driver.session() as session:
            query = f"""
            MATCH {pattern}
            RETURN m, labels(m) AS labels
            """
            result = session.run(query, node_id=node_id)

            neighbors = []
            for record in result:
                node_data = dict(record["m"])
                labels = record["labels"]
                neighbors.append(GraphNode(
                    id=node_data.get("id"),
                    label=labels[0] if labels else "Node",
                    properties=node_data
                ))

            return neighbors

    def find_path(
        self,
        start_id: str,
        end_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 5,
    ) -> Optional[List[GraphNode]]:
        """
        Find shortest path between two nodes

        Args:
            start_id: Start node ID
            end_id: End node ID
            relationship_type: Filter by relationship type
            max_depth: Maximum path length

        Returns:
            List of nodes in path, or None if no path found
        """
        rel_clause = f"[r:{relationship_type}*1..{max_depth}]" if relationship_type else f"[*1..{max_depth}]"

        with self.driver.session() as session:
            query = f"""
            MATCH path = shortestPath((start {{id: $start_id}})-{rel_clause}-(end {{id: $end_id}}))
            RETURN nodes(path) AS path_nodes
            """
            result = session.run(query, start_id=start_id, end_id=end_id)
            record = result.single()

            if record:
                path_nodes = []
                for node in record["path_nodes"]:
                    node_dict = dict(node)
                    path_nodes.append(GraphNode(
                        id=node_dict.get("id"),
                        label=list(node.labels)[0] if node.labels else "Node",
                        properties=node_dict
                    ))
                return path_nodes

            return None

    def find_connected_component(
        self,
        node_id: str,
        max_depth: int = 3,
    ) -> List[GraphNode]:
        """
        Find all nodes connected to a given node (BFS traversal)

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth

        Returns:
            List of connected nodes
        """
        with self.driver.session() as session:
            query = f"""
            MATCH (start {{id: $node_id}})-[*0..{max_depth}]-(connected)
            RETURN DISTINCT connected, labels(connected) AS labels
            """
            result = session.run(query, node_id=node_id)

            nodes = []
            for record in result:
                node_data = dict(record["connected"])
                labels = record["labels"]
                nodes.append(GraphNode(
                    id=node_data.get("id"),
                    label=labels[0] if labels else "Node",
                    properties=node_data
                ))

            return nodes

    def query_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        params = parameters or {}

        with self.driver.session() as session:
            result = session.run(query, **params)
            records = []

            for record in result:
                records.append(dict(record))

            return records

    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and all its relationships

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self.driver.session() as session:
            query = """
            MATCH (n {id: $node_id})
            DETACH DELETE n
            RETURN count(n) AS deleted
            """
            result = session.run(query, node_id=node_id)
            record = result.single()

            if record and record["deleted"] > 0:
                logger.debug(f"Deleted node {node_id}")
                return True
            return False

    def clear_graph(self) -> int:
        """
        Delete all nodes and relationships (use with caution!)

        Returns:
            Number of nodes deleted
        """
        with self.driver.session() as session:
            query = """
            MATCH (n)
            DETACH DELETE n
            RETURN count(n) AS deleted
            """
            result = session.run(query)
            record = result.single()
            count = record["deleted"] if record else 0

            logger.warning(f"Cleared graph: {count} nodes deleted")
            return count

    def get_node_count(self) -> int:
        """Get total number of nodes in graph"""
        with self.driver.session() as session:
            query = "MATCH (n) RETURN count(n) AS count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def get_relationship_count(self) -> int:
        """Get total number of relationships in graph"""
        with self.driver.session() as session:
            query = "MATCH ()-[r]->() RETURN count(r) AS count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics

        Returns:
            Dict with node count, relationship count, labels, etc.
        """
        with self.driver.session() as session:
            # Get node count
            node_count = self.get_node_count()

            # Get relationship count
            rel_count = self.get_relationship_count()

            # Get all labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            # Get all relationship types
            rel_types_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rel_types_result]

            return {
                "node_count": node_count,
                "relationship_count": rel_count,
                "labels": labels,
                "relationship_types": rel_types,
            }


# Singleton instance
_graph_store: Optional[GraphStore] = None


def get_graph_store(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
) -> GraphStore:
    """
    Get or create singleton GraphStore instance

    Args:
        uri: Neo4j URI
        user: Database user
        password: Database password

    Returns:
        GraphStore instance
    """
    global _graph_store

    if _graph_store is None:
        _graph_store = GraphStore(uri=uri, user=user, password=password)

    return _graph_store
