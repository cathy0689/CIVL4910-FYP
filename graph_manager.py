"""
- Connects to Neo4j using credentials from Config
- Accepts triples in {"head", "relation", "tail"} format
- Uses MERGE to avoid duplicate nodes/relationships
"""

from neo4j import GraphDatabase
from typing import List, Dict
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphManager:
    """Manages Neo4j connection and Cypher operations."""

    def __init__(self):
        if not Config.NEO4J_URI or not Config.NEO4J_PASSWORD:
            raise ValueError("Neo4j credentials missing in .env file.")
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        logger.info("Neo4j connection established.")

    def close(self):
        """Always call this when done to release the connection."""
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def verify_connection(self) -> bool:
        """Ping Neo4j to confirm the connection is live."""
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j connectivity verified.")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: Node & Relationship Operations
    # ──────────────────────────────────────────────────────────────────────────

    def _merge_node(self, tx, label: str, name: str):
        """
        MERGE a single node by label + name.
        Label is inferred from the ontology if possible (see _infer_label).
        """
        cypher = f"MERGE (n:{label} {{name: $name}})"
        tx.run(cypher, name=name)

    def _merge_relationship(self, tx, head: str, head_label: str,
                             relation: str, tail: str, tail_label: str):
        """MERGE both nodes and the relationship between them."""
        cypher = f"""
        MERGE (h:{head_label} {{name: $head}})
        MERGE (t:{tail_label} {{name: $tail}})
        MERGE (h)-[r:{relation}]->(t)
        """
        tx.run(cypher, head=head, tail=tail)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: Label Inference
    # Maps known relation types to the expected tail entity label
    # ──────────────────────────────────────────────────────────────────────────

    # Relation → expected TAIL label
    RELATION_TAIL_LABEL: Dict[str, str] = {
        "OCCUR_AT":       "Time",
        "OCCUR_IN":       "Location",
        "BELONG_TO":      "AccidentType",
        "AFFECTED_BY":    "Environment",
        "INVOLVE":        "Vehicle",       # overridden for Person* heads
        "MEASURE":        "Severity",
        "RESULT_IN":      "CasualtyLoss",
        "CAUSE":          "AccidentCase",  # head is Cause, tail is AccidentCase
        "BECAUSE_OF":     "Cause",
        "INCLUDE":        "Cause",
        "JURISDICTION":   "Road",
        "RESPONSIBILITY": "Department",
        "LOCATED_IN":     "Location",
    }

    # Relation → expected HEAD label
    RELATION_HEAD_LABEL: Dict[str, str] = {
        "CAUSE":          "Cause",
        "INCLUDE":        "MainCause",
        "JURISDICTION":   "Department",
        "RESPONSIBILITY": "Department",
    }

    def _infer_label(self, node_name: str, relation: str, is_head: bool) -> str:
        """
        Infer the Neo4j node label from the relation type and node name.
        - AccidentCase nodes are detected by the 'WA_' / source prefix pattern.
        - Person nodes are detected by 'Person' prefix.
        - Vehicle nodes are detected by 'Vehicle' prefix.
        - Falls back to ontology relation mapping, then 'Entity'.
        """
        # Detect AccidentCase by source prefix pattern (e.g., "WA_0")
        if "_" in node_name and node_name.split("_")[0].isupper():
            return "AccidentCase"

        # Detect Person/Vehicle by name prefix
        if node_name.startswith("Person"):
            return "Person"
        if node_name.startswith("Vehicle"):
            return "Vehicle"

        # Use relation-based lookup
        if is_head:
            return self.RELATION_HEAD_LABEL.get(relation, "AccidentCase")
        else:
            return self.RELATION_TAIL_LABEL.get(relation, "Entity")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: Batch Upload
    # ──────────────────────────────────────────────────────────────────────────

    def upload_triples(self, triples: List[Dict], pipeline_tag: str = "nlp") -> int:
        """
        Upload a list of {"head", "relation", "tail"} triples to Neo4j.
        Adds a 'pipeline' property to each relationship for traceability.

        Args:
            triples      : List of triple dicts from nlp_baseline or llm_pipeline
            pipeline_tag : "nlp" or "llm" — stored on each relationship

        Returns:
            Number of triples successfully uploaded.
        """
        success_count = 0
        with self.driver.session() as session:
            for triple in triples:
                head     = triple.get("head", "").strip()
                relation = triple.get("relation", "").strip().upper()
                tail     = triple.get("tail", "").strip()

                # Skip malformed triples
                if not head or not relation or not tail:
                    logger.warning(f"Skipping malformed triple: {triple}")
                    continue

                head_label = self._infer_label(head, relation, is_head=True)
                tail_label = self._infer_label(tail, relation, is_head=False)

                try:
                    session.execute_write(
                        self._merge_relationship_with_tag,
                        head, head_label, relation, tail, tail_label, pipeline_tag
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to upload triple {triple}: {e}")

        logger.info(f"Uploaded {success_count}/{len(triples)} triples [{pipeline_tag}].")
        return success_count

    @staticmethod
    def _merge_relationship_with_tag(tx, head: str, head_label: str,
                                      relation: str, tail: str,
                                      tail_label: str, pipeline_tag: str):
        """MERGE nodes + relationship, tag relationship with pipeline source."""
        cypher = f"""
        MERGE (h:{head_label} {{name: $head}})
        MERGE (t:{tail_label} {{name: $tail}})
        MERGE (h)-[r:{relation}]->(t)
        SET r.pipeline = $pipeline_tag
        """
        tx.run(cypher, head=head, tail=tail, pipeline_tag=pipeline_tag)

    def upload_pipeline_results(self, results: List[Dict], pipeline_tag: str = "nlp") -> Dict:
        """
        Upload all triples from a full pipeline result list.
        This is the main function called by main.py.

        Args:
            results      : Output of run_nlp_pipeline() or run_llm_pipeline()
                           Each item: {"case_id", "source", "triples", ...}
            pipeline_tag : "nlp" or "llm"

        Returns:
            Summary dict with total cases, triples attempted, triples uploaded.
        """
        total_triples    = 0
        uploaded_triples = 0

        for result in results:
            triples       = result.get("triples", [])
            total_triples += len(triples)
            uploaded_triples += self.upload_triples(triples, pipeline_tag=pipeline_tag)

        summary = {
            "pipeline":           pipeline_tag,
            "cases_processed":    len(results),
            "triples_attempted":  total_triples,
            "triples_uploaded":   uploaded_triples,
        }
        logger.info(f"Upload summary: {summary}")
        return summary

    def clear_pipeline_data(self, pipeline_tag: str):
        """
        Delete all relationships tagged with a specific pipeline.
        Useful for re-running experiments without duplicating data.
        Also removes orphan nodes (nodes with no relationships).
        """
        with self.driver.session() as session:
            session.run(
                "MATCH ()-[r]->() WHERE r.pipeline = $tag DELETE r",
                tag=pipeline_tag
            )
            session.run(
                "MATCH (n) WHERE NOT (n)--() DELETE n"
            )
        logger.info(f"Cleared all '{pipeline_tag}' data from Neo4j.")

    def get_graph_stats(self) -> Dict:
        """Return basic node/relationship counts for a sanity check."""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        return {"nodes": node_count, "relationships": rel_count}
