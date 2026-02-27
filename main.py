"""
Phase 1 — Test NLP baseline and upload triples to Neo4j.
Phase 2 — LLM pipeline will be plugged in later (marked with TODO).
"""

import json
import logging
from config import Config
from data_loader import load_data
from nlp_baseline import run_nlp_pipeline
from graph_manager import GraphManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Configuration
# Set these flags to control what runs each time you execute main.py
# ──────────────────────────────────────────────────────────────────────────────

RUN_NLP    = True   # Run NLP baseline pipeline
RUN_LLM    = False  # TODO: enable in Phase 2
UPLOAD_NEO4J = True # Upload triples to Neo4j
CLEAR_BEFORE_UPLOAD = False  # Set True to wipe old data before re-uploading
SAMPLE_SIZE = 10 # small sample for testing
# SAMPLE_SIZE = Config.SAMPLE_SIZE  # Change to None to run on full dataset


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_reports() -> list:
    """Load and clean traffic reports from CSV."""
    logger.info("Loading reports...")
    reports = load_data(sample_size=SAMPLE_SIZE, save_processed=True)
    logger.info(f"Loaded {len(reports)} reports.")
    return reports


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Pipeline Runners
# ──────────────────────────────────────────────────────────────────────────────

def run_nlp(reports: list) -> list:
    """Run NLP baseline and save triples to data/nlp_triples.json."""
    logger.info("=== Running NLP Baseline Pipeline ===")
    results = run_nlp_pipeline(reports, save_results=True)
    logger.info(f"NLP complete: {len(results)} cases, "
                f"{sum(r['triple_count'] for r in results)} total triples.")
    return results


# TODO Phase 2: Uncomment and implement when llm_pipeline.py is ready
# def run_llm(reports: list) -> list:
#     from llm_pipeline import run_llm_pipeline
#     logger.info("=== Running LLM Pipeline ===")
#     results = run_llm_pipeline(reports, save_results=True)
#     logger.info(f"LLM complete: {len(results)} cases, "
#                 f"{sum(r['triple_count'] for r in results)} total triples.")
#     return results


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Neo4j Upload
# ──────────────────────────────────────────────────────────────────────────────

def upload_to_neo4j(nlp_results: list):
    """Connect to Neo4j and upload NLP triples."""
    gm = GraphManager()

    if not gm.verify_connection():
        logger.error("Cannot reach Neo4j. Check .env credentials and database status.")
        gm.close()
        return

    try:
        if CLEAR_BEFORE_UPLOAD:
            logger.info("Clearing existing NLP data from Neo4j...")
            gm.clear_pipeline_data("nlp")

        if nlp_results:
            logger.info("Uploading NLP triples to Neo4j...")
            nlp_summary = gm.upload_pipeline_results(nlp_results, pipeline_tag="nlp")
            logger.info(f"NLP upload summary: {nlp_summary}")

        # TODO Phase 2: upload LLM results
        # if llm_results:
        #     llm_summary = gm.upload_pipeline_results(llm_results, pipeline_tag="llm")
        #     logger.info(f"LLM upload summary: {llm_summary}")

        # Final graph stats sanity check
        stats = gm.get_graph_stats()
        logger.info(f"Graph state → Nodes: {stats['nodes']}, "
                    f"Relationships: {stats['relationships']}")

    finally:
        gm.close()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: Save Run Summary
# ──────────────────────────────────────────────────────────────────────────────

def save_summary(nlp_results: list):
    """Save a lightweight run summary to data/processed/run_summary.json."""
    summary = {
        "total_cases":       len(nlp_results),
        "total_triples":     sum(r["triple_count"] for r in nlp_results),
        "avg_triples":       round(
            sum(r["triple_count"] for r in nlp_results) / len(nlp_results), 2
        ) if nlp_results else 0,
        "total_time_s":      round(
            sum(r["processing_time_s"] for r in nlp_results), 4
        ),
        "avg_time_s":        round(
            sum(r["processing_time_s"] for r in nlp_results) / len(nlp_results), 4
        ) if nlp_results else 0,
        # TODO Phase 2: add llm_results summary here
    }
    out_path = Config.DATA_DIR / "run_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Run summary saved → {out_path}")
    logger.info(f"Summary: {summary}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Config.ensure_dirs()

    # Step 1: Load data
    reports = load_reports()
    if not reports:
        logger.error("No reports loaded. Check your CSV file in data/raw_traffic_reports/")
        exit(1)

    nlp_results = []

    # Step 2: Run NLP pipeline
    if RUN_NLP:
        nlp_results = run_nlp(reports)

    # Step 3: Upload to Neo4j
    if UPLOAD_NEO4J and nlp_results:
        upload_to_neo4j(nlp_results)

    # Step 4: Save summary
    if nlp_results:
        save_summary(nlp_results)

    logger.info("=== Phase 1 Complete ===")
