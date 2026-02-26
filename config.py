# run by dedicate terminal

from perplexity import Perplexity
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List,Dict

# Load environment variables from .env file
load_dotenv("environment_test.env")


# client = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY")) # Uses PERPLEXITY_API_KEY from .env file

# response = client.responses.create(
#     model = "anthropic/claude-haiku-4-5",
#     input = "Explain what is knowledge graph.",
#     max_output_tokens = 500
# )

# print(f"Response ID: {response.id}")
# print(response.output_text)

class Config:
    # data path
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_CSV_DIR = DATA_DIR / "raw_traffic_reports"
    PROCESSED_DIR = DATA_DIR / "processed_traffic_reports"

    # API
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    LLM_MODEL = "anthropic/claude-haiku-4-5"
    MAX_OUTPUT_TOKENS = 1500
    TEMPERATURE = 0.0

    # Neo4j database
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    ## create a .env file to save URI, USER and PASSWORD for Neo4j

    # CSV format
    CSV_TEXT_COLUMN = 0
    CSV_HAS_HEADER = True
    CSV_ENCODING = "utf-8"

    # Experiment settings
    SAMPLE_SIZE = 100
    FEW_SHOT_COUNT = 5
    
    # KG ontology
    ENTITY_TYPES: List[str] = [
        "AccidentCase", "Person", "Vehicle", "Road", "Location", "Time", 
        "Environment", "Behavior", "Cause", "MainCause", "Severity", 
        "AccidentType", "CasualtyLoss", "Department", "Judgment"
    ]
    
    RELATIONSHIP_TYPES: List[str] = [
        "CAUSE", "INVOLVE", "OCCUR_AT", "OCCUR_IN", "AFFECTED_BY", 
        "BELONG_TO", "MEASURE", "RESULT_IN", "INCLUDE", "BECAUSE_OF",
        "LOCATED_IN", "JURISDICTION", "RESPONSIBILITY"
    ]

    # Few-shot examples for LLM prompt (hard-coded for consistency)
    FEW_SHOT_EXAMPLES: List[Dict] = [
    {
        "text": "At 3 PM on Jan 1, drunk driver Zhang ran red light on Highway 1, hitting pedestrian Li.",
        "triples": [
            ("Person", "Zhang", "drunk driving"),
            ("AccidentCase", "Jan 1 Highway 1 crash", "CAUSE", "Person", "Zhang"),
            ("AccidentCase", "Jan 1 Highway 1 crash", "INVOLVE", "Person", "Li"),
            ("AccidentCase", "Jan 1 Highway 1 crash", "OCCUR_AT", "Time", "3 PM Jan 1"),
            ("AccidentCase", "Jan 1 Highway 1 crash", "OCCUR_IN", "Road", "Highway 1")
        ]
    },
    {
        "text": "Rear-end collision at night due to rain. Driver fatigue caused major injuries.",
        "triples": [
            ("Behavior", "driver fatigue", "CAUSE", "AccidentCase", "rear-end collision"),
            ("Environment", "rainy night", "AFFECTED_BY", "AccidentCase", "rear-end collision"),
            ("AccidentCase", "rear-end collision", "BELONG_TO", "AccidentType", "rear-end"),
            ("AccidentCase", "rear-end collision", "MEASURE", "Severity", "major"),
            ("AccidentCase", "rear-end collision", "RESULT_IN", "CasualtyLoss", "major injuries")
        ]
    },
    {
        "text": "Head-on crash on urban road during morning rush hour. Speeding truck vs sedan. 2 deaths.",
        "triples": [
            ("Vehicle", "speeding truck", "INVOLVE", "AccidentCase", "head-on crash"),
            ("Vehicle", "sedan", "INVOLVE", "AccidentCase", "head-on crash"),
            ("AccidentCase", "head-on crash", "OCCUR_IN", "Road", "urban road"),
            ("Time", "morning rush hour", "OCCUR_AT", "AccidentCase", "head-on crash"),
            ("AccidentCase", "head-on crash", "BELONG_TO", "AccidentType", "head-on"),
            ("CasualtyLoss", "2 deaths", "RESULT_IN", "AccidentCase", "head-on crash"),
            ("Cause", "speeding", "CAUSE", "AccidentCase", "head-on crash")
        ]
    },
    {
        "text": "Pedestrian hit by motorcycle at crosswalk. Foggy weather, poor visibility. Minor injuries.",
        "triples": [
            ("Person", "pedestrian", "INVOLVE", "AccidentCase", "pedestrian hit"),
            ("Vehicle", "motorcycle", "INVOLVE", "AccidentCase", "pedestrian hit"),
            ("Environment", "foggy poor visibility", "AFFECTED_BY", "AccidentCase", "pedestrian hit"),
            ("AccidentCase", "pedestrian hit", "MEASURE", "Severity", "minor"),
            ("AccidentCase", "pedestrian hit", "OCCUR_IN", "Location", "crosswalk")
        ]
    },
    {
        "text": "Multi-vehicle pileup on icy highway. Primary cause: failure to maintain safe distance.",
        "triples": [
            ("MainCause", "human error", "INCLUDE", "Cause", "failure to maintain distance"),
            ("Cause", "failure to maintain distance", "CAUSE", "AccidentCase", "multi-vehicle pileup"),
            ("Environment", "icy highway", "AFFECTED_BY", "AccidentCase", "multi-vehicle pileup"),
            ("AccidentCase", "multi-vehicle pileup", "RESULT_IN", "CasualtyLoss", "multiple injuries"),
            ("Department", "Highway Patrol", "JURISDICTION", "Road", "icy highway")
        ]
    }
]

    EXTRACTION_PROMPT = """
You are a traffic accident analysis and Knowledge Graph construction expert. Extract {entity_types} and {relationship_types} as triples.
Output ONLY valid JSON triples: [{{"head": "...", "relation": "...", "tail": "..."}}]

Allowed entity types: {entity_types}
Allowed relationship types: {relationship_types}

Few-shot examples:
{few_show_examples}

Now extract from: "{text}"
"""
    
    # checked esitency of API, Few-shot examples, directories.
    @staticmethod
    def get_llm_client():
        if not Config.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY not found in .env file.")
        return Perplexity(api_key=Config.PERPLEXITY_API_KEY)

    @staticmethod
    def build_extraction_prompt(text: str) -> str:
        """Builds complete LLM prompt with few-shot."""
        examples_str = "\n".join([
            f"Text: {ex['text']}\nTriples: {ex['triples']}\n"
            for ex in Config.FEW_SHOT_EXAMPLES[:Config.FEW_SHOT_COUNT]
        ])
        return Config.EXTRACTION_PROMPT.format(
            entity_types = ", ".join(Config.ENTITY_TYPES),
            relationship_types = ", ".join(Config.RELATIONSHIP_TYPES),
            few_show_examples = examples_str,
            text = text
        )
    
    @staticmethod
    def ensure_dirs():
        """Ensure necessary directories exist."""
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        print("Directories ready!")

# Test
if __name__ == "__main__":
    Config.ensure_dirs()
    print(f"Raw CSV dir  : {Config.RAW_CSV_DIR}")
    print(f"Processed dir: {Config.PROCESSED_DIR}")
    print(f"API Key found: {'Y' if Config.PERPLEXITY_API_KEY else 'Missing!'}")
    print(f"Neo4j URI    : {Config.NEO4J_URI}")
    print(f"Entities     : {len(Config.ENTITY_TYPES)} types")
    print(f"Relations    : {len(Config.RELATIONSHIP_TYPES)} types")
    print(f"Few-shot     : {len(Config.FEW_SHOT_EXAMPLES)} examples")

    # Preview prompt
    sample_text = "The accident occurred in City A at 6:00 am on 2/3/2022."
    prompt = Config.build_extraction_prompt(sample_text)
    print(f"\n Prompt preview (first 300 chars):\n{prompt[:300]}...")