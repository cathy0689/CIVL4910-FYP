# run by dedicate terminal

from perplexity import Perplexity
import os
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
    # API
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    LLM_MODEL = "anthropic/claude-haiku-4-5"

    # Neo4j database
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    ## create a .env file to save URI, USER and PASSWORD for Neo4j

    # Experiment settings
    SAMPLE_SIZE = 100
    FEW_SHOT_EXAMPLES = 5
    MAX_OUTPUT_TOKENS = 1500
    TEMPERATURE = 0.0

    @staticmethod
    def get_llm_client():
        if not Config.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY not found in .env file.")
        return Perplexity(api_key=Config.PERPLEXITY_API_KEY)
    
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
You are a traffic accident KG expert. Extract {entity_types} and {relations} as triples.

Few-shot examples:
{examples}

Extract from: "{text}"
Output ONLY valid JSON triples: [{{"head": "...", "relation": "...", "tail": "..."}}]
"""
    
    @staticmethod
    def build_extraction_prompt(text: str) -> str:
        """Builds complete LLM prompt with few-shot."""
        examples_str = "\n".join([
            f"Text: {ex['text']}\nTriples: {ex['triples']}\n"
            for ex in Config.FEW_SHOT_EXAMPLES[:Config.FEW_SHOT_EXAMPLES]
        ])
        return Config.EXTRACTION_PROMPT.format(
            entity_types=", ".join(Config.ENTITY_TYPES),
            relations=", ".join(Config.RELATIONSHIP_TYPES),
            examples=examples_str,
            text=text
        )

# Test
if __name__ == "__main__":
    print("Config loaded!")
    print(f"Entities: {len(Config.ENTITY_TYPES)} types")
    print(f"Relations: {len(Config.RELATIONSHIP_TYPES)} types")
    print(f"Few-shot: {len(Config.FEW_SHOT_EXAMPLES)} examples")
    print("Ready for kg_extraction!")