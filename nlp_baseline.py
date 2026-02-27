""" Pipeline 1: Traditional NLP Baseline (spaCy + Regex)
- spaCy en_core_web_lg handles NER (dates, locations, persons, cardinals)
- Regex handles semi-structured fields (Person block, vehicle, road conditions)
- Outputs triples in {"head", "relation", "tail"} format matching llm_pipeline.py
"""

import re
import json
import time
import spacy
from typing import List, Dict
from config import Config

# ─── Load spaCy Model ─────────────────────────────────────────────────────────
# Run once: python -m spacy download en_core_web_lg
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise OSError(" spaCy model not found. Run: python -m spacy download en_core_web_lg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: spaCy NER Extraction
# Extracts standard entities: DATE, TIME, GPE, CARDINAL, PERSON
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spacy_entities(text: str) -> Dict[str, List[str]]:
    """
    Run spaCy NER and return entities grouped by label.

    Returns dict like:
    {
        "DATE":     ["March 2, 2022"],
        "TIME":     ["5:00 AM"],
        "GPE":      ["Richland", "Benton"],
        "CARDINAL": ["1", "24"],
        "PERSON":   ["Person 1"]
    }
    """
    doc = nlp(text)
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text.strip())
    return entities


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Regex Pattern Extractors
# Each function targets one specific semi-structured field in the report
# ═══════════════════════════════════════════════════════════════════════════════

def extract_datetime(text: str) -> str | None:
    """Extract date and time: 'occurred on March 2, 2022, at 5:00 AM'"""
    match = re.search(
        r"occurred on\s+([A-Za-z]+ \d{1,2},\s*\d{4}),?\s+at\s+(\d{1,2}:\d{2}\s*[APap][Mm])",
        text
    )
    return f"{match.group(1)} {match.group(2)}" if match else None


def extract_location(text: str) -> str | None:
    """Extract city/county: 'in Richland, Benton'"""
    match = re.search(r"\bin\s+([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+?)(?:,\s*on route|\.|$)", text)
    return match.group(1).strip() if match else None


def extract_route(text: str) -> str | None:
    """Extract road/route: 'on route 182'"""
    match = re.search(r"on route\s+(\w+)", text, re.IGNORECASE)
    return f"Route {match.group(1)}" if match else None


def extract_road_type(text: str) -> str | None:
    """Extract road classification: 'urban freeways with fewer than 4 lanes'"""
    match = re.search(r"road classification is\s+([^.]+)\.", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_vehicle_count(text: str) -> str | None:
    """Extract number of vehicles: '1 vehicle involved'"""
    match = re.search(r"(\d+)\s+vehicle[s]?\s+involved", text, re.IGNORECASE)
    return f"{match.group(1)} vehicle(s)" if match else None


def extract_environment(text: str) -> List[str]:
    """
    Extract weather/lighting conditions.
    Looks for: 'at dawn', 'wet road surface', 'rainy', 'foggy', etc.
    """
    conditions = []
    lighting_match = re.search(r"(?:were\s+)?at\s+(dawn|dusk|dark|daylight|night)", text, re.IGNORECASE)
    surface_match  = re.search(r"(wet|dry|icy|snowy|muddy)\s+road surface", text, re.IGNORECASE)
    weather_match  = re.search(r"(rain|snow|fog|wind|clear|cloudy)\w*", text, re.IGNORECASE)

    if lighting_match: conditions.append(lighting_match.group(1).lower())
    if surface_match:  conditions.append(f"{surface_match.group(1).lower()} road surface")
    if weather_match:  conditions.append(weather_match.group(1).lower())
    return conditions


def extract_cause(text: str) -> List[str]:
    """
    Extract causes via keyword matching against known cause patterns.
    Maps to CAUSE relationship in ontology.
    """
    cause_patterns = {
        "speeding":               r"exceeding a reasonable safe speed|over.?speed|speeding",
        "drunk driving":          r"influence of alcohol|drunk driving|DUI",
        "drug influence":         r"influence of drugs|under the influence of drug",
        "distracted driving":     r"distracted|using phone|inattention",
        "failure to yield":       r"failure to yield|did not yield",
        "ran red light":          r"ran a? red light|ran the red light",
        "improper lane change":   r"improper lane change|unsafe lane",
        "road defect":            r"road defect|pavement failure|pothole",
    }
    found = []
    for cause, pattern in cause_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(cause)
    return found


def extract_objects(text: str) -> List[str]:
    """Extract fixed objects involved: 'a Roadway Ditch', 'a Guard Rail'"""
    match = re.search(r"specifically\s+(?:a\s+)?([A-Za-z\s]+?)(?:\.|,|$)", text, re.IGNORECASE)
    return [match.group(1).strip()] if match else []


def extract_persons(text: str) -> List[Dict]:
    """
    Extract Person blocks.
    Pattern: 'Person 1: Motor Vehicle Driver, Female, 24, Lap & Shoulder Used'
    Returns list of dicts with role, gender, age, restraint.
    """
    persons = []
    pattern = re.finditer(
        r"Person\s+(\d+):\s+([^,]+),\s+(Male|Female),\s+(\d+)(?:,\s+([^.\n]+))?",
        text, re.IGNORECASE
    )
    for match in pattern:
        persons.append({
            "id":        f"Person{match.group(1)}",
            "role":      match.group(2).strip(),
            "gender":    match.group(3).strip(),
            "age":       match.group(4).strip(),
            "restraint": match.group(5).strip() if match.group(5) else "Unknown"
        })
    return persons


def extract_vehicles(text: str) -> List[Dict]:
    """
    Extract Vehicle info blocks.
    Pattern: 'Vehicle1 was moving east... The first vehicle was moving straight'
    """
    vehicles = []
    pattern = re.finditer(
        r"Vehicle\s*(\d+)\s+was\s+moving\s+([a-zA-Z]+)",
        text, re.IGNORECASE
    )
    for match in pattern:
        vehicles.append({
            "id":        f"Vehicle{match.group(1)}",
            "direction": match.group(2).strip()
        })

    # Extract vehicle type: 'non-commercial vehicle', 'truck', etc.
    type_match = re.search(r"(non-commercial|commercial|truck|motorcycle|bus)\s+vehicle", text, re.IGNORECASE)
    if type_match and vehicles:
        vehicles[0]["type"] = type_match.group(0).strip()

    return vehicles


def extract_severity(text: str) -> str | None:
    """Infer severity from casualty keywords."""
    if re.search(r"\d+\s+facilit|fatal|death|killed", text, re.IGNORECASE):
        return "Fatal"
    if re.search(r"\d+\s+injur|serious injur", text, re.IGNORECASE):
        return "Injury"
    if re.search(r"property damage|no injur", text, re.IGNORECASE):
        return "Property Damage"
    return None


def extract_casualties(text: str) -> List[str]:
    """Extract casualty counts: '2 injured', '1 fatality'"""
    casualties = []
    for pattern in [r"(\d+)\s+injur\w+", r"(\d+)\s+facilit\w+", r"(\d+)\s+death\w*"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            casualties.append(match.group(0).strip())
    return casualties


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Triple Builder
# Assembles all extracted info into ontology-aligned triples
# ═══════════════════════════════════════════════════════════════════════════════

def build_triples(text: str, case_id: str) -> List[Dict]:
    """
    Combine spaCy NER + regex extractions into a list of triples.
    All triples use case_id as the head anchor (AccidentCase node).
    """
    triples = []
    spacy_ents = extract_spacy_entities(text)

    # Helper: safely add triple if tail is not None/empty
    def add(relation: str, tail):
        if tail:
            triples.append({"head": case_id, "relation": relation, "tail": str(tail)})

    # ── Time ──────────────────────────────────────────────────────────────────
    # Regex first (more precise for this format), spaCy as fallback
    datetime_val = extract_datetime(text)
    if datetime_val:
        add("OCCUR_AT", datetime_val)
    elif spacy_ents.get("DATE") or spacy_ents.get("TIME"):
        combined = " ".join(spacy_ents.get("DATE", []) + spacy_ents.get("TIME", []))
        add("OCCUR_AT", combined)

    # ── Location ──────────────────────────────────────────────────────────────
    location_val = extract_location(text)
    if location_val:
        add("OCCUR_IN", location_val)
    else:
        # Fallback: use spaCy GPE entities
        for gpe in spacy_ents.get("GPE", []):
            add("OCCUR_IN", gpe)

    # ── Road ──────────────────────────────────────────────────────────────────
    add("OCCUR_IN", extract_route(text))
    add("BELONG_TO", extract_road_type(text))

    # ── Environment ───────────────────────────────────────────────────────────
    for condition in extract_environment(text):
        add("AFFECTED_BY", condition)

    # ── Cause ─────────────────────────────────────────────────────────────────
    for cause in extract_cause(text):
        triples.append({"head": cause, "relation": "CAUSE", "tail": case_id})

    # ── Vehicles ──────────────────────────────────────────────────────────────
    for vehicle in extract_vehicles(text):
        add("INVOLVE", vehicle["id"])

    # Fallback vehicle count from regex
    if not extract_vehicles(text):
        add("INVOLVE", extract_vehicle_count(text))

    # ── Objects ───────────────────────────────────────────────────────────────
    for obj in extract_objects(text):
        add("INVOLVE", obj)

    # ── Persons ───────────────────────────────────────────────────────────────
    for person in extract_persons(text):
        add("INVOLVE", person["id"])
        # Person-level triples
        triples.append({"head": person["id"], "relation": "INVOLVE", "tail": person["role"]})
        triples.append({"head": person["id"], "relation": "INVOLVE", "tail": f"{person['gender']}, Age {person['age']}"})

    # ── Severity ──────────────────────────────────────────────────────────────
    add("MEASURE", extract_severity(text))

    # ── Casualties ────────────────────────────────────────────────────────────
    for casualty in extract_casualties(text):
        add("RESULT_IN", casualty)

    return triples


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Main Pipeline Runner
# Processes all reports, tracks timing, saves results
# ═══════════════════════════════════════════════════════════════════════════════

def run_nlp_pipeline(reports: List[Dict], save_results: bool = True) -> List[Dict]:
    """
    Run NLP extraction on all cleaned reports from data_loader.py.

    Args:
        reports     : Output of load_data() — list of {"source", "id", "text"}
        save_results: Save output to data/processed/nlp_triples.json

    Returns:
        List of dicts: [{"case_id", "source", "triples", "processing_time_s"}, ...]
    """
    results = []

    for report in reports:
        case_id   = f"{report['source']}_{report['id']}"
        text      = report["text"]

        start     = time.time()
        triples   = build_triples(text, case_id)
        elapsed   = round(time.time() - start, 4)

        results.append({
            "case_id":            case_id,
            "source":             report["source"],
            "triples":            triples,
            "triple_count":       len(triples),
            "processing_time_s":  elapsed
        })

    total_time = sum(r["processing_time_s"] for r in results)
    avg_triples = sum(r["triple_count"] for r in results) / len(results) if results else 0
    print(f"NLP Pipeline complete: {len(results)} reports")
    print(f"Total time   : {round(total_time, 2)}s")
    print(f"Avg triples  : {round(avg_triples, 2)} per report")

    if save_results and results:
        out_path = Config.DATA_DIR / "nlp_triples.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved → {out_path}")

    return results


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [{
        "source": "WA",
        "id": 0,
        "text": (
            "This incident occurred on March 2, 2022, at 5:00 AM, in Richland, Benton, "
            "on route 182 increasing milepost direction at milepost 0.25. "
            "The road classification is urban freeways with fewer than 4 lanes. "
            "The conditions during the time of the accident were at dawn with a wet road surface. "
            "There were no pedestrians involved, 1 vehicle involved. "
            "There were objects involved, specifically a Roadway Ditch. "
            "Vehicle1 was moving east, in the direction of increasing milepost. "
            "The driver was going straight ahead, was not ejected, and was exceeding a reasonable safe speed. "
            "Person 1: Motor Vehicle Driver, Female, 24, Lap & Shoulder Used."
        )
    }]

    results = run_nlp_pipeline(sample, save_results=False)
    for triple in results[0]["triples"]:
        print(triple)
