import re
import json
import pandas as pd
from typing import List, Optional
from config import Config


# Dataset Registry
# Add or remove .csv files here. Keys become the "source" label in each record.
CITY_FILES = {
    "WA": Config.RAW_CSV_DIR / "inj.csv",
    # "IL": Config.RAW_CSV_DIR / "inj_IL.csv",
}


# Cleaning
def clean_report(raw_text: str) -> str:
    """Strip LLM chat formatting artifacts and normalize whitespace."""
    text = re.sub(r"<s>\s*Human:\s*", "", raw_text)   # Remove <s>Human:
    text = re.sub(r"</s>.*", "", text, flags=re.DOTALL) # Remove </s> onward
    text = re.sub(r"<[^>]+>", "", text)                 # Remove any <tag>
    text = re.sub(r"<\\+s>", "", text)                  # Remove <\\s> variants
    return re.sub(r"\s+", " ", text).strip()


# Main Loader 
def load_data(
    cities: Optional[List[str]] = None,
    sample_size: Optional[int] = Config.SAMPLE_SIZE,
    save_processed: bool = True
) -> List[dict]:
    """
    Load and clean reports from one or more city CSVs.

    Args:
        cities       : Keys from CITY_FILES to load. None = load all.
        sample_size  : Max reports per file. None = load all rows.
        save_processed: Save cleaned output to data/processed/cleaned_reports.json

    Returns:
        List of dicts: [{"source": "WA", "id": 0, "text": "..."}, ...]
    """
    Config.ensure_dirs()
    targets = {k: v for k, v in CITY_FILES.items() if k in (cities or CITY_FILES)}

    all_records = []
    for city_key, filepath in targets.items():
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, encoding=Config.CSV_ENCODING, dtype=str)

        if "text" not in df.columns:
            raise KeyError(f"Column 'text' not found in {filepath.name}. Found: {list(df.columns)}")

        df = df[["text"]].dropna()
        df["text"] = df["text"].str.strip()
        df = df[df["text"] != ""]

        if sample_size is not None:
            df = df.head(sample_size)

        records = [
            {"source": city_key, "id": idx, "text": clean_report(row["text"])}
            for idx, row in df.iterrows()
            if clean_report(row["text"])  # Skip empty after cleaning
        ]
        print(f"{city_key}: {len(records)} reports loaded from {filepath.name}")
        all_records.extend(records)

    print(f"Total: {len(all_records)} reports")

    if save_processed and all_records:
        out_path = Config.PROCESSED_DIR / "cleaned_reports.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        print(f"Saved → {out_path}")

    return all_records


# Quick Test
if __name__ == "__main__":
    # Load data, change sample_size=None to load all rows
    reports = load_data(sample_size=None)
    for r in reports[:2]:
        print(f"\n[{r['source']}] ID={r['id']}\n{r['text'][:300]}")

    # # Load one city only
    # wa_only = load_data(cities=["WA"], sample_size=3, save_processed=False)

    # # Test cleaner directly
    # raw = '<s>Human: Accident on March 2.</s><s>Assistant: <ZERO>\n<\\\\s>'
    # print("\nCLEAN:", clean_report(raw))
