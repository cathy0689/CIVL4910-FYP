import re
import pandas as pd
from pathlib import Path
from config import Config


# Constants
# Maps each CSV filename to the label column it will generate after merging.
CSV_FILES = {
    "inj.csv": "label_injury",
    "sev.csv": "label_severity",
    "type.csv": "label_type",
}

# The assistant tag pattern to strip from raw text.
# Matches: <ZERO>, <ONE>, <TWO>, ... and the surrounding <s>...</s> wrappers.
_TAG_PATTERN = re.compile(
    r"<s>Human:\s*"       # opening Human tag
    r"|</s>"              # closing tag
    r"|<s>Assistant:\s*"  # opening Assistant tag
    r"|<[A-Z]+>"          # label tags like <ZERO>, <ONE>, <TWO>
    r"|<\\\\s>",          # escaped closing tag variant seen in your sample
    re.IGNORECASE,
)


# Raw Loader
def _load_single_csv(filepath: Path) -> pd.DataFrame:
    """
    Read one CSV and return a DataFrame with two columns:
      - 'text'  : the raw accident report string
      - 'label' : the assistant answer extracted from the tail of the text
                  (e.g. "ZERO", "ONE", "TWO" → mapped to 0, 1, 2)

    Why:  Your CSV stores both the report AND the answer in one cell,
          separated by the <s>Assistant: <LABEL> pattern. We split them
          here so the label is not fed into the NLP/LLM as input text.
    """
    df = pd.read_csv(filepath, header=0, encoding=Config.CSV_ENCODING)
    df = df.rename(columns={df.columns[0]: "raw_text"})  # normalise column name

    # Extract the assistant label from the tail of each cell
    df["label"] = df["raw_text"].str.extract(
        r"<s>Assistant:\s*<([^>]+)>", flags=re.IGNORECASE
    )[0].str.strip()

    # Strip all special tokens to leave only the human-readable report
    df["text"] = df["raw_text"].apply(_clean_text)

    return df[["text", "label"]]


# Text Cleaner
def _clean_text(raw: str) -> str:
    """
    Remove template tokens and normalise whitespace.

    Why:  The raw cell contains chatbot formatting tags that carry no
          accident-related information. Feeding them to spaCy/BERT or the
          LLM wastes tokens and can confuse entity recognition.

    Steps:
      1. Strip the special tokens with the compiled regex.
      2. Collapse multiple spaces / newlines into a single space.
      3. Strip leading / trailing whitespace.
    """
    if not isinstance(raw, str):
        return ""
    text = _TAG_PATTERN.sub(" ", raw)           # remove tokens
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


# Label Normaliser
_LABEL_MAP = {
    "ZERO":  0,
    "ONE":   1,
    "TWO":   2,
    "THREE": 3,
    "FOUR":  4,
}

def _normalise_label(series: pd.Series) -> pd.Series:
    """
    Convert word labels (ZERO, ONE, …) to integers.

    Why:  Numeric labels are required by evaluator.py for F1 / accuracy
          calculations with scikit-learn. Keeping them as strings would
          require extra conversion in every downstream step.
    """
    return series.str.upper().map(_LABEL_MAP)


# Merger
def _merge_csvs() -> pd.DataFrame:
    """
    Load all three CSVs and merge on 'text' (inner join).

    Why inner join:  We only keep cases where ALL THREE labels exist.
                     Partial rows cannot be used as a complete Gold Standard
                     for multi-label evaluation.

    Result columns:
      case_id | text | label_injury | label_severity | label_type
    """
    dfs = {}
    for filename, label_col in CSV_FILES.items():
        path = Config.RAW_CSV_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = _load_single_csv(path)
        df = df.rename(columns={"label": label_col})
        dfs[label_col] = df

    # Merge all three on the cleaned text (same reports, same order assumed)
    merged = dfs["label_injury"]
    for label_col in ["label_severity", "label_type"]:
        merged = merged.merge(dfs[label_col][["text", label_col]],
                              on="text", how="inner")

    # Normalise all label columns to integers
    for label_col in CSV_FILES.values():
        merged[label_col] = _normalise_label(merged[label_col])

    # Drop rows where ANY label failed to parse
    merged = merged.dropna(subset=list(CSV_FILES.values()))
    for label_col in CSV_FILES.values():
        merged[label_col] = merged[label_col].astype(int)

    # Assign a stable case ID for traceability across pipelines
    merged = merged.reset_index(drop=True)
    merged.insert(0, "case_id", ["case_" + str(i).zfill(4)
                                  for i in range(len(merged))])

    return merged


# Public API
def load_reports(sample: bool = True) -> pd.DataFrame:
    """
    Main entry point called by nlp_baseline.py, llm_pipeline.py, and main.py.

    Args:
        sample: If True, return only Config.SAMPLE_SIZE rows (for fast
                experiments). Set to False to process all reports.

    Returns:
        DataFrame with columns:
          case_id | text | label_injury | label_severity | label_type

    Why save to processed/:  Merging + cleaning takes time. Caching the
        result means subsequent runs skip the merge entirely, saving
        time during iterative development and evaluation.
    """
    Config.ensure_dirs()
    cache_path = Config.PROCESSED_DIR / "merged_reports.csv"

    # Use cached result if available
    if cache_path.exists():
        print(f"Loading cached merged dataset from {cache_path}")
        df = pd.read_csv(cache_path, encoding=Config.CSV_ENCODING)
    else:
        print("Merging and cleaning raw CSVs...")
        df = _merge_csvs()
        df.to_csv(cache_path, index=False, encoding=Config.CSV_ENCODING)
        print(f"Saved merged dataset → {cache_path}  ({len(df)} rows)")

    if sample:
        df = df.head(Config.SAMPLE_SIZE)
        print(f"Using sample of {len(df)} reports for experiment.")

    return df


def get_texts(sample: bool = True) -> list[str]:
    """
    Convenience function: returns ONLY the cleaned text strings.
    Used by pipelines that don't need the label columns.

    Returns:
        List of cleaned accident report strings.
    """
    return load_reports(sample=sample)["text"].tolist()


# Quick Test
if __name__ == "__main__":
    df = load_reports(sample=True)
    print(f"\n Dataset shape : {df.shape}")
    print(f"Columns : {list(df.columns)}")
    print(f"\n First report preview :\n{df['text'].iloc[0][:300]}...")
    print(f"\n Labels sample :\n{df[list(CSV_FILES.values())].head()}")
