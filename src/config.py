from pathlib import Path

# Project root is one level above src/
ROOT = Path(__file__).resolve().parents[1]

RAW_XLSX = ROOT / "data" / "raw" / "dataset.xlsx"

# Live Google Sheet (public: anyone with link)
SHEET_ID = "1cDQ004uMDmlTk6wgvVaD9yHa5a7GxMMW13W6i1GNP8o"
SHEET_GID = "1834307442"  # Form Responses 1 tab er gid

SHEET_CSV_URL_1 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"
SHEET_CSV_URL_2 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SHEET_GID}"


OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
MET_DIR = OUT_DIR / "metrics"

CLEAN_CSV = OUT_DIR / "cleaned.csv"
DISCARDED_CSV = OUT_DIR / "discarded_rows.csv"

METRICS_JSON = MET_DIR / "metrics.json"
ABLATION_CSV = MET_DIR / "ablation_results.csv"

MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"

RANDOM_SEED = 42

# Consent values for your dataset
CONSENT_YES = {"Yes, I agree", "Yes", "I agree", "Agree"}

# Employed definition for your Q17
EMPLOYED_SET = {
    "Employed full-time",
    "Employed part-time",
    "Intern",
    "Freelancing income (regular)",
    "Freelancing income (irregular)",
    "Self-employed / business",
}

# Cleaning thresholds
MAX_MISSING_RATIO = 0.30
