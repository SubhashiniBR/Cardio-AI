"""Shared dataset and target definitions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
DATA_SHEET = "Sheet 1 - Table 1"

TARGETS = {
    "cad": {"column": "Cath", "positive_label": "CAD"},
    "lad": {"column": "LAD", "positive_label": "Stenotic"},
    "lcx": {"column": "LCX", "positive_label": "Stenotic"},
    "rca": {"column": "RCA", "positive_label": "Stenotic"},
}
LEAKAGE_COLUMNS = ("LAD", "LCX", "RCA", "Cath")
CONSTANT_COLUMNS_TO_EXCLUDE = ("Exertional CP",)
RANDOM_SEED = 42
