# src/config.py

DATA_PATH = "data/personal_finance_dataset.xlsx"

SHEET_NAME = "datathon_finance"

TARGET_COL = "PATTSITC"

# Feature columns (after rename)
FEATURES = [
    "PAGEMIEG",   # Age group
    "PPVRES",     # Province
    "PEDUCMIE",   # Education
    "PEFATINC",   # After-tax income
    "PFTENUR",    # Homeownership
    "PWDPRMOR",   # Mortgage debt
    "PWDSLOAN",   # Student loan debt
    "PWDSTCRD",   # Credit card debt
    "PWDSTLOC",   # Line of credit
    "PWASTDEP",   # Bank deposits
    "PWATFS"      # TFSA balance
]

CATEGORICAL_COLS = [
    "PAGEMIEG",
    "PPVRES",
    "PEDUCMIE",
    "PFTENUR"
]

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model selection: "logreg", "rf", "hgb"
DEFAULT_MODEL = "logreg"

# Class labels (adjust if your dataset uses different integers)
LABEL_MAP = {
    1: "Improved",
    2: "Worsened",
    3: "Stayed Same"
}
