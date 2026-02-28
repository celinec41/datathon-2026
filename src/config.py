# src/config.py

DATA_PATH = "data/personal_finance_dataset.xlsx"

TARGET_COL = "PATTSITC"

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
