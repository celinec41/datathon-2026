# src/config.py

DATA_PATH = "data/personal_finance_dataset.xlsx"
SHEET_NAME = "datathon_finance"

TARGET_COL = "COVID Financial Impact"   # this corresponds to PATTSITC

FEATURES = [
    "Age Group",                 # PAGEMIEG
    "Province of residence",     # PPVRES
    "Education Level",           # PEDUCMIE
    "After-Tax Income",          # PEFATINC
    "Home Ownership",            # PFTENUR
    "Mortgage Debt",             # PWDPRMOR
    "Student Loan Debt",         # PWDSLOAN
    "Credit Card Debt",          # PWDSTCRD
    "Line of Credit Debt",       # PWDSTLOC
    "Bank Deposits",             # PWASTDEP
    "TFSA Balance",              # PWATFS
]

CATEGORICAL_COLS = [
    "Age Group",
    "Province of residence",
    "Education Level",
    "Home Ownership",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_MODEL = "logreg"

LABEL_MAP = {
    1: "Improved",
    2: "Worsened",
    3: "Stayed Same",
}
