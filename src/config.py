DATA_PATH = "data/personal_finance_dataset.xlsx"
SHEET_NAME = "datathon_finance"

TARGET_COL = "COVID Financial Impact"

FEATURES = [
    "Age Group",
    "Province of residence",
    "Education Level",
    "After-Tax Income",
    "Home Ownership",
    "Mortgage Debt",
    "Student Loan Debt",
    "Credit Card Debt",
    "Line of Credit Debt",
    "Bank Deposits",
    "TFSA Balance",
    "Credit Card Payment",
]

CATEGORICAL_COLS = [
    "Age Group",
    "Province of residence",
    "Education Level",
    "Home Ownership",
    "Credit Card Payment"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_MODEL = "logreg"

LABEL_MAP = {
    1: "Improved",
    2: "Worsened",
    3: "Stayed same",
}
