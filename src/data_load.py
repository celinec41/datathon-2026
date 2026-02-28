import pandas as pd
from .config import DATA_PATH

RENAME = {
    "Age Group": "PAGEMIEG",
    "Province of residence": "PPVRES",
    "Education Level": "PEDUCMIE",
    "After-Tax Income": "PEFATINC",
    "Home Ownership": "PFTENUR",
    "Mortgage Debt": "PWDPRMOR",
    "Student Loan Debt": "PWDSLOAN",
    "Credit Card Debt": "PWDSTCRD",
    "Line of Credit Debt": "PWDSTLOC",
    "Bank Deposits": "PWASTDEP",
    "TFSA Balance": "PWATFS",
    "COVID Financial Impact": "PATTSITC"
}

def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=RENAME)
    return df
