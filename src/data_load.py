import pandas as pd
import numpy as np
from .config import DATA_PATH, SHEET_NAME

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
    "COVID Financial Impact": "PATTSITC",

    "Number of Earners": "PNBEARG",
    "Work Status 2022": "PLFFPTME",
    "Credit Card Payment": "PATTCRU",
    "Home Value": "PWAPRVAL",
}

def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=RENAME)

    # True missing: 9 = Not stated
    if "PEDUCMIE" in df.columns:
        df["PEDUCMIE"] = df["PEDUCMIE"].replace(9, np.nan)

    if "PNBEARG" in df.columns:
        df["PNBEARG"] = df["PNBEARG"].replace(9, np.nan)

    # Work status: 6 valid skip, 9 not stated -> treat as missing (handle later)
    if "PLFFPTME" in df.columns:
        df["PLFFPTME"] = df["PLFFPTME"].replace([6, 9], np.nan)

    # Credit card payment: 6 valid skip = no credit card (structural)
    if "PATTCRU" in df.columns:
        df["has_credit_card"] = (df["PATTCRU"] != 6).astype(int)
        df.loc[df["PATTCRU"] == 6, "PATTCRU"] = np.nan

    # Structural: home value = 0 if renting (PFTENUR = 3)
    if "PFTENUR" in df.columns and "PWAPRVAL" in df.columns:
        df.loc[df["PFTENUR"] == 3, "PWAPRVAL"] = 0

    # Structural: mortgage debt = 0 if not homeowner (PFTENUR != 2)
    if "PFTENUR" in df.columns and "PWDPRMOR" in df.columns:
        df.loc[df["PFTENUR"] != 2, "PWDPRMOR"] = 0

    return df
