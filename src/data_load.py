import pandas as pd
import numpy as np
from .config import DATA_PATH, SHEET_NAME

RENAME = {
    "PAGEMIEG": "Age Group",
    "PWDSTCRD": "Credit Card Debt",
    "PATTSITC": "COVID Financial Impact",
    "PATTSKP": "Skipped Payments",
    "PEDUCMIE": "Education Level",
    "PEFATINC": "After-Tax Income",
    "PFMTYPG": "Family Type",
    "PFTENUR": "Home Ownership",
    "PLFFPTME": "Work Status 2022",
    "PNBEARG": "Number of Earners",
    "PPVRES": "Province of residence",
    "PWAPRVAL": "Home Value",
    "PWASTDEP": "Bank Deposits",
    "PWATFS": "TFSA Balance",
    "PWDPRMOR": "Mortgage Debt",
    "PWDSLOAN": "Student Loan Debt",
    "PATTCRU": "Credit Card Payment",
    "PWDSTLOC": "Line of Credit Debt",
    "PWNETWPG": "Net Worth",
}

INV_RENAME = {v: k for k, v in RENAME.items()}

def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    df.columns = df.columns.astype(str).str.strip()
    rename_existing = {k: v for k, v in RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_existing)

    # True missing: 9 = Not stated
    if "Education Level" in df.columns:
        df["Education Level"] = df["Education Level"].replace(9, np.nan)

    if "Number of Earners" in df.columns:
        df["Number of Earners"] = df["Number of Earners"].replace(9, np.nan)

    # Work status: 6 valid skip, 9 not stated -> treat as missing
    if "Work Status 2022" in df.columns:
        df["Work Status 2022"] = df["Work Status 2022"].replace([6, 9], np.nan)

    # Credit card payment:
    # - 6 valid skip = no credit card (structural)
    # - 9 not stated (if present) = missing
    if "Credit Card Payment" in df.columns:
        df["Credit Card Payment"] = df["Credit Card Payment"].replace(9, np.nan)

        df["has_credit_card"] = np.nan
        known = df["Credit Card Payment"].notna()
        df.loc[known, "has_credit_card"] = (df.loc[known, "Credit Card Payment"] != 6).astype(int)

        df.loc[df["Credit Card Payment"] == 6, "Credit Card Payment"] = np.nan

    # Structural: home value = 0 if renting (Home Ownership = 3)
    if ("Home Ownership" in df.columns) and ("Home Value" in df.columns):
        df.loc[df["Home Ownership"] == 3, "Home Value"] = 0

    # Structural: mortgage debt = 0 if not homeowner (Home Ownership != 2)
    if ("Home Ownership" in df.columns) and ("Mortgage Debt" in df.columns):
        df.loc[df["Home Ownership"] != 2, "Mortgage Debt"] = 0

    return df