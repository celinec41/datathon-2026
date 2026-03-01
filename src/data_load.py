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

def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    df.columns = df.columns.astype(str).str.strip()
    rename_existing = {k: v for k, v in RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_existing)

    # -----------------------------
    # Missing rules from codebook
    # -----------------------------

    # PEDUCMIE: 9 = Not stated
    if "Education Level" in df.columns:
        df["Education Level"] = df["Education Level"].replace(9, np.nan)

    # PFMTYPG: 9 = Not stated
    if "Family Type" in df.columns:
        df["Family Type"] = df["Family Type"].replace(9, np.nan)

    # PNBEARG: 9 = Not stated
    if "Number of Earners" in df.columns:
        df["Number of Earners"] = df["Number of Earners"].replace(9, np.nan)

    # PLFFPTME: 6 valid skip, 9 not stated -> missing
    if "Work Status 2022" in df.columns:
        df["Work Status 2022"] = df["Work Status 2022"].replace([6, 9], np.nan)

    # PATTCRU: 5 and 6 are missing
    if "Credit Card Payment" in df.columns:
        df["Credit Card Payment"] = df["Credit Card Payment"].replace([5, 6], np.nan)

    # -----------------------------
    # Structural rules
    # -----------------------------

    # Home value = 0 if renting (Home Ownership = 3)
    if ("Home Ownership" in df.columns) and ("Home Value" in df.columns):
        df.loc[df["Home Ownership"] == 3, "Home Value"] = 0

    # Mortgage debt = 0 if not own with mortgage (Home Ownership != 2)
    if ("Home Ownership" in df.columns) and ("Mortgage Debt" in df.columns):
        df.loc[df["Home Ownership"] != 2, "Mortgage Debt"] = 0

    return df
