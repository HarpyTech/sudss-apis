# ==========================================================
# FILE: fetch_ehr.py
# PURPOSE: Extract UID from current scan filename
#          and fetch patient's EHR history from EHR_History.csv
# ==========================================================
import os
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================
EHR_CSV_PATH = (
    r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\EHR_data\EHR_History.csv"
)

# ==========================================================
# FUNCTION: Extract UID from file name
# ==========================================================
def extract_uid_from_filename(filepath: str) -> str:
    """
    Example:
      10_IM-0002-1001.dcm.png  -> UID = '10'
    """
    base = os.path.basename(filepath)
    uid = base.split("_")[0]
    return uid

# ==========================================================
# FUNCTION: Fetch patient's EHR record by UID
# ==========================================================
def fetch_ehr_for_uid(uid: str) -> str:
    """
    Fetches patient's EHR history for given UID.
    Returns formatted string.
    If no record exists, returns message but allows rest of the pipeline to continue.
    """
    if not os.path.exists(EHR_CSV_PATH):
        return f"❌ EHR file not found: {EHR_CSV_PATH}"

    ehr_df = pd.read_csv(EHR_CSV_PATH)
    ehr_df["uid"] = ehr_df["uid"].astype(str)

    # Find all rows with the matching UID
    patient_records = ehr_df[ehr_df["uid"] == uid]

    # If no record found, return message (do not stop execution)
    if patient_records.empty:
        return f"⚠️ No previous EHR history available for patient UID {uid} to give Comparative status. Showing only current status."

    # If record found, format them neatly
    ehr_texts = []
    for _, row in patient_records.iterrows():
        ehr_texts.append(
            f"📁 Filename     : {row.get('filename', 'N/A')}\n"
            f"🩻 Projection   : {row.get('projection', 'N/A')}\n"
            f"🧠 Findings     : {row.get('findings', 'N/A')}\n"
            f"📝 Impression   : {row.get('impression', 'N/A')}\n"
            f"💬 Problems     : {row.get('Problems', 'N/A')}\n"
            f"📊 Comparison   : {row.get('comparison', 'N/A')}\n"
            f"--------------------------------------------------"
        )

    ehr_output = "\n".join(ehr_texts)
    return f"🧾 PATIENT HISTORY for UID {uid}\n{ehr_output}"
