

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
FILE = "/Users/davelradindra/Downloads/referencetablevaccinehesitancy090821.xlsx"

# ------------------------------------------------------------------
# Database Connection Configuration
# ------------------------------------------------------------------
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
# Make sure special characters in the password are URL‑escaped
SUPABASE_DB_PASSWORD = quote_plus(SUPABASE_DB_PASSWORD) if SUPABASE_DB_PASSWORD else ""

# Using Supabase's Session Pooler connection string (IPv4 compatible)
dsn = f"postgresql+psycopg2://postgres.ylebxbxshnhtltukbjzx:{SUPABASE_DB_PASSWORD}@aws-0-us-east-2.pooler.supabase.com:5432/postgres?sslmode=require"
engine = create_engine(dsn)

# Read the age-band sheet without headers
xl_age = pd.read_excel(FILE, sheet_name="Table 13", header=None)

# Constants for sheet layout (0-based)
HEADER_ROW = 3           # Excel row 4: Expense affordability  
SUBHEADER_ROW = 4        # Excel row 5: "%  LCL  UCL"
MEASURE_ROWS = list(range(8, 14)) + [15, 16]  # Rows with measures and sentiments
WEIGHTED_ROW = 18        # Excel row 19: weighted counts
SAMPLE_SIZE_ROW = 19     # Excel row 20: sample sizes

# Helper function to safely convert values
def safe_convert(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if val_str == ".." or val_str == "":
        return np.nan
    if val_str.startswith("<"):
        return 0.5  # You can adjust this value as needed
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# Identify columns where the subheader is '%' (first of each triplet)
percent_cols = [c for c in xl_age.columns if xl_age.iloc[SUBHEADER_ROW, c] == '%']
expense_affordability = {c: xl_age.iloc[HEADER_ROW, c] for c in percent_cols}

# Extract measure names
measures = [str(xl_age.iloc[r, 0]).strip() for r in MEASURE_ROWS]

# Extract weighted counts and sample sizes for each age band
weighted_counts = {c: safe_convert(xl_age.iloc[WEIGHTED_ROW, c]) for c in percent_cols}
sample_sizes = {c: safe_convert(xl_age.iloc[SAMPLE_SIZE_ROW, c]) for c in percent_cols}

# Build records
records = []
for col in percent_cols:
    ability = str(expense_affordability[col]).strip()
            
    # Extract percent, LCL, and UCL for each measure row
    pcts = [safe_convert(x) for x in xl_age.iloc[MEASURE_ROWS, col]]
    lcls = [safe_convert(x) for x in xl_age.iloc[MEASURE_ROWS, col+1]]
    ucls = [safe_convert(x) for x in xl_age.iloc[MEASURE_ROWS, col+2]]
    for measure, pct, lcl, ucl in zip(measures, pcts, lcls, ucls):
        records.append({
            "wave_date": "2021-07-18",
            "expense_affordability": ability,
            "measure": measure,
            "percent": pct,
            "lcl": lcl,
            "ucl": ucl,
            "weighted_count": weighted_counts[col],
            "sample_size": sample_sizes[col]
        })

# Convert to DataFrame and load into Supabase
df = pd.DataFrame(records)
df.to_sql(
    "vaccine_hesitancy_expense_affordability",
    engine,
    if_exists="replace",
    index=False,
    schema="public"
)
print(f"✅ Loaded {len(df)} rows into vaccine_hesitancy_expense_affordability on Supabase")