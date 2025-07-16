import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

FILE = "/Users/davelradindra/Downloads/referencetablevaccinehesitancy090821.xlsx"

SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
SUPABASE_DB_PASSWORD = quote_plus(SUPABASE_DB_PASSWORD) if SUPABASE_DB_PASSWORD else ""
dsn = f"postgresql+psycopg2://postgres.ylebxbxshnhtltukbjzx:{SUPABASE_DB_PASSWORD}@aws-0-us-east-2.pooler.supabase.com:5432/postgres?sslmode=require"
engine = create_engine(dsn)

# Read the sheet
xl = pd.read_excel(FILE, sheet_name="Table 18", header=None)

# Helper function
def safe_convert(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if val_str == ".." or val_str == "":
        return np.nan
    if val_str.startswith("<"):
        return 0.5
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# Parse period
period = str(xl.iloc[1, 0]).strip()  # Row 2, col A

# Weighted count and sample size (row 40, 41; col B)
weighted_count = safe_convert(xl.iloc[39, 1])
sample_size = safe_convert(xl.iloc[40, 1])

group = "All persons"

# Block header row indices (0-indexed)
block_header_indices = [8, 14, 18, 22, 30, 35]
block = None
records = []
for row in range(8, 38):  # Data rows are from 9 to 38 (0-indexed, inclusive start, exclusive end)
    cell = str(xl.iloc[row, 0]).strip()
    if not cell:
        continue
    # If this is a block header row, treat as both block and measure
    if row in block_header_indices:
        block = cell
        percent = safe_convert(xl.iloc[row, 1])
        lcl = safe_convert(xl.iloc[row, 2])
        ucl = safe_convert(xl.iloc[row, 3])
        records.append({
            "period": period,
            "group": group,
            "block": block,
            "measure": cell,
            "percent": percent,
            "lcl": lcl,
            "ucl": ucl,
            "weighted_count": weighted_count,
            "sample_size": sample_size
        })
        continue
    # Skip summary/stat rows
    if cell.lower().startswith("weighted count") or cell.lower().startswith("sample size"):
        continue
    # Data row under a block
    percent = safe_convert(xl.iloc[row, 1])
    lcl = safe_convert(xl.iloc[row, 2])
    ucl = safe_convert(xl.iloc[row, 3])
    records.append({
        "period": period,
        "group": group,
        "block": block,
        "measure": cell,
        "percent": percent,
        "lcl": lcl,
        "ucl": ucl,
        "weighted_count": weighted_count,
        "sample_size": sample_size
    })

# Save to DB
df = pd.DataFrame(records)
df.to_sql(
    "vaccine_hesitancy_reasons",
    engine,
    if_exists="replace",
    index=False,
    schema="public"
)
print(f"✅ Loaded {len(df)} rows into vaccine_hesitancy_reasons on Supabase")
