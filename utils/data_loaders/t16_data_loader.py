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
xl = pd.read_excel(FILE, sheet_name="Table 16", header=None)

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

# Identify periods (column groups)
periods = []
period_col_map = {}
col = 1  # Start at column B (index 1)
while col < xl.shape[1]:
    period = str(xl.iloc[3, col]).strip()
    if period:
        periods.append(period)
        period_col_map[period] = col
    col += 4  # Skip to the next period group (B->F->J->N...)

# Identify blocks/sections and their row ranges
blocks = []
block_start = None
block_name = None
for row in range(5, xl.shape[0]):
    cell = str(xl.iloc[row, 0]).strip()
    if cell and not cell.startswith("Among") and not cell.startswith("How") and not cell.startswith("Vaccine") and not cell.startswith("Positive") and not cell.startswith("Vaccine hesitancy") and not cell.startswith("Weighted count") and not cell.startswith("Sample size"):
        if block_name:
            blocks.append((block_name, block_start, row-1))
        block_name = cell
        block_start = row+1
blocks.append((block_name, block_start, xl.shape[0]-1))

# Manually define block row ranges for Table 16 (based on visual inspection)
block_ranges = [
    ("Vaccine offers and uptake", 7, 17, 16, 17),
    ("How likely or unlikely are you to have a second dose of a coronavirus (COVID-19) vaccine?", 21, 32, 32, 33),
    ("Vaccine offers, uptake and sentiment (2 category - original definition)", 36, 39, 39, 40),
    ("Vaccine offers, uptake and sentiment (2 category - including attitudes to second dose)", 43, 46, 46, 47)
]

# For each period, block, and measure, extract %, LCL, UCL, weighted_count, sample_size
records = []
for period, col in period_col_map.items():
    for block, start_row, end_row, weighted_row, sample_row in block_ranges:
        weighted_count = safe_convert(xl.iloc[weighted_row, col])
        sample_size = safe_convert(xl.iloc[sample_row, col])
        for row in range(start_row, end_row):
            measure = str(xl.iloc[row, 0]).strip()
            if not measure or measure.lower().startswith("weighted count") or measure.lower().startswith("sample size"):
                continue
            for offset, value_type in enumerate(["%", "LCL", "UCL"]):
                if (col + offset) >= xl.shape[1]:
                    continue  # Skip if out of bounds
                value = safe_convert(xl.iloc[row, col+offset])
                if not np.isnan(value):
                    records.append({
                        "period": period,
                        "block": block,
                        "measure": measure,
                        "value_type": value_type,
                        "value": value,
                        "weighted_count": weighted_count,
                        "sample_size": sample_size
                    })

df = pd.DataFrame(records)
df.to_sql(
    "vaccine_hesitancy_trends",
    engine,
    if_exists="replace",
    index=False,
    schema="public"
)
print(f"✅ Loaded {len(df)} rows into vaccine_hesitancy_trends on Supabase")
