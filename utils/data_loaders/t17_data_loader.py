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
xl = pd.read_excel(FILE, sheet_name="Table 17", header=None)

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

# Define blocks: (block_name, start_row, end_row, weighted_row, sample_row)
block_ranges = [
    ("Among those who have received at least one dose of a vaccine", 8, 17, 18, 19),
    ("Among those who have not yet received a vaccine", 23, 34, 33, 34)
]

# Define groups and their starting columns
# (group_name, start_col)
groups = [
    ("All persons", 1),
    ("Aged 50 and over", 5),
    ("Aged 30 to 49", 9),
    ("Aged 16 to 29", 13),
    ("Men", 17),
    ("Women", 21),
    ("Disabled", 25),
    ("Non-disabled", 29),
    ("Don't know/Prefer not to say", 33)
]

# For each block, group, and measure, extract %, LCL, UCL, weighted_count, sample_size
records = []
for block, start_row, end_row, weighted_row, sample_row in block_ranges:
    for group, col in groups:
        weighted_count = safe_convert(xl.iloc[weighted_row, col])
        sample_size = safe_convert(xl.iloc[sample_row, col])
        for row in range(start_row, end_row+1):
            measure = str(xl.iloc[row, 0]).strip()
            if not measure or measure.lower().startswith("weighted count") or measure.lower().startswith("sample size"):
                continue
            for offset, value_type in enumerate(["%", "LCL", "UCL"]):
                if (col + offset) >= xl.shape[1]:
                    continue  # Skip if out of bounds
                value = safe_convert(xl.iloc[row, col+offset])
                if not np.isnan(value):
                    records.append({
                        "block": block,
                        "group": group,
                        "measure": measure,
                        "value_type": value_type,
                        "value": value,
                        "weighted_count": weighted_count,
                        "sample_size": sample_size
                    })

df = pd.DataFrame(records)
df.to_sql(
    "vaccine_hesitancy_barriers",
    engine,
    if_exists="replace",
    index=False,
    schema="public"
)
print(f"✅ Loaded {len(df)} rows into vaccine_hesitancy_barriers on Supabase")
