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
xl = pd.read_excel(FILE, sheet_name="Table 3", header=None)

# Constants for sheet layout (0-based)
HEADER_ROW = 3           # Excel row 4: Estimates (%)
GROUP_ROW = 4            # Excel row 5: Group names
SUBGROUP_ROW = 5         # Excel row 6: Men/Women
MEASURE_ROWS = list(range(9, 15)) + [16, 17]  # Excel rows 9-14, 16, 17
WEIGHTED_ROW = 19        # Excel row 20
SAMPLE_SIZE_ROW = 20     # Excel row 21

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

# Define columns for each group/subgroup
columns = [
    {"group": "All adults", "subgroup": None, "col": 1},
    {"group": "16 to 29", "subgroup": "Men", "col": 3},
    {"group": "16 to 29", "subgroup": "Women", "col": 4},
    {"group": "30 to 49", "subgroup": "Men", "col": 6},
    {"group": "30 to 49", "subgroup": "Women", "col": 7},
    {"group": "Aged 50 and over", "subgroup": "Men", "col": 9},
    {"group": "Aged 50 and over", "subgroup": "Women", "col": 10},
]

# Extract measure names
measures = [str(xl.iloc[r, 0]).strip() for r in MEASURE_ROWS]

# Build records
records = []
for col_info in columns:
    group = col_info["group"]
    subgroup = col_info["subgroup"]
    col = col_info["col"]
    weighted_count = safe_convert(xl.iloc[WEIGHTED_ROW, col])
    sample_size = safe_convert(xl.iloc[SAMPLE_SIZE_ROW, col])
    for i, measure in enumerate(measures):
        value = safe_convert(xl.iloc[MEASURE_ROWS[i], col])
        records.append({
            "wave_date": "2021-07-18",
            "group": group,
            "subgroup": subgroup,
            "measure": measure,
            "value": value,
            "weighted_count": weighted_count,
            "sample_size": sample_size
        })

df = pd.DataFrame(records)
df.to_sql(
    "vaccine_hesitancy_age_sex",
    engine,
    if_exists="replace",
    index=False,
    schema="public"
)
print(f"✅ Loaded {len(df)} rows into vaccine_hesitancy_age_sex on Supabase")