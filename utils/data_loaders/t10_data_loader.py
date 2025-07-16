import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

FILE = "/Users/davelradindra/Downloads/referencetablevaccinehesitancy090821.xlsx"
SHEET = "Table 10"          # Region-level sheet

# ------------------------------------------------------------------
# 1. Read entire sheet with NO header interpretation
# ------------------------------------------------------------------
xl = pd.read_excel(FILE, sheet_name=SHEET, header=None)

# Row indexes based on your description (0-based)
ROW_REGION_NAMES   = 3   # Excel row 4
ROW_SUBHEADER      = 4   # Excel row 5
ROW_MEASURE_START  = 8   # Excel row 9
ROW_MEASURE_END    = 19  # Excel row 20  (inclusive)

# ------------------------------------------------------------------
# 2. Identify columns where sub-header is '%'  (= first of each triplet)
# ------------------------------------------------------------------
percent_cols = [c for c in xl.columns if xl.iloc[ROW_SUBHEADER, c] == '%']
regions      = {c: xl.iloc[ROW_REGION_NAMES, c] for c in percent_cols}

# ------------------------------------------------------------------
# 3. Extract the measure names and percent values
# ------------------------------------------------------------------
measures = xl.iloc[ROW_MEASURE_START:ROW_MEASURE_END+1, 0].reset_index(drop=True)
data = pd.DataFrame()

for col in percent_cols:
    region_name = regions[col]
    vals = xl.iloc[ROW_MEASURE_START:ROW_MEASURE_END+1, col].reset_index(drop=True)
    df_region = pd.DataFrame({
        "region": region_name,
        "measure": measures,
        "percent": vals
    })
    data = pd.concat([data, df_region], ignore_index=True)

# ------------------------------------------------------------------
# 4. Attach weighted_count and sample_size per region
# ------------------------------------------------------------------
row_weighted = xl.iloc[18, percent_cols]   # Excel row 19
row_sample   = xl.iloc[19, percent_cols]   # Excel row 20

data["weighted_count"] = data["region"].map(
    {regions[c]: row_weighted[c] for c in percent_cols}
).astype(float)

data["sample_size"] = data["region"].map(
    {regions[c]: row_sample[c] for c in percent_cols}
).astype(float)

data["percent"] = (
    data["percent"]
      .replace({"<1": 0.5, "..": np.nan})   # pick the numeric value you prefer
      .astype(float)
)

# ------------------------------------------------------------------
# 5. Load into Supabase Postgres via direct DSN
# ------------------------------------------------------------------
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
# Make sure special characters in the password are URL‑escaped
SUPABASE_DB_PASSWORD = quote_plus(SUPABASE_DB_PASSWORD) if SUPABASE_DB_PASSWORD else ""

# Using Supabase's Session Pooler connection string (IPv4 compatible)
dsn = f"postgresql+psycopg2://postgres.ylebxbxshnhtltukbjzx:{SUPABASE_DB_PASSWORD}@aws-0-us-east-2.pooler.supabase.com:5432/postgres?sslmode=require"
print(f"Connecting to database with DSN: {dsn.replace(SUPABASE_DB_PASSWORD, '********')}")

engine = create_engine(dsn)

data.to_sql("vaccine_hesitancy_region", engine, if_exists="replace", index=False, schema="public")
print(f"✅  Loaded {len(data)} rows into vaccine_hesitancy_region on Supabase")