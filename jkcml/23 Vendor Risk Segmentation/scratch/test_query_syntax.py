import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
from src.db_connector import load_config, get_db_engine, build_schema_extraction_query
import pandas as pd

config = load_config()
engine = get_db_engine(config)

print("Testing extraction query on HANA_P2 with limit=5 to test syntax and execution...")
t0 = time.time()
q = build_schema_extraction_query(schema="HANA_P2", limit=5)
df = pd.read_sql(q, engine)
print(f"HANA_P2 limit=5 succeeded in {time.time()-t0:.2f}s! Columns: {len(df.columns)}, rows: {len(df)}")

print("Testing extraction query on HANA_P3 with limit=5...")
t0 = time.time()
q3 = build_schema_extraction_query(schema="HANA_P3", limit=5)
try:
    df3 = pd.read_sql(q3, engine)
    print(f"HANA_P3 limit=5 succeeded in {time.time()-t0:.2f}s! Columns: {len(df3.columns)}, rows: {len(df3)}")
except Exception as e:
    print(f"HANA_P3 failed: {e}")
