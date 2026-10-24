import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
from src.db_connector import extract_vendor_risk_data

print("Calling extract_vendor_risk_data(limit=None)...")
t0 = time.time()
df = extract_vendor_risk_data(limit=None)
print(f"Extraction completed in {time.time()-t0:.2f}s! Total vendors: {len(df):,}")
print("Sample schema distribution:", df['source_schema'].value_counts().to_dict())
