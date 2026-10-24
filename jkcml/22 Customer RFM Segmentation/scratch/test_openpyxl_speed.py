import time
import openpyxl
import pandas as pd
import numpy as np

print("Testing openpyxl write speed...")
n = 100000
df_test = pd.DataFrame({
    'col1': [f"CUST_{i}" for i in range(n)],
    'col2': [f"Customer Name {i}" for i in range(n)],
    'col3': np.random.rand(n) * 1000,
    'col4': np.random.randint(1, 100, n),
    'col5': ['' if i % 3 == 0 else f"City_{i}" for i in range(n)]
})

t0 = time.time()
wb = openpyxl.Workbook(write_only=True)
ws = wb.create_sheet(title="TestSheet")
ws.append(list(df_test.columns))
for row in df_test.itertuples(index=False):
    ws.append(list(row))
wb.save("scratch/test_openpyxl_100k.xlsx")
print(f"openpyxl write_only for {n:,} rows took: {time.time() - t0:.2f}s")
