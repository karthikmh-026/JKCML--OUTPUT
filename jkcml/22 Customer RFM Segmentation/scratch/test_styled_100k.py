import time
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import pandas as pd
import numpy as np

n = 101797
print(f"Testing openpyxl write speed for {n:,} rows...")

t0 = time.time()
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "ML_20_Enriched_Output(Predict)"

# Add metadata
ws.append(["Insight ID", "ML_20"])
ws.append(["Use Case", "Customer RFM Segmentation"])
ws.append(["Description", "Group customers by behavior and value"])
ws.append(["Generated On", "16-10-2026 16:50"])
ws.append(["Row Count", f"{n:,}"])
ws.append([]) # row 6 blank

headers = [f"Col_{i}" for i in range(46)]
ws.append(headers)

sample_row = ["1000", "J.K. CEMENT LIMITED", "Kanpur", "IN", "INR", "1000", "10", "10", "1001", "Customer Name", "Kanpur", "08", "208001", "0001", "Trading Goods", "01", "Dist 1", "Office 1", "01", "0001", "A", "Cement Retail", "10-01-2024", "15-04-2024", "20-06-2026", 45, 120, 150, 24, 2500000.0, 20833.33, 150000.0, 2, 5, 25000.0, 85.5, 90.0, 92.5, "555", 88.5, 11.5, 0, "Champions / High-Value Loyal", 0.052, 0, "Low Risk"]

for i in range(n):
    ws.append(sample_row)

save_path = "scratch/test_styled_100k.xlsx"
wb.save(save_path)
print(f"Workbook saved with {n:,} rows in {time.time() - t0:.2f}s!")
