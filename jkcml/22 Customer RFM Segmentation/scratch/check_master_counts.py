import time
import psycopg2
import pandas as pd

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

# Check row counts of joined query for active vs all
t0 = time.time()
cur.execute("""
    SELECT COUNT(*) FROM "HANA_P3"."KNA1"
""")
kna1_cnt = cur.fetchone()[0]
print(f"Total KNA1 in HANA_P3: {kna1_cnt:,}")

cur.execute("""
    SELECT COUNT(DISTINCT "KUNNR") FROM "HANA_P3"."KNVV"
""")
knvv_cnt = cur.fetchone()[0]
print(f"Total distinct KUNNR in KNVV: {knvv_cnt:,}")

cur.execute("""
    SELECT COUNT(DISTINCT "KUNNR") FROM "HANA_P3"."KNB1"
""")
knb1_cnt = cur.fetchone()[0]
print(f"Total distinct KUNNR in KNB1: {knb1_cnt:,}")
