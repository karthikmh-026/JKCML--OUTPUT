import time
import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

print("Testing query count for all KNA1 customers with LEFT JOIN to VBRK/VBAK/BSID...")
t0 = time.time()
cur.execute("""
    SELECT COUNT(*) 
    FROM "HANA_P3"."KNA1" k
""")
print(f"Total KNA1: {cur.fetchone()[0]:,} in {time.time() - t0:.2f}s")
