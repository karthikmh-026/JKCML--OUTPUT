import psycopg2
import pandas as pd

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

print("--- Date ranges and Customer counts in VBRK ---")
for s in ['HANA_P2', 'HANA_P3']:
    cur.execute(f"""
        SELECT 
            MIN("FKDAT") as min_dt, 
            MAX("FKDAT") as max_dt, 
            COUNT(*) as total_rows, 
            COUNT(DISTINCT "KUNAG") as distinct_customers
        FROM "{s}"."VBRK"
        WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
    """)
    res = cur.fetchone()
    print(f"{s}.VBRK: min_date={res[0]}, max_date={res[1]}, total_rows={res[2]:,}, distinct_customers={res[3]:,}")

print("\n--- Overlap of KNA1 customers ---")
cur.execute("""
    SELECT 
        (SELECT COUNT(DISTINCT "KUNNR") FROM "HANA_P2"."KNA1") as p2_cust,
        (SELECT COUNT(DISTINCT "KUNNR") FROM "HANA_P3"."KNA1") as p3_cust
""")
print("KNA1 distinct kunnr:", cur.fetchone())

cur.execute("""
    SELECT COUNT(*) FROM (
        SELECT "KUNNR" FROM "HANA_P2"."KNA1"
        INTERSECT
        SELECT "KUNNR" FROM "HANA_P3"."KNA1"
    ) t
""")
print("KNA1 overlap count:", cur.fetchone()[0])

print("\n--- Overlap of VBRK customers ---")
cur.execute("""
    SELECT COUNT(*) FROM (
        SELECT DISTINCT "KUNAG" FROM "HANA_P2"."VBRK" WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
        INTERSECT
        SELECT DISTINCT "KUNAG" FROM "HANA_P3"."VBRK" WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
    ) t
""")
print("VBRK overlap count:", cur.fetchone()[0])
