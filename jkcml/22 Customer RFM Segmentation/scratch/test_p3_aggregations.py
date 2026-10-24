import time
import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

print("Testing VBRK date formats in HANA_P3...")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN "FKDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN 1 END) as iso_dates,
        COUNT(CASE WHEN "FKDAT" ~ '^[0-9]{2}\.[0-9]{2}\.[0-9]{4}' THEN 1 END) as dot_dates,
        COUNT(CASE WHEN "FKDAT" IS NULL OR "FKDAT" = '' THEN 1 END) as null_dates
    FROM "HANA_P3"."VBRK"
""")
print("VBRK FKDAT patterns:", cur.fetchone())

print("Testing VBRK NETWR numeric casting...")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g') = '' THEN 1 END) as empty_netwr
    FROM "HANA_P3"."VBRK"
""")
print("VBRK NETWR empty:", cur.fetchone())

print("Testing aggregation query on HANA_P3.VBRK...")
t0 = time.time()
cur.execute("""
    SELECT 
        COUNT(DISTINCT "KUNAG") as cust_cnt,
        COUNT(*) as total_bills,
        SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) as total_netwr
    FROM "HANA_P3"."VBRK"
    WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
""")
res = cur.fetchone()
print(f"Aggregated {res[1]:,} bills across {res[0]:,} customers in {time.time() - t0:.2f}s, total_netwr={res[2]:,}")
