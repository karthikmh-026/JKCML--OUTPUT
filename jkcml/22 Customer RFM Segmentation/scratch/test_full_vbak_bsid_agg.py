import time
import psycopg2

t0 = time.time()
conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

print("Testing full VBAK aggregation...")
cur.execute("""
    SELECT
        COUNT(DISTINCT "KUNNR") as cust_cnt,
        COUNT(DISTINCT "VBELN") as total_orders,
        SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) as total_order_val
    FROM "HANA_P3"."VBAK"
    WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
""")
res = cur.fetchone()
print(f"VBAK aggregated in {time.time() - t0:.2f}s: {res[0]:,} customers, {res[1]:,} orders, netwr: {res[2]:,}")

t1 = time.time()
print("Testing full BSID aggregation...")
cur.execute("""
    SELECT
        COUNT(DISTINCT "KUNNR") as cust_cnt,
        COUNT(DISTINCT "BELNR") as total_ar_items,
        SUM(
            CASE WHEN "SHKZG" = 'S'
            THEN COALESCE(CAST(NULLIF(REGEXP_REPLACE("DMBTR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)
            ELSE -COALESCE(CAST(NULLIF(REGEXP_REPLACE("DMBTR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)
            END
        ) as total_ar_bal
    FROM "HANA_P3"."BSID"
    WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
""")
res = cur.fetchone()
print(f"BSID aggregated in {time.time() - t1:.2f}s: {res[0]:,} customers, {res[1]:,} items, balance: {res[2]:,}")
