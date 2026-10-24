import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

# Check total unique customers in KNA1 across P2 and P3
cur.execute("""
    SELECT COUNT(DISTINCT "KUNNR") FROM (
        SELECT "KUNNR" FROM "HANA_P2"."KNA1"
        UNION
        SELECT "KUNNR" FROM "HANA_P3"."KNA1"
    ) t
""")
total_kna1 = cur.fetchone()[0]
print(f"Total unique customers across KNA1 (P2 + P3): {total_kna1:,}")

# Check total unique customers who have commercial billing activity (VBRK)
cur.execute("""
    SELECT COUNT(DISTINCT kunnr) FROM (
        SELECT "KUNAG" as kunnr FROM "HANA_P2"."VBRK" WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
        UNION
        SELECT "KUNAG" as kunnr FROM "HANA_P3"."VBRK" WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
    ) t
""")
total_vbrk_cust = cur.fetchone()[0]
print(f"Total unique customers with VBRK billing (P2 + P3): {total_vbrk_cust:,}")

# Check total unique customers with VBAK orders
cur.execute("""
    SELECT COUNT(DISTINCT kunnr) FROM (
        SELECT "KUNNR" as kunnr FROM "HANA_P2"."VBAK" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
        UNION
        SELECT "KUNNR" as kunnr FROM "HANA_P3"."VBAK" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
    ) t
""")
total_vbak_cust = cur.fetchone()[0]
print(f"Total unique customers with VBAK orders (P2 + P3): {total_vbak_cust:,}")

# Check total active commercial customers (VBRK OR VBAK OR BSID)
cur.execute("""
    SELECT COUNT(DISTINCT kunnr) FROM (
        SELECT "KUNAG" as kunnr FROM "HANA_P2"."VBRK" WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
        UNION
        SELECT "KUNAG" as kunnr FROM "HANA_P3"."VBRK" WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
        UNION
        SELECT "KUNNR" as kunnr FROM "HANA_P2"."VBAK" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
        UNION
        SELECT "KUNNR" as kunnr FROM "HANA_P3"."VBAK" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
        UNION
        SELECT "KUNNR" as kunnr FROM "HANA_P2"."BSID" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
        UNION
        SELECT "KUNNR" as kunnr FROM "HANA_P3"."BSID" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
    ) t
""")
total_active_cust = cur.fetchone()[0]
print(f"Total unique active commercial customers (VBRK | VBAK | BSID in P2 + P3): {total_active_cust:,}")
