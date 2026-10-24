import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

cur.execute("""
    SELECT COUNT(*) FROM (
        SELECT "VBELN" FROM "HANA_P2"."VBRK"
        INTERSECT
        SELECT "VBELN" FROM "HANA_P3"."VBRK"
    ) t
""")
overlap_vbeln = cur.fetchone()[0]
print(f"VBRK VBELN overlap between P2 and P3: {overlap_vbeln:,}")

cur.execute("""
    SELECT COUNT(*) FROM (
        SELECT "VBELN" FROM "HANA_P2"."VBRK"
        EXCEPT
        SELECT "VBELN" FROM "HANA_P3"."VBRK"
    ) t
""")
p2_only_vbeln = cur.fetchone()[0]
print(f"VBRK VBELN in P2 but not P3: {p2_only_vbeln:,}")

cur.execute("""
    SELECT COUNT(*) FROM (
        SELECT "VBELN" FROM "HANA_P3"."VBRK"
        EXCEPT
        SELECT "VBELN" FROM "HANA_P2"."VBRK"
    ) t
""")
p3_only_vbeln = cur.fetchone()[0]
print(f"VBRK VBELN in P3 but not P2: {p3_only_vbeln:,}")
