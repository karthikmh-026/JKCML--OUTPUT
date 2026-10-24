import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

for tbl, id_col in [('VBAK', 'VBELN'), ('BSID', 'BELNR')]:
    cur.execute(f"""
        SELECT 
            (SELECT COUNT(*) FROM "HANA_P2"."{tbl}") as p2_cnt,
            (SELECT COUNT(*) FROM "HANA_P3"."{tbl}") as p3_cnt
    """)
    p2_cnt, p3_cnt = cur.fetchone()
    
    cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT "{id_col}" FROM "HANA_P2"."{tbl}"
            EXCEPT
            SELECT "{id_col}" FROM "HANA_P3"."{tbl}"
        ) t
    """)
    p2_only = cur.fetchone()[0]
    print(f"{tbl}: P2 rows={p2_cnt:,}, P3 rows={p3_cnt:,}, in P2 but NOT in P3={p2_only:,}")
