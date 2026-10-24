import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

for tbl in ['VBRK', 'VBAK', 'BSID', 'KNA1', 'KNVV', 'KNB1']:
    print(f"\n=== Comparing {tbl} ===")
    for schema in ['HANA_P2', 'HANA_P3']:
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (schema, tbl))
        cols = dict(cur.fetchall())
        print(f"{schema}.{tbl}: {len(cols)} columns")
        # Check key columns
        key_cols = ['KUNNR', 'KUNAG', 'FKDAT', 'ERDAT', 'BUDAT', 'NETWR', 'MWSBK', 'DMBTR', 'PENDAYS']
        found = {k: cols.get(k) for k in key_cols if k in cols}
        print(f"  Key col types in {schema}: {found}")

for schema in ['HANA_P2', 'HANA_P3']:
    cur.execute(f'SELECT "FKDAT", "NETWR" FROM "{schema}"."VBRK" WHERE "FKDAT" IS NOT NULL LIMIT 5')
    print(f"\nSample VBRK in {schema}:", cur.fetchall())

for schema in ['HANA_P2', 'HANA_P3']:
    cur.execute(f'SELECT "ERDAT", "NETWR" FROM "{schema}"."VBAK" WHERE "ERDAT" IS NOT NULL LIMIT 5')
    print(f"\nSample VBAK in {schema}:", cur.fetchall())

for schema in ['HANA_P2', 'HANA_P3']:
    cur.execute(f'SELECT "BUDAT", "DMBTR", "PENDAYS" FROM "{schema}"."BSID" WHERE "BUDAT" IS NOT NULL LIMIT 5')
    print(f"\nSample BSID in {schema}:", cur.fetchall())
