import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

cur.execute("""
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
""")
schemas = [r[0] for r in cur.fetchall()]
print('Available Schemas:', schemas)

for schema in schemas:
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s", (schema,))
    tables = [r[0] for r in cur.fetchall()]
    relevant = [t for t in tables if any(k in t.upper() for k in ['KNA1', 'KNVV', 'KNB1', 'VBRK', 'VBRP', 'VBAK', 'VBAP', 'BSID', 'BSAD'])]
    print(f"\nSchema {schema}: total tables = {len(tables)}, relevant customer/sales tables = {relevant}")

for schema in ['HANA_P2', 'HANA_P3']:
    if schema in schemas:
        print(f"\n--- Row Counts in {schema} ---")
        for tbl in ['KNA1', 'KNVV', 'KNB1', 'VBRK', 'VBAK', 'BSID', 'T077X', 'T001']:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{tbl}"')
                cnt = cur.fetchone()[0]
                print(f'{schema}.{tbl}: {cnt:,} rows')
            except Exception as e:
                conn.rollback()
                print(f'{schema}.{tbl}: NOT FOUND ({e})')
