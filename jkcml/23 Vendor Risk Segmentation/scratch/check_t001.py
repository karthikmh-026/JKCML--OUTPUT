import psycopg2
conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()
for s in ['HANA_P2', 'HANA_P3']:
    cur.execute(f'SELECT count(*) FROM "{s}"."T001" WHERE "BUKRS" = \'1000\'')
    print(f'{s}.T001 BUKRS=1000 count:', cur.fetchone()[0])
conn.close()
