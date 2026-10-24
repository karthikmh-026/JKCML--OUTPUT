import psycopg2
conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()
cur.execute('SELECT * FROM "HANA_P3"."T001" WHERE "BUKRS" = \'1000\'')
for r in cur.fetchall():
    print(r)
conn.close()
