import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

cur.execute("""
    SELECT COUNT(DISTINCT "KTOKD") FROM "HANA_P3"."T077X"
""")
print("Distinct KTOKD in T077X:", cur.fetchone()[0])

cur.execute("""
    SELECT COUNT(DISTINCT "BUKRS") FROM "HANA_P3"."T001"
""")
print("Distinct BUKRS in T001:", cur.fetchone()[0])
