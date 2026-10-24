import psycopg2

conn = psycopg2.connect(
    host="192.168.1.44",
    port=5432,
    dbname="JKC_Input",
    user="postgres",
    password="postgres",
    connect_timeout=5
)
cur = conn.cursor()

# Check T077Y in HANA_P3
cur.execute('SELECT "KTOKK", count(*) FROM "HANA_P3"."T077Y" WHERE "SPRAS" = \'E\' GROUP BY "KTOKK" HAVING count(*) > 1 LIMIT 5')
dup_ktokk = cur.fetchall()
print("T077Y duplicate KTOKK for SPRAS='E':", dup_ktokk)

# Check T077Y languages
cur.execute('SELECT DISTINCT "SPRAS" FROM "HANA_P3"."T077Y"')
print("T077Y SPRAS distinct:", cur.fetchall())

# Check T001
cur.execute('SELECT "BUKRS", count(*) FROM "HANA_P3"."T001" GROUP BY "BUKRS" HAVING count(*) > 1 LIMIT 5')
print("T001 duplicate BUKRS:", cur.fetchall())

# Check LFA1 duplicate LIFNR in HANA_P3
cur.execute('SELECT "LIFNR", count(*) FROM "HANA_P3"."LFA1" GROUP BY "LIFNR" HAVING count(*) > 1 LIMIT 5')
print("LFA1 duplicate LIFNR:", cur.fetchall())

conn.close()
