import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

# Check duplicates in KNA1 for KUNNR
cur.execute("""
    SELECT "KUNNR", COUNT(*) 
    FROM "HANA_P3"."KNA1" 
    GROUP BY "KUNNR" 
    HAVING COUNT(*) > 1 
    LIMIT 5
""")
dup_kna1 = cur.fetchall()
print("KNA1 duplicate KUNNR:", dup_kna1)

# Check T077X
cur.execute("""
    SELECT "KTOKD", "SPRAS", COUNT(*) 
    FROM "HANA_P3"."T077X" 
    GROUP BY "KTOKD", "SPRAS" 
    HAVING COUNT(*) > 1 
    LIMIT 5
""")
dup_t077x = cur.fetchall()
print("T077X duplicates:", dup_t077x)

# Check T001
cur.execute("""
    SELECT "BUKRS", COUNT(*) 
    FROM "HANA_P3"."T001" 
    GROUP BY "BUKRS" 
    HAVING COUNT(*) > 1 
    LIMIT 5
""")
dup_t001 = cur.fetchall()
print("T001 duplicates:", dup_t001)
