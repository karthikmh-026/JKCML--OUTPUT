import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

print("Testing VBAK ERDAT patterns in HANA_P3...")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN "ERDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN 1 END) as iso_dates,
        COUNT(CASE WHEN "ERDAT" ~ '^[0-9]{2}\.[0-9]{2}\.[0-9]{4}' THEN 1 END) as dot_dates,
        COUNT(CASE WHEN "ERDAT" IS NULL OR "ERDAT" = '' THEN 1 END) as null_dates
    FROM "HANA_P3"."VBAK"
""")
print("VBAK ERDAT patterns:", cur.fetchone())

print("Testing BSID BUDAT patterns in HANA_P3...")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN "BUDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN 1 END) as iso_dates,
        COUNT(CASE WHEN "BUDAT" ~ '^[0-9]{2}\.[0-9]{2}\.[0-9]{4}' THEN 1 END) as dot_dates,
        COUNT(CASE WHEN "BUDAT" IS NULL OR "BUDAT" = '' THEN 1 END) as null_dates
    FROM "HANA_P3"."BSID"
""")
print("BSID BUDAT patterns:", cur.fetchone())
