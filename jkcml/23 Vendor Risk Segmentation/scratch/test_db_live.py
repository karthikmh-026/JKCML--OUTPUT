import psycopg2
import sys

print("Attempting connection to 192.168.1.44:5432...")
try:
    conn = psycopg2.connect(
        host="192.168.1.44",
        port=5432,
        dbname="JKC_Input",
        user="postgres",
        password="postgres",
        connect_timeout=10
    )
    print("SUCCESS: Connected to PostgreSQL server!")
    cur = conn.cursor()
    
    # Check schemas
    cur.execute("SELECT schema_name FROM information_schema.schemata;")
    schemas = [r[0] for r in cur.fetchall()]
    print("Available schemas:", [s for s in schemas if 'HANA' in s.upper()])
    
    for s in ["HANA_P2", "HANA_P3"]:
        try:
            cur.execute(f'SELECT count(*) FROM "{s}"."LFA1"')
            lfa1_cnt = cur.fetchone()[0]
            print(f'{s}."LFA1" count: {lfa1_cnt:,}')
        except Exception as e:
            print(f'Error counting {s}."LFA1":', e)
            conn.rollback()

        try:
            cur.execute(f'SELECT count(DISTINCT "LIFNR") FROM "{s}"."LFA1"')
            distinct_cnt = cur.fetchone()[0]
            print(f'{s}."LFA1" DISTINCT LIFNR count: {distinct_cnt:,}')
        except Exception as e:
            print(f'Error counting {s}."LFA1" distinct:', e)
            conn.rollback()

        try:
            cur.execute(f'SELECT count(*) FROM "{s}"."LFB1"')
            lfb1_cnt = cur.fetchone()[0]
            print(f'{s}."LFB1" count: {lfb1_cnt:,}')
        except Exception as e:
            print(f'Error counting {s}."LFB1":', e)
            conn.rollback()

    # Check distinct LIFNR across both schemas combined
    try:
        cur.execute('''
            SELECT count(DISTINCT "LIFNR") FROM (
                SELECT "LIFNR" FROM "HANA_P2"."LFA1" WHERE "LIFNR" IS NOT NULL AND "LIFNR" != ''
                UNION
                SELECT "LIFNR" FROM "HANA_P3"."LFA1" WHERE "LIFNR" IS NOT NULL AND "LIFNR" != ''
            ) combined
        ''')
        union_cnt = cur.fetchone()[0]
        print(f"Total Unique Vendors (LIFNR) across HANA_P2 UNION HANA_P3: {union_cnt:,}")
    except Exception as e:
        print("Error counting union:", e)
        conn.rollback()

    conn.close()

except Exception as e:
    print("FAILED to connect to PostgreSQL:", e)
