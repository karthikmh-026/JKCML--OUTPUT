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
cur.execute("SELECT pid, state, query_start, now() - query_start AS duration, query FROM pg_stat_activity WHERE state != 'idle' AND query NOT LIKE '%pg_stat_activity%';")
rows = cur.fetchall()
print(f"Active queries count: {len(rows)}")
for r in rows:
    print(f"PID: {r[0]} | Duration: {r[3]} | Query: {r[4][:120]}...")
conn.close()
