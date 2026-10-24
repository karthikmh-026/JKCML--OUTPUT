import psycopg2
import time

print("Testing direct psycopg2 connection with connect_timeout=30...")
t0 = time.time()
conn = psycopg2.connect(
    host="192.168.1.44",
    port=5432,
    dbname="JKC_Input",
    user="postgres",
    password="postgres",
    connect_timeout=30
)
print(f"Connected in {time.time()-t0:.2f}s!")
conn.close()
