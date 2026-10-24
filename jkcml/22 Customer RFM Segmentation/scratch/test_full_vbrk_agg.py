import time
import psycopg2
import pandas as pd

t0 = time.time()
conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')

query = """
WITH vbrk_norm AS (
    SELECT
        "KUNAG" AS kunnr,
        "VKORG" AS vkorg,
        "VTWEG" AS vtweg,
        "SPART" AS spart,
        "BUKRS" AS bukrs,
        "KDGRP" AS kdgrp,
        "BZIRK" AS bzirk,
        "PLTYP" AS pltyp,
        "ZTERM" AS zterm,
        "WAERK" AS waerk,
        "VBELN" AS vbeln,
        "FKSTO" AS fksto,
        "FKART" AS fkart,
        CASE
            WHEN "FKDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST("FKDAT" AS DATE)
            WHEN "FKDAT" ~ '^[0-9]{2}\.[0-9]{2}\.[0-9]{4}' THEN TO_DATE("FKDAT", 'DD.MM.YYYY')
            ELSE NULL
        END AS fkdat_clean,
        COALESCE(CAST(NULLIF(REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS netwr_clean,
        COALESCE(CAST(NULLIF(REGEXP_REPLACE("MWSBK", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS mwsbk_clean
    FROM "HANA_P3"."VBRK"
    WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
),
vbrk_agg AS (
    SELECT
        kunnr,
        MAX(vkorg) AS vkorg,
        MAX(vtweg) AS vtweg,
        MAX(spart) AS spart,
        MAX(bukrs) AS bukrs,
        MAX(kdgrp) AS kdgrp,
        MAX(bzirk) AS bzirk,
        MAX(pltyp) AS pltyp,
        MAX(zterm) AS zterm,
        MAX(waerk) AS waerk,
        COUNT(DISTINCT vbeln) AS billing_doc_count,
        COUNT(DISTINCT CASE WHEN fksto = 'X' OR fkart IN ('RE', 'G2', 'S1', 'S2', 'ZRE') THEN vbeln END) AS cancelled_or_return_bills,
        SUM(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE 0.0 END) AS total_billed_net_value,
        SUM(CASE WHEN fksto IS NULL OR fksto = '' THEN mwsbk_clean ELSE 0.0 END) AS total_billed_tax_value,
        AVG(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE NULL END) AS avg_bill_value,
        MAX(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE 0.0 END) AS max_bill_value,
        STDDEV_POP(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE NULL END) AS std_bill_value,
        MIN(fkdat_clean) AS first_billing_date,
        MAX(fkdat_clean) AS last_billing_date,
        COUNT(DISTINCT fkdat_clean) AS distinct_billing_days,
        COUNT(DISTINCT SUBSTRING(CAST(fkdat_clean AS VARCHAR), 1, 7)) AS active_billing_months
    FROM vbrk_norm
    GROUP BY kunnr
)
SELECT COUNT(*) as cnt, MIN(first_billing_date), MAX(last_billing_date), SUM(total_billed_net_value)
FROM vbrk_agg;
"""

cur = conn.cursor()
cur.execute(query)
res = cur.fetchone()
print(f"Aggregated all VBRK in {time.time() - t0:.2f}s: {res[0]:,} customers, date range: {res[1]} to {res[2]}, total net: {res[3]:,}")
