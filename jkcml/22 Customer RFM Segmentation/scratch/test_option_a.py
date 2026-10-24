import time
import psycopg2
import pandas as pd

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

# Test joined query for active commercial customers (101,797)
t0 = time.time()
cur.execute("""
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
            WHEN "FKDAT" ~ '^[0-9]{2}\\.[0-9]{2}\\.[0-9]{4}' THEN TO_DATE("FKDAT", 'DD.MM.YYYY')
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
),
vbak_norm AS (
    SELECT
        "KUNNR" AS kunnr,
        "VBELN" AS vbeln,
        "LIFSK" AS lifsk,
        "FAKSK" AS faksk,
        "CMGST" AS cmgst,
        CASE
            WHEN "ERDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST("ERDAT" AS DATE)
            WHEN "ERDAT" ~ '^[0-9]{2}\\.[0-9]{2}\\.[0-9]{4}' THEN TO_DATE("ERDAT", 'DD.MM.YYYY')
            ELSE NULL
        END AS erdat_clean,
        COALESCE(CAST(NULLIF(REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS netwr_clean
    FROM "HANA_P3"."VBAK"
    WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
),
vbak_agg AS (
    SELECT
        kunnr,
        COUNT(DISTINCT vbeln) AS sales_order_count,
        SUM(netwr_clean) AS total_order_net_value,
        AVG(netwr_clean) AS avg_order_value,
        MIN(erdat_clean) AS first_order_date,
        MAX(erdat_clean) AS last_order_date,
        COUNT(DISTINCT CASE WHEN (lifsk IS NOT NULL AND lifsk != '') OR (faksk IS NOT NULL AND faksk != '') OR cmgst IN ('B', 'C') THEN vbeln END) AS blocked_order_count
    FROM vbak_norm
    GROUP BY kunnr
),
bsid_norm AS (
    SELECT
        "KUNNR" AS kunnr,
        "BELNR" AS belnr,
        "SHKZG" AS shkzg,
        CASE
            WHEN "BUDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST("BUDAT" AS DATE)
            WHEN "BUDAT" ~ '^[0-9]{2}\\.[0-9]{2}\\.[0-9]{4}' THEN TO_DATE("BUDAT", 'DD.MM.YYYY')
            ELSE NULL
        END AS budat_clean,
        COALESCE(CAST(NULLIF(REGEXP_REPLACE("PENDAYS", '[^0-9-]', '', 'g'), '') AS INTEGER), 0) AS pendays_clean,
        COALESCE(CAST(NULLIF(REGEXP_REPLACE("DMBTR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS dmbtr_clean
    FROM "HANA_P3"."BSID"
    WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
),
bsid_agg AS (
    SELECT
        kunnr,
        COUNT(DISTINCT belnr) AS open_ar_item_count,
        SUM(CASE WHEN shkzg = 'S' THEN dmbtr_clean ELSE -dmbtr_clean END) AS open_ar_balance,
        MAX(budat_clean) AS last_ar_posting_date,
        AVG(pendays_clean) AS avg_ar_pending_days
    FROM bsid_norm
    GROUP BY kunnr
),
active_customers AS (
    SELECT kunnr FROM vbrk_agg
    UNION
    SELECT kunnr FROM vbak_agg
    UNION
    SELECT kunnr FROM bsid_agg
),
knvv_primary AS (
    SELECT DISTINCT ON (kv."KUNNR")
        kv."KUNNR", kv."VKORG", kv."VTWEG", kv."SPART", kv."KDGRP", kv."BZIRK",
        kv."PLTYP", kv."KONDA", kv."KLABC", kv."ZTERM" AS knvv_zterm,
        kv."VKBUR", kv."VKGRP", kv."WAERS" AS knvv_waers,
        kv."ZCUSTOMER_GROUP", kv."ZCUSTOMER_GRP_DES", kv."ZRURAL_URBAN",
        kv."ZMETRO_NONMETRO", kv."ZDISTRIBUTOR_TYP", kv."ZTIER"
    FROM "HANA_P3"."KNVV" kv
    ORDER BY kv."KUNNR", kv."VKORG" ASC
),
knb1_primary AS (
    SELECT DISTINCT ON (kb."KUNNR")
        kb."KUNNR", kb."BUKRS", kb."AKONT", kb."ZTERM" AS knb1_zterm,
        kb."ZWELS", kb."FDGRV", kb."SPERR" AS knb1_sperr, kb."LOEVM" AS knb1_loevm
    FROM "HANA_P3"."KNB1" kb
    ORDER BY kb."KUNNR", kb."BUKRS" ASC
)
SELECT COUNT(*)
FROM active_customers ac
JOIN "HANA_P3"."KNA1" k ON ac.kunnr = k."KUNNR"
LEFT JOIN vbrk_agg vb ON ac.kunnr = vb.kunnr
LEFT JOIN vbak_agg vo ON ac.kunnr = vo.kunnr
LEFT JOIN bsid_agg ar ON ac.kunnr = ar.kunnr
LEFT JOIN knvv_primary kv ON ac.kunnr = kv."KUNNR"
LEFT JOIN knb1_primary kb ON ac.kunnr = kb."KUNNR"
""")
res = cur.fetchone()[0]
print(f"Option A (Active commercial customers matched with KNA1): {res:,} rows in {time.time() - t0:.2f}s")
