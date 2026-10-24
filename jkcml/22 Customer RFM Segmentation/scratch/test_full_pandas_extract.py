import time
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@192.168.1.44:5432/JKC_Input")

t0 = time.time()
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
    JOIN active_customers ac ON kv."KUNNR" = ac.kunnr
    ORDER BY kv."KUNNR", kv."VKORG" ASC
),
knb1_primary AS (
    SELECT DISTINCT ON (kb."KUNNR")
        kb."KUNNR", kb."BUKRS", kb."AKONT", kb."ZTERM" AS knb1_zterm,
        kb."ZWELS", kb."FDGRV", kb."SPERR" AS knb1_sperr, kb."LOEVM" AS knb1_loevm
    FROM "HANA_P3"."KNB1" kb
    JOIN active_customers ac ON kb."KUNNR" = ac.kunnr
    ORDER BY kb."KUNNR", kb."BUKRS" ASC
)
SELECT
    COALESCE(vb.bukrs, kb."BUKRS", '1000') AS company_code,
    COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
    COALESCE(comp."ORT01", 'Kanpur') AS company_city,
    COALESCE(comp."LAND1", 'IN') AS company_country,
    COALESCE(vb.waerk, kv.knvv_waers, comp."WAERS", 'INR') AS currency,
    ac.kunnr AS customer_id,
    COALESCE(k."NAME1", '') AS customer_name,
    COALESCE(k."ORT01", '') AS customer_city,
    COALESCE(k."PSTLZ", '') AS postal_code,
    COALESCE(k."REGIO", '') AS customer_region,
    COALESCE(k."LAND1", 'IN') AS customer_country,
    COALESCE(k."KTOKD", '') AS account_group,
    COALESCE(ag."TXT30", k."KTOKD", '') AS account_group_desc,
    COALESCE(k."KUKLA", '') AS customer_classification,
    COALESCE(k."BRSCH", '') AS industry_code,
    k."ERDAT" AS master_created_date,
    COALESCE(k."ERNAM", '') AS created_by,
    COALESCE(k."AUFSD", '') AS order_block,
    COALESCE(k."FAKSD", '') AS billing_block,
    COALESCE(k."LIFSD", '') AS delivery_block,
    COALESCE(k."SPERR", kb.knb1_sperr, '') AS posting_block,
    COALESCE(k."LOEVM", kb.knb1_loevm, '') AS deletion_flag,
    COALESCE(k."STCD3", '') AS gstin_tax_no,
    COALESCE(vb.vkorg, kv."VKORG", '1000') AS sales_org,
    COALESCE(vb.vtweg, kv."VTWEG", '10') AS dist_channel,
    COALESCE(vb.spart, kv."SPART", '10') AS division,
    COALESCE(vb.kdgrp, kv."KDGRP", '') AS customer_group,
    COALESCE(vb.bzirk, kv."BZIRK", '') AS sales_district,
    COALESCE(vb.pltyp, kv."PLTYP", '') AS price_list_type,
    COALESCE(kv."KONDA", '') AS price_group,
    COALESCE(kv."KLABC", '') AS abc_class,
    COALESCE(kv."VKBUR", '') AS sales_office,
    COALESCE(kv."VKGRP", '') AS sales_group,
    COALESCE(vb.zterm, kv.knvv_zterm, kb.knb1_zterm, '') AS payment_terms,
    COALESCE(kb."AKONT", '') AS recon_account,
    COALESCE(kb."ZWELS", '') AS payment_methods,
    COALESCE(kv."ZCUSTOMER_GRP_DES", kv."ZCUSTOMER_GROUP", '') AS channel_segment_desc,
    COALESCE(kv."ZRURAL_URBAN", '') AS rural_urban_flag,
    COALESCE(kv."ZMETRO_NONMETRO", '') AS metro_tier_flag,
    COALESCE(kv."ZDISTRIBUTOR_TYP", '') AS distributor_type,
    COALESCE(kv."ZTIER", '') AS dealer_tier,
    COALESCE(vb.billing_doc_count, 0) AS billing_doc_count,
    COALESCE(vb.cancelled_or_return_bills, 0) AS cancelled_or_return_bills,
    COALESCE(vb.total_billed_net_value, 0.0) AS total_billed_net_value,
    COALESCE(vb.total_billed_tax_value, 0.0) AS total_billed_tax_value,
    COALESCE(vb.avg_bill_value, 0.0) AS avg_bill_value,
    COALESCE(vb.max_bill_value, 0.0) AS max_bill_value,
    COALESCE(vb.std_bill_value, 0.0) AS std_bill_value,
    vb.first_billing_date,
    vb.last_billing_date,
    COALESCE(vb.distinct_billing_days, 0) AS distinct_billing_days,
    COALESCE(vb.active_billing_months, 0) AS active_billing_months,
    COALESCE(vo.sales_order_count, 0) AS sales_order_count,
    COALESCE(vo.total_order_net_value, 0.0) AS total_order_net_value,
    COALESCE(vo.avg_order_value, 0.0) AS avg_order_value,
    vo.first_order_date,
    vo.last_order_date,
    COALESCE(vo.blocked_order_count, 0) AS blocked_order_count,
    COALESCE(ar.open_ar_item_count, 0) AS open_ar_item_count,
    COALESCE(ar.open_ar_balance, 0.0) AS open_ar_balance,
    ar.last_ar_posting_date,
    COALESCE(ar.avg_ar_pending_days, 0.0) AS avg_ar_pending_days
FROM active_customers ac
LEFT JOIN "HANA_P3"."KNA1" k ON ac.kunnr = k."KUNNR"
LEFT JOIN vbrk_agg vb ON ac.kunnr = vb.kunnr
LEFT JOIN vbak_agg vo ON ac.kunnr = vo.kunnr
LEFT JOIN bsid_agg ar ON ac.kunnr = ar.kunnr
LEFT JOIN knvv_primary kv ON ac.kunnr = kv."KUNNR"
LEFT JOIN knb1_primary kb ON ac.kunnr = kb."KUNNR"
LEFT JOIN "HANA_P3"."T077X" ag ON k."KTOKD" = ag."KTOKD" AND ag."SPRAS" = 'E'
LEFT JOIN "HANA_P3"."T001" comp ON COALESCE(vb.bukrs, kb."BUKRS", '1000') = comp."BUKRS";
"""

print("Executing full SQL query into pandas...")
df = pd.read_sql(query, engine)
print(f"Extraction completed in {time.time() - t0:.2f}s! Shape: {df.shape}")
print(df.head(5))
