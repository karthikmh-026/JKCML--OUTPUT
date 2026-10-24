import psycopg2

conn = psycopg2.connect(host='192.168.1.44', port=5432, dbname='JKC_Input', user='postgres', password='postgres')
cur = conn.cursor()

schema = 'HANA_P2'
cur.execute(f"""
EXPLAIN
    WITH mara_pool AS (
        SELECT
            a."MATNR", a."ERSDA", a."ERNAM", a."LAEDA", a."AENAM",
            a."VPSTA", a."PSTAT", a."LVORM", a."MTART", a."MBRSH",
            a."MATKL", a."BISMT", a."MEINS", a."BSTME", a."SPART",
            a."PRDHA", a."GROES", a."WRKST", a."NORMT", a."TRAGR",
            a."XCHPF", a."MSTAE", a."BRGEW", a."NTGEW", a."GEWEI",
            a."VOLUM", a."VOLEH"
        FROM "{schema}"."MARA" a
        WHERE a."MATNR" IS NOT NULL AND a."MATNR" != ''
    ),
    marc_primary AS (
        SELECT DISTINCT ON (c."MATNR")
            c."MATNR", c."WERKS",
            c."PSTAT" AS marc_pstat,
            c."LVORM" AS marc_lvorm,
            c."MMSTA" AS plant_mat_status,
            c."MAABC" AS abc_indicator,
            c."EKGRP" AS purchasing_group,
            c."DISMM" AS mrp_type,
            c."DISPO" AS mrp_controller,
            c."PLIFZ" AS planned_deliv_days,
            c."WEBAZ" AS gr_processing_days,
            c."BESKZ" AS procurement_type,
            c."PRCTR" AS profit_center,
            c."LADGR" AS loading_group
        FROM "{schema}"."MARC" c
        JOIN mara_pool mp ON c."MATNR" = mp."MATNR"
        ORDER BY c."MATNR", c."WERKS" ASC
    ),
    ekpo_agg AS (
        SELECT
            p."MATNR",
            COUNT(*) AS po_line_count,
            SUM(COALESCE(p."NETWR", 0.0)) AS total_po_spend,
            AVG(CASE WHEN COALESCE(p."NETPR", 0.0) > 0 THEN p."NETPR" ELSE NULL END) AS avg_po_net_price,
            MAX(p."TXZ01") AS sample_po_short_text,
            MAX(p."MATKL") AS po_material_group,
            COUNT(DISTINCT p."MATKL") AS distinct_po_matkl_cnt
        FROM "{schema}"."EKPO" p
        JOIN mara_pool mp ON p."MATNR" = mp."MATNR"
        WHERE p."MATNR" IS NOT NULL AND p."MATNR" != ''
          AND (p."LOEKZ" IS NULL OR p."LOEKZ" = '')
        GROUP BY p."MATNR"
    )
    SELECT COUNT(*) FROM mara_pool a
    JOIN marc_primary c ON a."MATNR" = c."MATNR"
    LEFT JOIN ekpo_agg po ON a."MATNR" = po."MATNR";
""")
for r in cur.fetchall()[:25]:
    print(r[0])
