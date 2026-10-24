import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import pandas as pd
import numpy as np
from src.db_connector import load_config, get_db_engine
from scratch.test_fast_query import build_fast_query

config = load_config()
engine = get_db_engine(config)

print("Starting full live extraction across HANA_P2 and HANA_P3 (limit=None)...")
t0 = time.time()

# 1. Extract HANA_P2
print("[1/2] Extracting HANA_P2...")
t_p2 = time.time()
q_p2 = build_fast_query("HANA_P2", limit=None)
df_p2 = pd.read_sql(q_p2, engine)
print(f"HANA_P2 completed in {time.time()-t_p2:.2f}s! Rows: {len(df_p2):,}")

# 2. Extract HANA_P3
print("[2/2] Extracting HANA_P3...")
t_p3 = time.time()
q_p3 = build_fast_query("HANA_P3", limit=None)
df_p3 = pd.read_sql(q_p3, engine)
print(f"HANA_P3 completed in {time.time()-t_p3:.2f}s! Rows: {len(df_p3):,}")

# 3. Fast consolidation
print("[3/3] Consolidating multi-schema vendors...")
t_merge = time.time()

# Schema tracking sets
p2_v = set(df_p2['vendor_id'])
p3_v = set(df_p3['vendor_id'])
both_v = p2_v & p3_v
print(f"Vendor overlap: P2 only = {len(p2_v - both_v):,}, P3 only = {len(p3_v - both_v):,}, Both = {len(both_v):,}")

combined = pd.concat([df_p2, df_p3], ignore_index=True)
del df_p2
del df_p3
import gc
gc.collect()

# Ensure datetime types
date_cols = ['master_created_date', 'first_po_date', 'last_po_date', 'first_invoice_date', 'last_invoice_date', 'last_ap_posting_date']
for c in date_cols:
    if c in combined.columns:
        combined[c] = pd.to_datetime(combined[c], errors='coerce', dayfirst=True)

sum_cols = [
    'po_count', 'po_item_count', 'deleted_po_count', 'deleted_po_item_count',
    'return_po_item_count', 'delivery_completed_items', 'final_invoiced_items',
    'total_po_net_value', 'total_po_dunning_reminders', 'active_po_months',
    'invoice_count', 'total_invoice_gross_value', 'total_unplanned_deliv_costs',
    'total_invoice_tax_value', 'payment_blocked_invoices', 'reversed_invoices',
    'credit_memo_or_adj_count', 'mrm_blocked_invoice_count', 'open_ap_item_count',
    'open_ap_balance', 'blocked_ap_item_count', 'inspection_lot_count',
    'total_inspected_qty', 'total_accepted_qty', 'total_rejected_blocked_qty',
    'bank_account_count'
]
max_cols = [
    'max_po_item_value', 'max_invoice_value', 'avg_po_item_value',
    'avg_invoice_value', 'avg_gr_processing_days', 'avg_overdelivery_tol',
    'avg_ap_pending_days', 'std_po_item_value', 'std_invoice_value',
    'last_po_date', 'last_invoice_date', 'last_ap_posting_date'
]
min_cols = ['master_created_date', 'first_po_date', 'first_invoice_date']

text_cols = [
    'company_code', 'company_name', 'company_city', 'company_country', 'currency',
    'vendor_name', 'vendor_city', 'postal_code', 'vendor_region', 'vendor_country',
    'account_group', 'account_group_desc', 'industry_code', 'created_by',
    'posting_block', 'purchasing_block', 'quality_block', 'payment_block',
    'deletion_flag', 'gstin_tax_no', 'pan_number', 'alt_payee_allowed',
    'one_time_vendor_flag', 'purchasing_org', 'purchasing_group', 'primary_po_type',
    'payment_terms', 'recon_account', 'payment_methods', 'dup_invoice_check_flag',
    'withholding_tax_code', 'cash_mgmt_group'
]

# Numeric casting
for c in sum_cols + ['max_po_item_value', 'max_invoice_value', 'avg_po_item_value', 'avg_invoice_value', 'avg_gr_processing_days', 'avg_overdelivery_tol', 'avg_ap_pending_days', 'std_po_item_value', 'std_invoice_value']:
    if c in combined.columns:
        combined[c] = pd.to_numeric(combined[c], errors='coerce').fillna(0.0)

agg_dict = {}
for c in sum_cols:
    if c in combined.columns:
        agg_dict[c] = 'sum'
for c in max_cols:
    if c in combined.columns:
        agg_dict[c] = 'max'
for c in min_cols:
    if c in combined.columns:
        agg_dict[c] = 'min'
for c in text_cols:
    if c in combined.columns:
        agg_dict[c] = 'last'

consolidated = combined.groupby('vendor_id', as_index=False).agg(agg_dict)
del combined
gc.collect()

# Vectorized schema assignment
conds = [
    consolidated['vendor_id'].isin(both_v),
    consolidated['vendor_id'].isin(p2_v)
]
consolidated['source_schema'] = np.select(conds, ['HANA_P2, HANA_P3', 'HANA_P2'], default='HANA_P3')

print(f"Consolidation completed in {time.time()-t_merge:.2f}s!")
print(f"TOTAL UNIQUE VENDORS: {len(consolidated):,}")
print("Source schema distribution:", consolidated['source_schema'].value_counts().to_dict())
print(f"TOTAL PIPELINE EXTRACTION TIME: {time.time()-t0:.2f}s")
