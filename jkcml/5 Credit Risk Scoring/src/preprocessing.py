"""
Preprocessing and Feature Engineering Module - Use Case 5: Credit Risk Scoring
Transforms raw SAP customer master, accounts receivable (BSID/BSAD), sales orders (VBAK),
and billing documents (VBRK) into 32 domain-specific Credit Risk and Delinquency features.
Scales features with RobustScaler for 100% unsupervised clustering and default estimation.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


CREDIT_RISK_FEATURE_COLS = [
    "credit_risk_index",             # 0: Composite credit risk metric
    "solvency_score",                # 1: Solvency & clearing reliability score
    "overdue_ratio",                 # 2: Overdue AR balance / Total open AR
    "max_overdue_days",              # 3: Max days past due on open invoices
    "avg_overdue_days",              # 4: Average days past due
    "severe_overdue_ratio",          # 5: Ratio of AR overdue > 90 days
    "open_ar_log_val",               # 6: Log1p of open AR exposure
    "overdue_ar_log_val",            # 7: Log1p of overdue AR exposure
    "blocked_ar_rate",               # 8: Proportion of open items with payment blocks
    "blocked_balance_ratio",         # 9: Ratio of blocked balance to total open balance
    "avg_clearing_delay_days",       # 10: Historical settlement delay (clearing date - due date)
    "max_clearing_delay_days",       # 11: Longest clearing delay observed historically
    "late_settlement_rate",          # 12: Ratio of cleared items paid after due date
    "cleared_to_open_ratio",         # 13: Total cleared volume / Open AR balance
    "credit_limit_utilization",      # 14: Open AR exposure / Credit limit
    "is_credit_limit_breached",      # 15: Flag if open exposure exceeds credit limit
    "billing_volatility_ratio",      # 16: Std dev / Mean of billing invoices
    "order_to_billing_ratio",        # 17: Order net amount / Billed net amount
    "blocked_order_rate",            # 18: Proportion of blocked sales orders
    "return_memo_rate",              # 19: Proportion of credit memos / returns
    "has_central_posting_block",     # 20: KNA1 SPERR flag
    "has_cocode_posting_block",      # 21: KNB1 SPERR flag
    "has_deletion_flag",             # 22: LOEVM flag
    "customer_tenure_years",         # 23: Age of customer relationship in years
    "payment_terms_risk_score",      # 24: Ordinal risk score of payment terms
    "open_ar_item_density",          # 25: Open items per active transaction
    "unsecured_credit_gap",          # 26: Log1p of unhedged exposure over limit
    "historical_default_signal",     # 27: Severity score from historical late payments
    "settlement_velocity_index",     # 28: Cleared docs / Total transaction docs
    "high_dunning_exposure_flag",    # 29: Flag if overdue > 45 days or blocked items exist
    "cash_flow_strain_ratio",        # 30: Overdue balance relative to total business volume
    "distress_momentum_score"        # 31: Multi-factor distress indicator
]


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data and engineers 32 domain features for unsupervised credit risk scoring.
    Also builds the objective Basel II / IFRS 9 post-hoc benchmark exception label for accuracy checking.
    """
    df = df.copy()

    # Numeric safety conversions
    num_cols = [
        "credit_limit", "bill_of_exchange_limit", "allowed_excess_days",
        "open_ar_item_count", "open_ar_balance", "overdue_ar_balance",
        "severe_overdue_balance", "max_overdue_days", "avg_overdue_days",
        "blocked_ar_item_count", "blocked_ar_balance", "cleared_doc_count",
        "total_cleared_volume", "avg_clearing_delay_days", "max_clearing_delay_days",
        "late_settlement_count", "billing_doc_count", "total_billed_amount",
        "avg_bill_value", "std_bill_value", "return_memo_count",
        "sales_order_count", "total_order_amount", "blocked_order_count"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 1. Overdue & Aging Ratios
    pos_open_ar = np.maximum(0.0, df["open_ar_balance"].values)
    df["open_ar_pos"] = pos_open_ar
    df["overdue_ratio"] = np.clip(df["overdue_ar_balance"].values / np.maximum(100.0, pos_open_ar), 0.0, 1.0)
    df["severe_overdue_ratio"] = np.clip(df["severe_overdue_balance"].values / np.maximum(100.0, pos_open_ar), 0.0, 1.0)
    df["open_ar_log_val"] = np.log1p(pos_open_ar)
    df["overdue_ar_log_val"] = np.log1p(np.maximum(0.0, df["overdue_ar_balance"].values))

    # 2. Blocked AR & Disputes
    df["blocked_ar_rate"] = np.clip(df["blocked_ar_item_count"].values / np.maximum(1.0, df["open_ar_item_count"].values), 0.0, 1.0)
    df["blocked_balance_ratio"] = np.clip(df["blocked_ar_balance"].values / np.maximum(100.0, pos_open_ar), 0.0, 1.0)

    # 3. Historical Clearing Delays (BSAD)
    cleared_cnt = np.maximum(0.0, df["cleared_doc_count"].values)
    df["late_settlement_rate"] = np.clip(df["late_settlement_count"].values / np.maximum(1.0, cleared_cnt), 0.0, 1.0)
    df["cleared_to_open_ratio"] = np.clip(np.maximum(0.0, df["total_cleared_volume"].values) / np.maximum(100.0, pos_open_ar), 0.0, 50.0)

    # 4. Credit Limit & Utilization
    eff_credit_limit = np.maximum(50000.0, df["credit_limit"].values)  # default floor if unassigned
    df["credit_limit_utilization"] = np.clip(pos_open_ar / eff_credit_limit, 0.0, 10.0)
    df["is_credit_limit_breached"] = (pos_open_ar > eff_credit_limit).astype(float)
    df["unsecured_credit_gap"] = np.log1p(np.maximum(0.0, pos_open_ar - eff_credit_limit))

    # 5. Billing & Order Friction
    bill_cnt = np.maximum(0.0, df["billing_doc_count"].values)
    avg_bill = np.maximum(1.0, df["avg_bill_value"].values)
    df["billing_volatility_ratio"] = np.clip(df["std_bill_value"].values / avg_bill, 0.0, 10.0)
    df["order_to_billing_ratio"] = np.clip(df["total_order_amount"].values / np.maximum(100.0, df["total_billed_amount"].values), 0.0, 10.0)
    df["blocked_order_rate"] = np.clip(df["blocked_order_count"].values / np.maximum(1.0, df["sales_order_count"].values), 0.0, 1.0)
    df["return_memo_rate"] = np.clip(df["return_memo_count"].values / np.maximum(1.0, bill_cnt), 0.0, 1.0)

    # 6. Master Risk Flags
    df["has_central_posting_block"] = df["central_posting_block"].fillna("").astype(str).str.strip().ne("").astype(float)
    df["has_cocode_posting_block"] = df["co_code_posting_block"].fillna("").astype(str).str.strip().ne("").astype(float)
    df["has_deletion_flag"] = (
        df["central_deletion_flag"].fillna("").astype(str).str.strip().ne("") |
        df["co_code_deletion_flag"].fillna("").astype(str).str.strip().ne("")
    ).astype(float)

    # 7. Customer Tenure
    today = datetime.now()
    def parse_tenure(erdat):
        if pd.isna(erdat) or not str(erdat).strip():
            return 3.0
        try:
            d = pd.to_datetime(erdat)
            return max(0.1, (today - d).days / 365.25)
        except Exception:
            return 3.0

    df["customer_tenure_years"] = df["customer_created_date"].apply(parse_tenure)

    # 8. Payment Terms Risk Score
    def term_risk(term):
        term = str(term).upper().strip()
        if any(k in term for k in ["IMM", "0001", "CASH", "ADV"]):
            return 0.1  # immediate cash / advance
        elif "15" in term or "07" in term:
            return 0.3
        elif "30" in term or "IN09" in term:
            return 0.5
        elif "45" in term or "60" in term:
            return 0.7
        elif "90" in term or "120" in term:
            return 0.95
        return 0.5

    df["payment_terms_risk_score"] = df["payment_terms"].apply(term_risk)

    # 9. Additional Behavioral & Density Signals
    total_tx_docs = bill_cnt + cleared_cnt + df["open_ar_item_count"].values
    df["open_ar_item_density"] = np.clip(df["open_ar_item_count"].values / np.maximum(1.0, total_tx_docs), 0.0, 1.0)
    df["settlement_velocity_index"] = np.clip(cleared_cnt / np.maximum(1.0, total_tx_docs), 0.0, 1.0)

    # Historical late payment severity
    df["historical_default_signal"] = np.clip(
        (df["avg_clearing_delay_days"].values / 30.0) * 0.5 +
        (df["max_clearing_delay_days"].values / 60.0) * 0.5,
        0.0, 5.0
    )

    df["high_dunning_exposure_flag"] = (
        (df["max_overdue_days"].values > 45) |
        (df["blocked_ar_item_count"].values > 0)
    ).astype(float)

    total_bus_vol = np.maximum(1000.0, df["total_cleared_volume"].values + df["total_billed_amount"].values)
    df["cash_flow_strain_ratio"] = np.clip(df["overdue_ar_balance"].values / total_bus_vol, 0.0, 5.0)

    df["distress_momentum_score"] = (
        0.30 * df["overdue_ratio"].values +
        0.25 * df["severe_overdue_ratio"].values +
        0.20 * df["blocked_order_rate"].values +
        0.15 * df["has_central_posting_block"].values +
        0.10 * df["is_credit_limit_breached"].values
    )

    # Composite Solvency & Credit Risk Indices
    df["credit_risk_index"] = (
        0.25 * df["overdue_ratio"].values +
        0.20 * np.clip(df["max_overdue_days"].values / 90.0, 0.0, 2.0) +
        0.15 * df["severe_overdue_ratio"].values +
        0.15 * df["credit_limit_utilization"].values.clip(0, 2) / 2.0 +
        0.10 * df["blocked_ar_rate"].values +
        0.10 * df["distress_momentum_score"].values +
        0.05 * df["has_central_posting_block"].values
    )

    df["solvency_score"] = np.clip(
        1.0 - (
            0.35 * df["credit_risk_index"].values +
            0.25 * df["late_settlement_rate"].values +
            0.20 * (1.0 - df["settlement_velocity_index"].values) +
            0.20 * np.clip(df["avg_clearing_delay_days"].values / 45.0, 0.0, 1.0)
        ),
        0.0, 1.0
    )

    # 10. Objective Basel II / IFRS 9 Post-Hoc Ground Truth Benchmark Exception
    # (Used strictly for ranking the 5 unsupervised models and checking accuracy)
    # Standard credit risk impairment conditions:
    # 1. Delinquency > 60 days past due with overdue balance > 10,000 INR
    # 2. Or severe overdue > 90 days past due
    # 3. Or posting block / bankruptcy / legal block with active unpaid debt
    # 4. Or credit limit breached > 1.5x with overdue debt
    is_impaired = (
        ((df["max_overdue_days"] >= 60) & (df["overdue_ar_balance"] > 10000.0)) |
        (df["severe_overdue_balance"] > 25000.0) |
        ((df["has_central_posting_block"] == 1.0) & (pos_open_ar > 5000.0)) |
        ((df["has_cocode_posting_block"] == 1.0) & (pos_open_ar > 5000.0)) |
        ((df["credit_limit_utilization"] >= 1.5) & (df["overdue_ar_balance"] > 10000.0)) |
        (df["distress_momentum_score"] >= 0.50)
    ).astype(int)

    df["is_credit_risk_exception"] = is_impaired

    return df


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> dict:
    """
    Extracts the 32 credit risk features, splits into 80/20 train/test sets,
    and applies RobustScaler (fitted strictly on training set).
    Returns dictionary with scaled arrays and feature metadata.
    """
    X = df[CREDIT_RISK_FEATURE_COLS].values
    y = df["is_credit_risk_exception"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": CREDIT_RISK_FEATURE_COLS
    }
